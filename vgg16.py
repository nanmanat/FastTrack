import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

# 1. SSD Model Definition
class VGG16Backbone(nn.Module):
    def __init__(self):
        super(VGG16Backbone, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights)
        self.features = nn.Sequential(*list(vgg.features.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x

class SSD(nn.Module):
    def __init__(self, num_classes=21):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.backbone = VGG16Backbone()

        self.extras = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True)
            )
        ])

        self.loc = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
        ])

        self.conf = nn.ModuleList([
            nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
        ])

    def forward(self, x):
        locs = []
        confs = []

        x = self.backbone(x)
        locs.append(self.loc[0](x))
        confs.append(self.conf[0](x))

        for i in range(len(self.extras)):
            x = self.extras[i](x)
            locs.append(self.loc[i+1](x))
            confs.append(self.conf[i+1](x))

        locs = [l.permute(0, 2, 3, 1).contiguous() for l in locs]
        confs = [c.permute(0, 2, 3, 1).contiguous() for c in confs]

        locs = torch.cat([l.view(l.size(0), -1) for l in locs], dim=1)
        confs = torch.cat([c.view(c.size(0), -1) for c in confs], dim=1)

        locs = locs.view(locs.size(0), -1, 4)
        confs = confs.view(confs.size(0), -1, self.num_classes)

        return locs, confs

# 2. Loss Function Definition
class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, threshold=0.5, neg_pos_ratio=3, alpha=1.0):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.conf_loss = nn.CrossEntropyLoss(reduction='none')
        self.loc_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, predictions, targets):
        loc_preds, conf_preds = predictions
        loc_targets, conf_targets = targets
        pos_mask = conf_targets > 0
        loc_loss = self.loc_loss(loc_preds[pos_mask], loc_targets[pos_mask])

        batch_size, num_priors, num_classes = conf_preds.size()
        conf_loss = self.conf_loss(conf_preds.view(-1, num_classes), conf_targets.view(-1))

        num_pos = pos_mask.sum().item()
        conf_loss = conf_loss.view(batch_size, -1)
        conf_loss[pos_mask] = 0
        _, loss_idx = conf_loss.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_neg = self.neg_pos_ratio * num_pos
        neg_mask = idx_rank < num_neg.unsqueeze(1)

        conf_loss = conf_loss[pos_mask | neg_mask].sum()
        loc_loss = loc_loss.sum()
        total_loss = (self.alpha * loc_loss + conf_loss) / num_pos

        return total_loss

# 3. VOCDataset Definition
class VOCDataset(Dataset):
    def __init__(self, root, image_set, transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform

        image_set_file = os.path.join(f'{image_set}.txt')
        with open(image_set_file) as f:
            self.image_ids = [x.strip() for x in f.readlines()]

        self.annotation_dir = os.path.join(self.root, 'Annotations')
        self.image_dir = os.path.join(self.root, 'JPEGImages')

        self.class_to_ind = {'fore': 1}
        self.classes = ['__background__', 'fore']

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        annotation_file = os.path.join(self.annotation_dir, f'{image_id}.xml')
        image_file = os.path.join(self.image_dir, f'{image_id}.jpg')

        tree = ET.parse(annotation_file)
        root = tree.getroot()

        image = Image.open(image_file).convert('RGB')

        boxes = []
        labels = []
        for obj in root.findall('object'):
            label = self.class_to_ind[obj.find('name').text.lower().strip()]
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if self.transform is not None:
            image, boxes = self.transform(image, boxes)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        return image, target

def transform(image, boxes):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    image = transform(image)
    return image, boxes

# 4. Training Function
def train_ssd(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loc_preds, conf_preds = model(images)
            loc_targets = torch.stack([t['boxes'] for t in targets])
            conf_targets = torch.stack([t['labels'] for t in targets])
            loss = criterion((loc_preds, conf_preds), (loc_targets, conf_targets))

            loss.backward()
            optimizer.step()

# 5. Main Function
def main():
    # Hyperparameters
    num_classes = 2  # Background + 'fore' class
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.001

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    transform_fn = transform
    dataset = VOCDataset(root='/', image_set='NNEW_trainval_1', transform=transform_fn)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Model, Loss, and Optimizer
    model = SSD(num_classes=num_classes)
    criterion = MultiBoxLoss(num_classes=num_classes)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Train the model
    train_ssd(model, train_loader, criterion, optimizer, num_epochs, device)

if __name__ == "__main__":
    main()