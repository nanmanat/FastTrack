import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class VGG16WithResidual(nn.Module):
    def __init__(self, num_classes=1):
        super(VGG16WithResidual, self).__init__()
        self.vgg16 = models.vgg16(weights=None)
        self.features = self.vgg16.features

        # Adding Residual Blocks after VGG16 layers
        self.residual_block = nn.Sequential(
            ResidualBlock(512, 512),
            ResidualBlock(512, 512)
        )

        # Replace classifier with custom classifier
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),  # Predict class scores for each pixel
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)  # Upsample to match input size
        )

    def forward(self, x):
        x = self.features(x)  # Features from VGG16
        x = self.residual_block(x)  # Add residual blocks
        x = self.segmentation_head(x)  # Segmentation head
        return x


class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = self.parse_xml(mask_path)

        if self.transform:
            image = self.transform(image)

        return image, mask

    def parse_xml(self, xml_path):
        # Implement XML parsing to get the mask or annotations
        # For simplicity, returning a dummy mask
        return np.zeros((224, 224))  # Adjust based on your mask format


def get_image_and_mask_paths(image_folder, mask_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.xml')]

    image_paths = [os.path.join(image_folder, f) for f in image_files]
    mask_paths = [os.path.join(mask_folder, f) for f in mask_files]

    return image_paths, mask_paths


def compute_precision_recall(predictions, ground_truths, num_classes=1):
    # Flatten lists of predictions and ground truths
    y_true = np.array(ground_truths).flatten()
    y_pred = np.array(predictions).flatten()

    # Compute precision and recall for multi-class classification
    precision = precision_score(y_true, y_pred, average='macro', labels=range(num_classes))
    recall = recall_score(y_true, y_pred, average='macro', labels=range(num_classes))

    return precision, recall


def main():
    image_folder = "ImageSets/JPEGImages"
    mask_folder = "Annotations"

    image_paths, mask_paths = get_image_and_mask_paths(image_folder, mask_folder)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(image_paths, mask_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = VGG16WithResidual(num_classes=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    all_preds = []
    all_labels = []

    # Training Loop (simplified)
    for epoch in range(10):  # number of epochs
        epoch_preds = []
        epoch_labels = []

        count = 0

        for images, masks in dataloader:
            optimizer.zero_grad()
            outputs = model(images)

            # outputs: (batch_size, num_classes, H, W)
            # masks: (batch_size, H, W)

            # Reshape outputs and masks for loss computation
            outputs = outputs.permute(0, 2, 3, 1).contiguous()  # Shape: (batch_size, H, W, num_classes)
            outputs = outputs.view(-1, outputs.size(-1))  # Shape: (batch_size * H * W, num_classes)
            masks = masks.view(-1)  # Shape: (batch_size * H * W)
            masks = torch.reshape(masks, [len(masks), 1])

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            # Predictions for precision/recall
            _, predicted = torch.max(outputs, 1)
            epoch_preds.extend(predicted.cpu().numpy())
            epoch_labels.extend(masks.cpu().numpy())
            count += 1
            print((count*100/len(dataloader)), "%", count)


        # Calculate precision and recall for this epoch
        precision, recall = compute_precision_recall(epoch_preds, epoch_labels, num_classes=2)
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Precision: {precision:.4f}, Recall: {recall:.4f}')


if __name__ == "__main__":
    main()


