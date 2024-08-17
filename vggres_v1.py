import torch.nn as nn
import torchvision.models as models
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchmetrics

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class VGG16Modified(nn.Module):
    def __init__(self, num_classes=50):
        super(VGG16Modified, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights)
        self.features = vgg16.features

        # Add a residual block after certain layers
        self.residual_block = ResidualBlock(512, 512)

        # Modify classifier to predict bounding boxes
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes * 4),  # num_classes * 4 for bounding box coordinates (x, y, w, h)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.residual_block(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.reshape(-1, 4)  # Reshape to have each row correspond to a bounding box (x, y, w, h)

class ObjectDetectionDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, transform=None):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform
        self.images = sorted(os.listdir(image_folder))
        self.annotations = sorted(os.listdir(annotation_folder))

    def __len__(self):
        return len(self.images)

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.images[idx])
        annotation_path = os.path.join(self.annotation_folder, self.annotations[idx])

        image = Image.open(img_path).convert("RGB")
        boxes = self.parse_annotation(annotation_path)

        if self.transform:
            image = self.transform(image)

        return image, boxes


def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Compute intersection area
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Compute areas of the bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute union area
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    iou = inter_area / union_area
    return iou


def evaluate_model(model, dataloader, device):
    model.eval()
    ious = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            outputs = outputs.view(-1, 4)  # Ensure outputs are in the correct shape

            # Calculate IoU for each predicted and true box
            for i in range(outputs.size(0)):
                predicted_box = outputs[i].cpu().numpy()
                true_box = targets[i].cpu().numpy()

                iou = calculate_iou(predicted_box, true_box)
                ious.append(iou)

    mean_iou = sum(ious) / len(ious)
    return mean_iou


def collate_fn(batch):
    images, targets = zip(*batch)

    # Stack images as usual
    images = torch.stack(images, dim=0)

    # Pad targets to the maximum number of boxes (or to a fixed number)
    max_num_boxes = max(len(boxes) for boxes in targets)
    fixed_num_boxes = 100  # Example: set a fixed number of boxes
    padded_targets = []
    for boxes in targets:
        padded_boxes = torch.zeros((fixed_num_boxes, 4))
        padded_boxes[:len(boxes), :] = boxes[:fixed_num_boxes]
        padded_targets.append(padded_boxes)

    padded_targets = torch.stack(padded_targets, dim=0)

    return images, padded_targets


def calculate_iou(box1, box2):
    # Calculate the intersection
    reshaped_array = box2.reshape(4, 1)
    box1 = box1.reshape(4)
    box2 = reshaped_array.reshape(4)

    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate the union
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union = box1_area + box2_area - intersection

    return intersection / union

def calculate_precision_recall(pred_boxes, target_boxes, iou_threshold=0.5):
    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives

    for pred, target in zip(pred_boxes, target_boxes):
        ious = calculate_iou(pred.unsqueeze(1), target.unsqueeze(0))

        matches = ious > iou_threshold
        TP += matches.sum().item()
        FP += (matches == 0).sum().item()
        FN += (ious == 0).sum().item()

    precision = TP / (TP + FP + 1e-6)  # Add small value to prevent division by zero
    recall = TP / (TP + FN + 1e-6)

    return precision, recall

def main():
    # Hyperparameters
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 8
    num_classes = 50  # Single class for object detection

    # Paths to the dataset
    image_folder = "JPEGImages"
    annotation_folder = "Annotations"

    # Transformations
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])

    # Dataset and DataLoader
    dataset = ObjectDetectionDataset(image_folder=image_folder,
                                     annotation_folder=annotation_folder,
                                     transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    # Model, Loss Function, and Optimizer
    model = VGG16Modified(num_classes=num_classes).to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for bounding box regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize Mean Average Precision metric
    metric = MeanAveragePrecision()

    # Initialize precision and recall metrics
    precision_metric = torchmetrics.Precision(num_classes=1, average='none', task='binary' )
    recall_metric = torchmetrics.Recall(num_classes=1, average='none', task='binary')

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_precision = 0.0
        epoch_recall = 0.0

        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)

            # Move each target tensor to the device
            targets = [target.to(device) for target in targets]

            # Forward pass
            outputs = model(images)  # Shape: [batch_size, num_predictions, 4]

            # Resize outputs to match targets
            flattened_outputs = outputs.view(-1, 4)
            flattened_targets = torch.cat(targets).view(-1, 4)

            # Ensure output and target sizes match
            num_boxes = min(flattened_outputs.size(0), flattened_targets.size(0))
            flattened_outputs = flattened_outputs[:num_boxes, :]
            flattened_targets = flattened_targets[:num_boxes, :]

            # Compute loss
            loss = criterion(flattened_outputs, flattened_targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate precision and recall
            preds = flattened_outputs.argmax(dim=1)
            targ = flattened_targets.argmax(dim=1)

            # Normalize to [0, 1]
            min_val = flattened_targets.min()
            max_val = flattened_targets.max()
            normalized_tensor = (flattened_targets - min_val) / (max_val - min_val)

            # Scale to [-1, 0]
            scaled_tensor = normalized_tensor * -1

            precision, recall = calculate_precision_recall(flattened_outputs, scaled_tensor)
            epoch_precision += precision
            epoch_recall += recall

        avg_precision = epoch_precision / len(dataloader)
        avg_recall = epoch_recall / len(dataloader)

        # Print loss at the end of each epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "object_detection_model.pth")
    print("Model saved as object_detection_model.pth")


if __name__ == "__main__":
    device = torch.device("cuda")
    main()

def EvaluateModel():
    # Paths to the dataset
    image_folder = "JPEGImages"
    annotation_folder = "Annotations"

    # Transformations
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])

    # Dataset and DataLoader
    test_dataset = ObjectDetectionDataset(image_folder=image_folder,
                                          annotation_folder=annotation_folder,
                                          transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load the trained model
    model = VGG16Modified(num_classes=1)
    model.load_state_dict(torch.load("object_detection_model.pth"))
    model = model.to(device)

    # Evaluate the model
    mean_iou = evaluate_model(model, test_dataloader, device)
    print(f"Mean IoU: {mean_iou:.4f}")

#
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     EvaluateModel()