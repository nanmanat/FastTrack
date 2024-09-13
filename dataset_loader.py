import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, txt_file, img_dir, transform=None):
        # Load the dataset
        self.data = []
        self.img_dir = img_dir
        with open(txt_file, 'r') as f:
            for line in f:
                filename, label, _ = line.strip().split()
                self.data.append((filename, int(label)))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label = self.data[idx]
        # Construct the full image path
        img_path = os.path.join(self.img_dir, filename)
        # Open the image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataset(file_name):
    # Define any transformations you want to apply to the images (e.g., resizing, normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image
        transforms.ToTensor(),  # Convert image to PyTorch tensor
    ])

    # Path to the dataset file and image directory
    txt_file = file_name  # Replace with the actual file path
    img_dir = 'JPEGImages'  # Replace with the directory where images are stored

    # Initialize the dataset and dataloader
    dataset = ImageDataset(txt_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    return dataloader


def main():
    # Load the dataset for a specific fold
    fold = '1'
    dataloader = get_dataset(fold)

    # Iterate over the dataset
    for images, labels in dataloader:
        print(images.shape, labels)


if __name__ == '__main__':
    main()