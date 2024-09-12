import torch
import torchvision
from torch import nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader

import torch
import torchvision
from torch import nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
tranform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = torchvision.datasets.ImageFolder(root='dataset/train', transform=tranform)
test_dataset = torchvision.datasets.ImageFolder(root='dataset/test', transform=tranform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class Custom_VGG16(nn.Module):
    def __init__(self):
        super(Custom_VGG16, self).__init__()
        self.ConvSet_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.ConvSet_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.ConvSet_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.ConvSet_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.ConvSet_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.FC_Layers = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 131),
        )

    def forward(self, x):
        out = self.ConvSet_1(x)
        out = self.ConvSet_2(out)
        out = self.ConvSet_3(out)
        out = self.ConvSet_4(out)
        out = self.ConvSet_5(out)
        out = out.reshape(out.shape[0], -1)
        out = self.FC_Layers(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Custom_VGG16()
model = model.to(device=device)

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

num_epochs = 100
for epoch in range(num_epochs):
    loss_ep = 0
    for images, labels in train_dataloader:
        images = images.to(device=device)
        labels = labels.to(device=device)
        optimizer.zero_grad()
        scores = model(images)
        loss = criterion(scores,labels)
        loss.backward()
        optimizer.step()
        loss_ep += loss.item()
    print ('Epoch [{}/{}] :::: Loss: {:.4f}'.format(epoch+1, num_epochs, loss_ep/len(train_dataloader)))

torch.save(model.state_dict(), "Custom_VGG16.pt")

model = Custom_VGG16()
model.load_state_dict(torch.load("Custom_VGG16.pt"))
model = model.to(device=device)
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataloader:
        images = images.to(device=device)
        labels = labels.to(device=device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the Model on the {} Test Images: {}%'.format(len(test_dataset), 100*correct /total))

