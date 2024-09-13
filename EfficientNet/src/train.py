import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import precision_score, recall_score, f1_score

from tqdm.auto import tqdm

import dataset_loader
from model import build_model
from datasets import get_datasets, get_data_loaders
from utils import save_model, save_plots

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=20,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-pt', '--pretrained', action='store_true',
    help='Whether to use pretrained weights or not'
)
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.0001,
    help='Learning rate for training the model'
)
args = vars(parser.parse_args())

# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    all_preds = []
    all_labels = []

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

        # Store predictions and labels for metric calculation
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))

    # Calculate precision, recall, and F1 score
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, epoch_acc, precision, recall, f1


# Validation function.
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Store predictions and labels for metric calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

    # Calculate precision, recall, and F1 score
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, epoch_acc, precision, recall, f1


if __name__ == '__main__':
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets(args['pretrained'])
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    # Load the training and validation data loaders.
    for i in range(5):
        # train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)

        train_loader = dataset_loader.get_dataset('NNew_train_val_' + str(i) + '.txt')
        valid_loader = dataset_loader.get_dataset('NNew_test' + str(i) + '.txt')

        # Learning_parameters.
        lr = args['learning_rate']
        epochs = args['epochs']
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Computation device: {device}")
        print(f"Learning rate: {lr}")
        print(f"Epochs to train for: {epochs}\n")

        model = build_model(
            pretrained=args['pretrained'],
            fine_tune=True,
            num_classes=len(dataset_classes)
        ).to(device)

        # Total parameters and trainable parameters.
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")

        # Optimizer.
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Loss function.
        criterion = nn.CrossEntropyLoss()

        # Lists to keep track of losses and accuracies.
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        # Start the training.
        for epoch in range(epochs):
            print(f"[INFO]: Epoch {epoch+1} of {epochs}")
            train_epoch_loss, train_epoch_acc, train_epoch_precision, train_epoch_recall, train_epoch_f1 = train(model, train_loader,
                                                    optimizer, criterion)
            valid_epoch_loss, valid_epoch_acc, valid_epoch_precision, valid_epoch_recall, valid_epoch_f1 = validate(model, valid_loader,
                                                        criterion)
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)
            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            print('-'*50)
            time.sleep(5)

        # Save the trained model weights.
        save_model(epochs, model, optimizer, criterion, args['pretrained'])
        # Save the loss and accuracy plots.
        save_plots(train_acc, valid_acc, train_loss, valid_loss, args['pretrained'])
        print('TRAINING COMPLETE')

