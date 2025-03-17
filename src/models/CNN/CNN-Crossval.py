## INCLUDES

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from PIL import Image

import os
import pandas as pd
import numpy as np
from itertools import product
import csv


## CNN CLASS

class CNN(nn.Module):
    def __init__(self, epochs=10):
        super(CNN, self).__init__()

        # CNN Model
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7)
        )

        # Parameters
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Automatically use GPU/CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Metrics
        self.history = {'train_loss': [], 'test_loss': []}

    def fit(self, train_loader, test_loader=None):
        for epoch in range(self.epochs):
            running_loss = 0.0
            
            # Training phase
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            # Record average training loss for this epoch
            epoch_loss = running_loss / len(train_loader)
            self.history['train_loss'].append(epoch_loss)
            
            # Testing phase
            if test_loader:
                test_accuracy = self.score(test_loader)
                self.history['test_loss'].append(test_accuracy)
        
        return self
    
    def score(self, data_loader):
        self.model.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accuracy = correct / total
        error_rate = 1 - accuracy
        return error_rate


## DATA PROCESSING

# Labels to use
labels_using = ['happiness', 'neutral', 'sadness', 'anger', 'surprise', 'disgust', 'fear']
label_map = {
    'happiness': 0,
    'neutral': 1,
    'sadness': 2,
    'anger': 3,
    'surprise': 4,
    'disgust': 5,
    'fear': 6
}

# Transformation object (for images)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Grayscale(), # Images should already be grayscale
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset extraction class for use with PyTorch
class Extract(Dataset):
    def __init__(self, dataframe, img_dir):
        self.dataframe = dataframe
        self.img_dir = img_dir
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0])
        label = self.dataframe.iloc[idx, 1]
        image = Image.open(img_name)
        
        image = transform(image)

        label = torch.tensor(label_map[label])
        
        return image, label

# Paths
fold_dir = os.path.join(os.path.dirname(__file__), "../luca-split/7fold-splits/")
img_dir = os.path.join(os.path.dirname(__file__), "../../rsrc/facial_expressions/images")


## RUN MODELS
crossval_error = []

# Go through all combinations of constants
for i in range(1, 8):
    print (f"[{i*"="}{(8-i)*" "}] :: Running fold {i}\r*", end="")
    
    # Load CSV's
    train_df = pd.read_csv(os.path.join(fold_dir, f"train_{i}.csv"))
    test_df = pd.read_csv(os.path.join(fold_dir, f"test_{i}.csv"))
    
    # Cross-validation dataset
    train_dataset = Extract(train_df, img_dir)
    test_dataset = Extract(test_df, img_dir)
    
    # PyTorch Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Model
    model = CNN(epochs=32)
    model.fit(train_loader, test_loader)
    
    crossval_error.append(model.history["test_loss"])
    
print(crossval_error)
print(np.mean(crossval_error))

    
    
