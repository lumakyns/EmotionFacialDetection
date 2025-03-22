import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from math import prod
from itertools import product
from PIL import Image
import os
import pandas as pd
import numpy as np
import csv

"""
---------------------------------------------------
Author: Lumakyns
Date: 2025-03-20
Description: Convolutional Neural Network for detecting emotions in images.
Version: 1.0
---------------------------------------------------
"""


## CNN CLASS
class CNN(nn.Module):
    def __init__(self, parameters, epochs=10):
        super(CNN, self).__init__()

        # CNN Model
        self.model = nn.Sequential(
            nn.Conv2d(1, parameters["filter1"], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(parameters["filter1"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(parameters["filter1"], parameters["filter2"], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(parameters["filter2"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(parameters["filter2"] * 8 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(parameters["dropout_rate"]),
            nn.Linear(128, 7)
        )

        # Parameters
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=parameters["learning_rate"])

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

# Paths
train_csv_path = os.path.join(os.path.dirname(__file__), "../../split/split-csv/train.csv")
test_csv_path = os.path.join(os.path.dirname(__file__), "../../split/split-csv/test.csv")
img_dir = os.path.join(os.path.dirname(__file__), "../../../rsrc/facial_expressions/images")
output_dir = os.path.join(os.path.dirname(__file__), "../../results/")

# Load CSV's
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

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

# Datasets
train_dataset = Extract(train_df, img_dir)
test_dataset = Extract(test_df, img_dir)

# Grid Search
# NOTE: Feel free to change these values as you please, A CSV will automatically be deposited in ../results
param_grid = {
    'learning_rate': [0.001, 0.0001],
    'batch_size': [32, 64],
    'filter1': [16, 32],
    'filter2': [32, 64],
    'dropout_rate': [0.3, 0.5]
}

keys = param_grid.keys()
vals = param_grid.values()
num_combinations = prod(len(v) for v in vals)

# Go through all parameter combinations
for i, combination in enumerate(product(*vals)):
    params = dict(zip(keys, combination))
    
    print(f"Running configuration {i+1}/{num_combinations}: {params}\r", end="")
    
    # PyTorch Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # Model
    model = CNN(parameters=params, epochs=32)
    model.fit(train_loader, test_loader)
    
    # Output CSV
    run_id = f"lr-{params['learning_rate']}_bs-{params['batch_size']}_f1-{params['filter1']}_f2-{params['filter2']}_dr-{params['dropout_rate']}"
    csv_path = os.path.join(output_dir, run_id)
    
    # Combine losses
    rows = zip(model.history['train_loss'], model.history['test_loss'])
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(['train_loss', 'test_loss'])
        writer.writerows(rows)
    
    
