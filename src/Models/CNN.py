import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.model_selection import train_test_split

## CNN CLASS
class CNN(nn.Module):
    def __init__(self, learning_rate=0.001, epochs=10):
        super(CNN, self).__init__()

        # CNN Model
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

        # Parameters
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Automatically use GPU/CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Metrics
        self.history = {'train_loss': [], 'val_accuracy': []}

    def fit(self, train_loader, val_loader=None):
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
            
            # Validation phase
            if val_loader:
                val_accuracy = self.score(val_loader)
                self.history['val_accuracy'].append(val_accuracy)
        
        return self
    
    def predict(self, data_loader):
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                all_predictions.extend(predictions.cpu().numpy())
        
        return all_predictions
