import datetime
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from shared_infrastructure import CustomImageDataset, train_model

#%%
# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)


        # Fully Connected Layers
        fc1_in_features = 1660224
        self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=5)

        # Output activitation function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # First Convolutional Block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Second Convolutional Block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten for Fully Connected Layers
        x = x.view(-1, self.fc1.in_features)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        # Output layer
        x = self.softmax(x)
        
        return x

if __name__ == '__main__':
    # Check if GPU is available and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}", flush=True)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Set batch size
    batch_size = 16

    # Number of training epochs
    epochs = 25

    print(f'BEGIN TRAINING: Regular data {datetime.datetime.now()}', flush=True)
    train_set = CustomImageDataset("./preprocessed/train/labels.csv", "preprocessed/train/specs")
    val_set = CustomImageDataset("./preprocessed/val/labels.csv", "preprocessed/val/specs")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # Create an instance of the SimpleCNN model and move it to the specified device (GPU if available)
    model = SimpleCNN().to(device)

    # Define the loss criterion (CrossEntropyLoss) and the optimizer (Adam) for training the model
    weights = torch.tensor([0.17768543*5, 0.1938815*5, 0.20993698*5, 0.14523569*5, 0.2732604*5]).to(device) # Weighted based on data availability (see 1 Data Analysis.ipynb)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model using the defined training function
    val_losses_simple = train_model(model, train_loader, val_loader, epochs, criterion, optimizer, batch_size)
    print(val_losses_simple, flush=True)

    print(f'BEGIN TRAINING: Augmented data {datetime.datetime.now()}', flush=True)
    train_set = CustomImageDataset("./preprocessed-augmented/train/labels.csv", "preprocessed/train/specs")
    val_set = CustomImageDataset("./preprocessed-augmented/val/labels.csv", "preprocessed/val/specs")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # Create an instance of the SimpleCNN model and move it to the specified device (GPU if available)
    model = SimpleCNN().to(device)

    # Define the loss criterion (CrossEntropyLoss) and the optimizer (Adam) for training the model
    weights = torch.tensor([0.17768543 * 5, 0.1938815 * 5, 0.20993698 * 5, 0.14523569 * 5, 0.2732604 * 5]).to(
        device)  # Weighted based on data availability (see 1 Data Analysis.ipynb)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model using the defined training function
    val_losses_simple = train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device, batch_size)
    print(val_losses_simple, flush=True)