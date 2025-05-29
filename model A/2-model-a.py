#%%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
# Matplotlib plots using https://stackoverflow.com/questions/37360568/python-organisation-of-3-subplots-with-matplotlib
from torcheval.metrics.functional import multiclass_f1_score
import matplotlib.gridspec as gridspec
from torcheval.metrics.aggregation.auc import AUC


#%%
# Set random seed for reproducibility
torch.manual_seed(42)
batch_size = 32

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to: {device}", flush=True)
#%%

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = torch.as_tensor(np.load(img_path))
        image = image.unsqueeze(0)
        label = one_hot(torch.as_tensor(self.img_labels.iloc[idx, 1]-1),5).unsqueeze(0).to(torch.float)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
#%%
val_set = CustomImageDataset("./preprocessed/val/labels.csv", "preprocessed/val/specs")
train_set = CustomImageDataset("preprocessed/train/labels.csv", "preprocessed/train/specs")
#%%
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
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
        fc1_in_features = 32 * 8 * 25 # need length of input audio before deciding this
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
#%%
# Training loop

# Training Loop with Validation
def train_model(model, train_loader, val_loader, epochs, criterion, optimizer):
    # Lists to store training and validation losses, and accuracies
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_AUCs = []
    val_AUCs = []


    metric = AUC()
    
    # Loop over epochs
    for epoch in range(epochs):
        # Set the model to training mode
        y_true = torch.tensor([]).to(device)
        y_true_num = torch.tensor([]).to(device)
        y_pred = torch.tensor([]).to(device)

        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for images, labels in train_loader:
            for image, label in zip(images,labels):
                image, label = image.to(device), label.to(device)
                optimizer.zero_grad()
                outputs = model(image)

                y_true = torch.cat((y_true,label),0)
                # y_true.add(label) # Store label
                y_pred = torch.cat((y_pred,outputs.data),0)
                # y_pred.add(outputs.data) # Store predictions

                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

                metric.update(outputs.data,label) # Update AUC

                _, predicted = torch.max(outputs.data, 1)
                # print(torch.max(label,1).indices[0])
                y_true_num = torch.cat((y_true_num,torch.tensor([torch.max(label,1).indices[0]]).to(device)),0)
                # print(y_true_num)
                outputs.data = torch.tensor([0.,0.,0.,0.,0.]).to(device)
                outputs.data[predicted] = 1.
                
                total_train += label.size(0)
                correct_train += (label == outputs.data).sum().item()//5


                
        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Calculate training accuracy
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Calculate training AUC
        train_AUC = metric.compute().item()
        train_AUCs.append(train_AUC)

        # Calculate training F1 score
        train_f1 = multiclass_f1_score(y_pred,y_true_num,num_classes=5)

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0


        # Reset AUC metric
        metric.reset()

        y_true_val = torch.tensor([]).to(device)
        y_true_num_val = torch.tensor([]).to(device)
        y_pred_val = torch.tensor([]).to(device)

        # Validation without gradient computation
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                for image, label in zip(val_images,val_labels):
                    image, label = image.to(device), label.to(device)
                    val_outputs = model(image)
                    val_loss = criterion(val_outputs, label)
                    total_val_loss += val_loss.item()
 
                    metric.update(val_outputs.data,label) # Update AUC
                    y_true_val = torch.cat((y_true,label),0)
                    y_pred_val = torch.cat((y_pred_val,val_outputs.data),0)
                    # print(val_outputs.data)

                    _, predicted_val = torch.max(val_outputs.data, 1)
                    y_true_num_val = torch.cat((y_true_num_val,torch.tensor([torch.max(label,1).indices[0]]).to(device)),0)
                    # print(label)

                    val_outputs.data = torch.tensor([0.,0.,0.,0.,0.]).to(device)
                    val_outputs.data[predicted_val[0]] = 1.

                    total_val += label.size(0)
                    correct_val += (label == val_outputs.data).sum().item()//5

        # Calculate average validation loss
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate validation accuracy
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        # Calculate validation AUC
        val_AUC = metric.compute().item()
        val_AUCs.append(val_AUC)

        metric.reset()

        # Calculate validation F1 score
        val_f1 = multiclass_f1_score(y_pred_val,y_true_num_val,num_classes=5)
        # Print progress every 10 epochs
        # if (epoch + 1) % 10 == 0:
        if True:
            print(f'Epoch [{epoch+1}/{epochs}], '
                f'Train - loss: {avg_train_loss:.4f}, acc: {train_accuracy * 100:.2f}%, AUC: {train_AUC:.3f}, F1: {train_f1:.3f}; '
                f'Validation - loss: {avg_val_loss:.4f}, acc: {val_accuracy * 100:.2f}%, AUC: {val_AUC:.3f}, F1: {val_f1:.3f}')

    # Plotting the loss and accuracy over epochs
    gs = gridspec.GridSpec(2, 2)
    plt.figure(figsize=(12,8))
    ax = plt.subplot(gs[0, 0]) # row 0, col 0
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    ax = plt.subplot(gs[0, 1]) # row 0, col 1
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()


    ax = plt.subplot(gs[1, :]) # row 1, span all columns
    plt.plot(train_AUCs, label='Training AUC')
    plt.plot(val_AUCs, label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return val_losses
#%%

# Setting Hyperparameters and Training the Model

# Number of training epochs
epochs = 25

# Create an instance of the SimpleCNN model and move it to the specified device (GPU if available)
model = SimpleCNN().to(device)

# Define the loss criterion (CrossEntropyLoss) and the optimizer (Adam) for training the model
weights = torch.tensor([0.17768543*5, 0.1938815*5, 0.20993698*5, 0.14523569*5, 0.2732604*5]).to(device) # Weighted based on data availability (see 1 Data Analysis.ipynb)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model using the defined training function
val_losses_simple = train_model(model, train_loader, val_loader, epochs, criterion, optimizer)