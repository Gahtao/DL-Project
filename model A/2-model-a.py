import datetime
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from shared_infrastructure import CustomImageDataset, train_model
import argparse

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
        fc1_in_features = 1664000
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
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        # Output layer
        x = self.softmax(x)
        
        return x


class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Fully Connected Layers
        rnn1_in_size = 1664000
        self.rnn1 = nn.RNN(input_size=rnn1_in_size, hidden_size=32)
        self.rnn2 = nn.RNN(input_size=32, hidden_size=16)
        self.fc1 = nn.Linear(in_features=16, out_features=5)

        # Output activitation function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # First Convolutional Block
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool(x)

        # Second Convolutional Block
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.pool(x)

        # Flatten for Fully Connected Layers
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x, _ = self.rnn1(x)
        x = self.tanh(x)

        x, _ = self.rnn2(x)
        x = self.tanh(x)

        x = self.fc1(x)
        x = self.tanh(x)

        # Output layer
        x = self.softmax(x)

        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model A')
    parser.add_argument('-m', '--model', help='The desired model', type=str, required=True)
    parser.add_argument('-d', '--data_path', help='Base path to data', type=str, required=True)
    parser.add_argument('-s', '--seed', help='Seed to ensure standard behavior', type=int, default=42, required=False)
    parser.add_argument('-b', '--batch_size', help='Size of the batch', type=int, default=16, required=False)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', type=float, default=0.0001, required=False)
    parser.add_argument('-e', '--epochs', help='Number of training epochs', type=int, default=25, required=False)

    args = vars(parser.parse_args())

    # Check if GPU is available and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}", flush=True)

    # Set random seed for reproducibility
    torch.manual_seed(args['seed'])

    print(f'BEGIN TRAINING - model {args['model']} - seed {args['seed']} - batch_size {args['batch_size']} - lr {args['learning_rate']} - epochs {args['epochs']} - data {args['data_path']} - {datetime.datetime.now()}', flush=True)
    train_set = CustomImageDataset(args['data_path'] + "/train/labels.csv", args['data_path'] + "train/specs")
    val_set = CustomImageDataset(args['data_path'] + "/val/labels.csv", args['data_path'] + "val/specs")

    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=True)

    # Create an instance of the SimpleCNN or SimpeRNN model and move it to the specified device (GPU if available)
    if args['model'] == 'CNN':
        model = SimpleCNN().to(device)
    elif args['model'] == 'RNN':
        model = SimpleRNN().to(device)
    else:
        raise Exception('Non-existent model was input. Please use CNN or RNN.')

    # Define the loss criterion (CrossEntropyLoss) and the optimizer (Adam) for training the model
    weights = torch.tensor([0.17768543*5, 0.1938815*5, 0.20993698*5, 0.14523569*5, 0.2732604*5]).to(device) # Weighted based on data availability (see 1 Data Analysis.ipynb)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Train the model using the defined training function
    val_losses_simple = train_model(model, train_loader, val_loader, args['epochs'], criterion, optimizer, device, args['batch_size'])
    print(val_losses_simple, flush=True)
    print(f'END TRAINING - {datetime.datetime.now()}', flush=True)
    print('', flush=True)
