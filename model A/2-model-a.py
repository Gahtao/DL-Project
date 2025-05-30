import datetime
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import os
from shared_infrastructure import CustomImageDataset, train_model
import argparse

#%%
# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.dropout_prob = 0.2

        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=16)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=32)

        self.dropout1 = nn.Dropout(self.dropout_prob)
        self.dropout2 = nn.Dropout(self.dropout_prob)

        # Fully Connected Layers
        fc1_in_features = 1664000
        self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=32)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=5)

        # Output activitation function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # First Convolutional Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Second Convolutional Block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten for Fully Connected Layers
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Output layer
        x = self.softmax(x)
        
        return x


class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.dropout_prob = 0.2
        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=16)
        self.tanh = nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=32)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=64)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(num_features=128)

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(num_features=256)

        # Fully Connected Layers
        rnn1_in_size = 1664000
        self.rnn1 = nn.RNN(input_size=rnn1_in_size, hidden_size=256, num_layers=2, batch_first=True)
        self.bnr1 = nn.BatchNorm1d(num_features=256)
        self.rnn2 = nn.RNN(input_size=256, hidden_size=128, num_layers=2, batch_first=True)
        self.bnr2 = nn.BatchNorm1d(num_features=128)
        self.rnn3 = nn.RNN(input_size=128, hidden_size=64, num_layers=2, batch_first=True)
        self.bnr3 = nn.BatchNorm1d(num_features=64)
        self.rnn4 = nn.RNN(input_size=64, hidden_size=32, num_layers=2, batch_first=True)
        self.bnr4 = nn.BatchNorm1d(num_features=32)

        self.fc1 = nn.Linear(in_features=32, out_features=5)

        self.dropout = nn.Dropout(self.dropout_prob)

        # Output activitation function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.tanh(x)
        x = self.pool(x)

        # Flatten for Fully Connected Layers
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        h01 = torch.zeros(self.rnn1.num_layers, self.rnn1.hidden_size, requires_grad=True).to(device)
        h01 = h01.to(device)
        x, hn1 = self.rnn1(x, h01)
        x = self.bnr1(x)
        x = self.tanh(x)
        x = self.dropout(x)

        h02 = torch.zeros(self.rnn2.num_layers, self.rnn2.hidden_size, requires_grad=True).to(device)
        x, hn2 = self.rnn2(x, h02)
        x = self.bnr2(x)
        x = self.tanh(x)
        x = self.dropout(x)

        h03 = torch.zeros(self.rnn3.num_layers, self.rnn3.hidden_size, requires_grad=True).to(device)
        x, hn3 = self.rnn3(x, h03)
        x = self.bnr3(x)
        x = self.tanh(x)
        x = self.dropout(x)

        h04 = torch.zeros(self.rnn4.num_layers, self.rnn4.hidden_size, requires_grad=True).to(device)
        x, hn4 = self.rnn4(x, h04)
        x = self.bnr4(x)
        x = self.tanh(x)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout(x)

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

    # Save the model
    if not os.path.exists('./models/'):
        os.makedirs('./models/')
    now = datetime.datetime.now()
    torch.save(model, f'./models/{args['model']}-{now.hour}-{now.minute}-{now.second}{now.microsecond}')

    print(val_losses_simple, flush=True)
    print(f'END TRAINING - {datetime.datetime.now()}', flush=True)
    print('', flush=True)
