import datetime
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
import torch.nn.functional as F

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
        label = one_hot(torch.as_tensor(self.img_labels.iloc[idx, 1] - 1), 5).unsqueeze(0).to(torch.float)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Training Loop with Validation
def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device, batch_size):
    # Lists to store training and validation losses, and accuracies
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_AUCs = []
    val_AUCs = []
    train_f1_scores = []
    val_f1_scores = []

    metric = AUC(n_tasks=batch_size)

    best_f1 = 0.0
    best_weights = model.state_dict()

    # Loop over epochs
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Lists to store true labels and predictions for F1 score
        all_train_labels = []
        all_train_preds = []

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move to GPU if available
            optimizer.zero_grad()

            # Get model outputs
            outputs = model(images)  # Pass the entire batch to the model

            # Calculate loss
            # print(labels.view(-1))
            labels = labels.reshape(len(labels), 5)
            labels_indices = torch.argmax(labels, dim=1)
            # print(labels.shape)
            # print(labels)
            # print(outputs.shape)
            loss = criterion(outputs, labels_indices)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Convert logits to probabilities
            outputs_prob = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities

            # Update metrics
            if len(images) == batch_size:
                metric.update(outputs_prob, labels)  # Update AUC with probabilities

            for i in range(len(images)):
                output_prob = outputs_prob[i]
                label = labels[i]

                # Get predicted classes
                # print(output_prob)
                _, predicted = torch.max(output_prob, dim=0)

                # Store true labels and predictions for F1 score
                all_train_labels.append(label.cpu())
                all_train_preds.append(predicted.cpu())

                # Store true labels and predictions
                total_train += label.size(0)
                correct_train += (predicted == label).sum().item()  # Count correct predictions

        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Calculate training accuracy
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Calculate training AUC
        # print(metric.compute())
        train_AUC = metric.compute()
        train_AUCs.append(train_AUC)
        # print(train_AUCs)
        # Reset AUC metric for the next epoch
        metric.reset()

        # Calculate weighted F1 score
        train_f1 = multiclass_f1_score(torch.stack(all_train_preds), torch.stack(all_train_labels).argmax(1),
                                       average='weighted', num_classes=5)
        train_f1_scores.append(train_f1.item())

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0

        y_true_val = torch.tensor([]).to(device)
        y_pred_val = torch.tensor([]).to(device)

        # Lists to store true labels and predictions for F1 score
        all_val_labels = []
        all_val_preds = []

        # Validation without gradient computation
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)  # Move to GPU if available

                val_outputs = model(val_images)  # Pass the entire batch to the model

                # Calculate loss
                val_labels = val_labels.reshape(len(val_labels), 5)
                val_labels_indices = torch.argmax(val_labels, dim=1)
                # print(labels.shape)
                # print(labels)
                # print(outputs.shape)
                val_loss = criterion(val_outputs, val_labels_indices)
                total_val_loss += val_loss.item()

                # Convert logits to probabilities
                val_outputs_prob = F.softmax(val_outputs, dim=1)  # Apply softmax to get probabilities

                # Update metrics
                if len(val_images) == batch_size:
                    metric.update(val_outputs_prob, val_labels)  # Update AUC

                # Get predicted classes
                _, predicted_val = torch.max(val_outputs_prob, 1)

                # Store true labels and predictions
                total_val += val_labels.size(0)
                for i in range(len(val_images)):
                    val_output_prob = val_outputs_prob[i]
                    val_label = val_labels[i]

                    # Get predicted classes
                    # print(output_prob)
                    _, predicted_val = torch.max(val_output_prob, dim=0)

                    all_val_labels.append(val_label.cpu())
                    all_val_preds.append(predicted_val.cpu())

                    # Store true labels and predictions
                    total_val += val_label.size(0)
                    # print(correct_val)
                    # print(predicted_val.shape)
                    # print(val_label.shape)
                    correct_val += (predicted_val == val_label).sum().item()  # Count correct predictions

        # Calculate average validation loss
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate validation accuracy
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        # Calculate validation AUC
        val_AUC = metric.compute()
        val_AUCs.append(val_AUC)

        # Reset AUC metric for the next epoch
        metric.reset()

        # Calculate weighted F1 score for training
        val_f1 = multiclass_f1_score(torch.stack(all_val_preds), torch.stack(all_val_labels).argmax(1),
                                     average='weighted', num_classes=5)
        if val_f1 > best_f1:
            best_weights = model.state_dict()
            best_f1 = val_f1
        val_f1_scores.append(val_f1.item())

        # print(type(avg_train_loss), avg_train_loss)
        # print(type(train_accuracy), train_accuracy)
        # print(type(avg_val_loss), avg_val_loss)
        # print(type(val_accuracy),val_accuracy)
        # print(type(train_AUC), train_AUC.mean().item())
        # print(type(val_AUC), val_AUC.mean().item())

        # Print progress
        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Train - loss: {avg_train_loss:.4f}, acc: {train_accuracy * 100:.2f}%, AUC: {train_AUC.mean().item():.3f}, F1: {train_f1:.3f}; '
              f'Validation - loss: {avg_val_loss:.4f}, acc: {val_accuracy * 100:.2f}%, AUC: {val_AUC.mean().item():.3f}, F1: {val_f1:.3f}', flush=True)


    # Plotting the loss, accuracy, AUC, and F1 score over epochs
    plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2)  # Create a 3x2 grid

    # Plot Training and Validation Loss
    ax1 = plt.subplot(gs[0, 0])  # row 0, col 0
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()

    # Plot Training and Validation Accuracy
    ax2 = plt.subplot(gs[0, 1])  # row 0, col 1
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy over Epochs')
    ax2.legend()

    # # Plot Training and Validation AUC
    # ax3 = plt.subplot(gs[1, 0])  # row 1, col 0
    # ax3.plot([train_AUC.cpu() for train_AUC in train_AUCs], label='Training AUC', color='blue')
    # ax3.plot([val_AUC.cpu() for val_AUC in val_AUCs], label='Validation AUC', color='orange')
    # ax3.set_xlabel('Epoch')
    # ax3.set_ylabel('AUC')
    # ax3.set_title('AUC over Epochs')
    # ax3.legend()

    # Plot Training and Validation F1 Score
    ax4 = plt.subplot(gs[1, 1])  # row 1, col 1
    ax4.plot(train_f1_scores, label='Training F1 Score', color='blue')
    ax4.plot(val_f1_scores, label='Validation F1 Score', color='orange')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Score over Epochs')
    ax4.legend()

    # Adjust layout
    plt.tight_layout()

    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    now = datetime.datetime.now()
    plt.savefig(f'./plots/plot-{now.hour}-{now.minute}-{now.second}{now.microsecond}.png')

    model.load_state_dict(best_weights)
    print(f"Best F1-score: {best_f1}")
    return val_losses
