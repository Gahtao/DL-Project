# This file is used to import important and large functions easily

import pandas as pd
import torchaudio
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import os
import math
import numpy as np

def load_train(path = "../data/Train"):
    # Import train data
    df_train = pd.DataFrame(columns=["file_name", "stratify", "accent", "gender", "audio", "sample_rate"])
    df_train.file_name = os.listdir(path)
    df_train.stratify = [name.split("_")[0] for name in df_train.file_name]
    df_train.accent = [file_name[0] for file_name in df_train.file_name]
    df_train.gender = [file_name[1] for file_name in df_train.file_name]
    loaded_audio = []
    # display(df_train)
    sample_rates = set()
    for file_name in df_train.file_name:
        tensor, sr = torchaudio.load(f"{path}/{file_name}")
        loaded_audio.append(tensor[0])
        sample_rates.add(sr)

    df_train.audio = loaded_audio
    df_train["length"] = [len(df_train.audio[i])/16000 for i  in range(0, len(df_train))]
    return df_train, sample_rates

def loop_audio_df(df, target_duration, sr):
    looped_audios = []
    for _, row in df.iterrows():
        # Calculate the number of loops needed to reach 10 seconds

        num_loops = math.ceil(target_duration / row.length)

        # Loop the audio file the required number of times
        looped_audio = row.audio
        for _ in range(num_loops - 1):
            looped_audio = np.concatenate((looped_audio, row.audio))

        # Trim the audio to exactly 10 seconds
        if len(looped_audio) > target_duration * sr:
            looped_audio = looped_audio[:target_duration * sr]
        elif len(looped_audio) < target_duration * sr:
            # Pad the audio with silence to reach 10 seconds
            silence = np.zeros(target_duration * sr - len(looped_audio))
            looped_audio = np.concatenate((looped_audio, silence))
        looped_audios.append(torch.tensor(looped_audio))

    df["looped_audio"] = looped_audios

    looped_lengths = [len(x)/sr for x in df.looped_audio]
    df["looped_length"] = looped_lengths
    return df, sr

def load_train_post_processing(path = "./preprocessed/train/"):
    df_train = pd.DataFrame(columns=["file_name", "stratify", "accent", "gender", "audio"])
    df_train.file_name = os.listdir(path)
    df_train.stratify = [name.split("_")[0] for name in df_train.file_name]
    df_train.accent = [file_name[0] for file_name in df_train.file_name]
    df_train.gender = [file_name[1] for file_name in df_train.file_name]
    loaded_audio = []
    # display(df_train)
    sample_rates = set()
    for file_name in df_train.file_name:
        tensor = torch.tensor(np.load(f"{path}/{file_name}"))
        sr = 16000
        loaded_audio.append(tensor[0])
        sample_rates.add(sr)

    df_train.audio = loaded_audio
    df_train["length"] = [len(loaded_audio[i]) / 16000 for i in range(0, len(df_train))]
    return df_train, loaded_audio, sample_rates


def get_data_loaders(batch_size=128):
    # Load the full dataset
    df, loaded_audio, _ = load_train_post_processing()

    target = df['accent'].values
    features = torch.tensor(loaded_audio)

    # Passing to DataLoader
    dataset_ = TensorDataset(features, target)
    dataset_train = dataset_

    # Get the class names of CIFAR-10
    class_names = df['accent'].values

    # Define the size of the train, validation, and test sets
    train_size = 5000
    val_size = 1000
    subset_size = train_size + val_size

    # Create a balanced subset
    # Use a dictionary to store indices for each class
    # Initialize an empty dictionary, where keys are class indices and values are lists to store indices for each class.
    class_indices_dict = {class_idx: [] for class_idx in range(len(class_names))}

    # Iterate over the dataset to collect indices for each class
    # Iterate through the dataset, extract the index `i` and label from each element, and append the index to the corresponding class in the dictionary.
    for i, accent in enumerate(df['accent'].values):
        class_indices_dict[accent].append(i)

    # Split the indices into training, validation, and test sets
    train_indices = []
    val_indices = []

    train_size_per_class = train_size // len(class_names)
    val_size_per_class = val_size // len(class_names)
    for idx in range(len(class_names)):
        indices = class_indices_dict[idx][:subset_size]
        train_indices.extend(indices[:train_size_per_class])
        val_indices.extend(indices[train_size_per_class:train_size_per_class + val_size_per_class])

    # Create subsets
    train_dataset = Subset(dataset_train, train_indices)
    val_dataset = Subset(dataset_, val_indices)

    # Create data loader for the training set with WeightedRandomSampler
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # Create data loader for the validation set
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    return train_loader, val_loader, class_names

def genLabelsCSV(file_path):
    df_labels = pd.DataFrame(columns=["file_name", "label"])
    df_labels.file_name = os.listdir(file_path)
    df_labels.label = [name[0] for name in df_labels.file_name]
    df_labels.to_csv(f'{file_path}/labels.csv', index=False)