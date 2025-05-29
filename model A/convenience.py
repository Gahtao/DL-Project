import pandas as pd
import torchaudio
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import os
import math
import numpy as np

'''
input:
    path: Path to original train_val data
output:
    df_train_val: Pandas dataframe with "file_name", "stratify", "accent", "gender", "audio"
    sample_rates: List of sample rates of audio files
'''

def load_train(path = "../data/Train"):
    # Import train data
    df_train = pd.DataFrame(columns=["file_name", "stratify", "accent", "gender", "audio"])
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

def load_train_val(path = "../data/Train"):
    # Import train data
    df_train_val = pd.DataFrame(columns=["file_name", "stratify", "accent", "gender", "audio", "sample_rate"])
    df_train_val.file_name = os.listdir(path)
    df_train_val.stratify = [name.split("_")[0] for name in df_train_val.file_name]
    df_train_val.accent = [file_name[0] for file_name in df_train_val.file_name]
    df_train_val.gender = [file_name[1] for file_name in df_train_val.file_name]
    loaded_audio = []
    # display(df_train)
    sample_rates = set()
    for file_name in df_train_val.file_name:
        tensor, sr = torchaudio.load(f"{path}/{file_name}")
        loaded_audio.append(tensor[0])
        sample_rates.add(sr)

    df_train_val.audio = loaded_audio
    df_train_val["length"] = [len(df_train_val.audio[i])/16000 for i  in range(0, len(df_train_val))]
    return df_train_val, sample_rates

'''
input:
    df: Pandas dataframe with "file_name", "stratify", "accent", "gender", "audio"
    target_duration: The length of all resulting audio
    sr: Desired sampling rate
output:
    df: df with additional "looped audio" section
    sr: The same sampling rate as input
'''
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

'''
input:
    path: Path to where the preprocessed audio files are
output:
    df_train: Pandas dataframe with "file_name", "stratify", "accent", "gender", "audio"
    loaded_audio: A list of tensors, each an audio file
    sample_rates: A list of the sample rates of each file (should all be the same?)
'''
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
        tnsor = torch.tensor(np.load(f"{path}/{file_name}"))
        sr = 16000
        loaded_audio.append(tnsor[0])
        sample_rates.add(sr)

    df_train.audio = loaded_audio
    df_train["length"] = [len(loaded_audio[i]) / 16000 for i in range(0, len(df_train))]
    return df_train, loaded_audio, sample_rates

'''
input:
    batch_size: The batch_size
output:
    train_loader: Torch audio loader for the training data
    val_loader: Torch audio loader for the validation data
    class_names: List of class_names (accents)
'''
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

'''
input:
    file_path: A path to the folder with audio files
output:
    None (directly generated a .csv file)
'''
def genLabelsCSV(file_path):
    df_labels = pd.DataFrame(columns=["file_name", "label"])
    df_labels.file_name = os.listdir(file_path)
    df_labels.label = [name[0] for name in df_labels.file_name]
    df_labels.to_csv(f'{file_path}/labels.csv', index=False)


'''
input:
    tnsr: a single tensor
output:
    n_tensr: a standardized tensor
'''
def standardize(tnsr):
    n_tensr = torch.zeros_like(tnsr)
    mean = tnsr.mean()
    std = tnsr.std()
    for i, n in enumerate(tnsr):
        n_tensr[i] = (n - mean) / std
    return n_tensr

'''
input:
    tnsrs: a list of tensors
output:
    n_tnsrs: a list of tensors of the same length (appended with zeros where necessary)
    max_length: the new length of all tensors
'''
def zero_fills(tnsrs):
    max_length = max([len(x) for x in tnsrs])
    n_tnsrs = []
    for tnsr in tnsrs:
        n_tnsr = tnsr
        for i in range(max_length- len(tnsr)):
            n_tnsr = torch.cat((n_tnsr, torch.tensor([0])), 0)
        n_tnsrs.append(n_tnsr)
    print(f"Filled tensors! New length: {max_length}")
    return n_tnsrs, max_length