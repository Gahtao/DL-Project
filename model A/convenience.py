import pandas as pd
import torchaudio
import torch
import os

def load_train(path = "../data/Train"):
    # Import train data
    df_train = pd.DataFrame(columns=["file_name", "accent", "gender", "audio"])
    df_train.file_name = os.listdir(path)
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