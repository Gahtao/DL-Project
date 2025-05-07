# This file is used to import important and large functions easily

import pandas as pd
import torchaudio
import torch
import os
import math
import numpy as np

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