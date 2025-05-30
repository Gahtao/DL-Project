#%%
import convenience as c
import numpy as np
import os
import datetime
from sklearn.model_selection import train_test_split
import math

def preprocess(train_val_path, output_path):
    print(f'Data pre-processing START {datetime.datetime.now()}', flush=True)
    print(f'From {train_val_path} to {output_path}', flush=True)
    print(f'{"-"*20}', flush=True)

    df_train_val, sample_rates = c.load_train_val(train_val_path)
    print(f'Data loaded! {datetime.datetime.now()}', flush=True)

    df_train_val['audio'] = df_train_val.apply(lambda row: c.standardize(row['audio']), axis=1)
    print(f'Data standardized! {datetime.datetime.now()}', flush=True)

    target_duration = math.ceil(df_train_val.length.max())
    sr = list(sample_rates)[0]
    df_train_val, _ = c.loop_audio_df(df_train_val, target_duration, sr)
    print(f'Data looped! {datetime.datetime.now()}', flush=True)

    df_train, df_val = train_test_split(df_train_val, test_size=0.2, random_state=42, stratify=df_train_val.stratify)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    print(f'Data split! {datetime.datetime.now()}', flush=True)

    train_output_path = output_path + '/train/'
    val_output_path = output_path + '/val/'
    if not os.path.exists(train_output_path + '/specs/'):
        os.makedirs(train_output_path + '/specs/')
    if not os.path.exists(val_output_path + '/specs/'):
        os.makedirs(val_output_path + '/specs/')

    for i in range(len(df_train.looped_audio)):
        np.save(f'{train_output_path}/specs/{df_train.file_name[i][:-4]}.npy', df_train.looped_audio[i])
    print(f'Training data saved! {datetime.datetime.now()}', flush=True)


    for i in range(len(df_val.looped_audio)):
        np.save(f'{val_output_path}/specs/{df_val.file_name[i][:-4]}.npy', df_val.looped_audio[i])
    print(f'Validation data saved! {datetime.datetime.now()}', flush=True)

    c.genLabelsCSV(train_output_path + '/specs/', train_output_path)
    print(f'Generated labels.csv for training data! {datetime.datetime.now()}', flush=True)

    c.genLabelsCSV(val_output_path + '/specs/', val_output_path)
    print(f'Generated labels.csv for validation data! {datetime.datetime.now()}', flush=True)

    print(f'{"-"*20}', flush=True)
    print(f'Data pre-processing END {datetime.datetime.now()}', flush=True)

if __name__ == '__main__':
    preprocess('../data/Train/', './preprocessed/')
    preprocess('./augmented/', './preprocessed-augmented/')