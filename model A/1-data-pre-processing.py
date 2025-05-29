#%%
import convenience as c
import numpy as np
import os
import datetime
from sklearn.model_selection import train_test_split

print(f'Data pre-processing START', flush=True)
print(f'{"-"*20}', flush=True)

df_train_val, sample_rates = c.load_train_val()
print(f'Data loaded! {datetime.datetime.now()}', flush=True)
df_train_val.head()
#%%

df_train_val['audio_standardized'] = df_train_val.apply(lambda row: c.standardize(row['audio']), axis=1)
print(f'Data standardized! {datetime.datetime.now()}', flush=True)

df_train_val['audio_standardized'] = c.zero_fills(df_train_val['audio_standardized'])[0]
print(f'Data zero filled! {datetime.datetime.now()}', flush=True)

df_train, df_val = train_test_split(df_train_val, test_size=0.2, random_state=42, stratify=df_train_val.stratify)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

print(f'Data split! {datetime.datetime.now()}', flush=True)

#%%

if not os.path.exists('./preprocessed/train/'):
    os.makedirs('./preprocessed/train/')
if not os.path.exists('./preprocessed/val/'):
    os.makedirs('./preprocessed/val/')

for i in range(len(df_train.audio_standardized)):
    np.save(f'./preprocessed/train/{df_train.file_name[i][:-4]}.npy', df_train.audio_standardized[i])
print(f'Training data saved! {datetime.datetime.now()}', flush=True)


for i in range(len(df_val.audio_standardized)):
    np.save(f'./preprocessed/val/{df_val.file_name[i][:-4]}.npy', df_val.audio_standardized[i])
print(f'Validation data saved! {datetime.datetime.now()}', flush=True)


#%%

c.genLabelsCSV('./preprocessed/train/')
print(f'Generated labels.csv for training data! {datetime.datetime.now()}', flush=True)

c.genLabelsCSV('./preprocessed/val/')
print(f'Generated labels.csv for validation data! {datetime.datetime.now()}', flush=True)

print(f'{"-"*20}', flush=True)
print(f'Data pre-processing END', flush=True)
