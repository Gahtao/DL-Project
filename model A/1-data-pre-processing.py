#%%
import convenience as c

df_train_val, sample_rates = c.load_train()
df_train_val.head()
#%%
import torch
    
def standardize(tnsr):
    n_tensr = torch.zeros_like(tnsr)
    mean = tnsr.mean()
    std = tnsr.std()
    for i, n in enumerate(tnsr):
        n_tensr[i] = (n - mean) / std
    print("Standardized tensor!")
    return n_tensr

#%%
from sklearn.model_selection import train_test_split


df_train_val['audio_standardized'] = df_train_val.apply(lambda row: standardize(row['audio']), axis=1)
df_train_val.head()

df_train, df_val = train_test_split(df_train_val, test_size=0.2, random_state=42, stratify=df_train_val.stratify)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
#%%
import numpy as np

for i in range(len(df_train.audio_standardized)):
    np.save(f'preprocessed/train/{df_train.file_name[i][:-4]}.npy', df_train.audio_standardized[i])
    
for i in range(len(df_val.audio_standardized)):
    np.save(f'preprocessed/val/{df_train.file_name[i][:-4]}.npy', df_train.audio_standardized[i])
#%%
import convenience as c

c.genLabelsCSV('./preprocessed/train/')
c.genLabelsCSV('./preprocessed/val/')