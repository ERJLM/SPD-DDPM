import os, shutil, random
from pathlib import Path
#from kaggle import api
import torch

from torch.utils.data import DataLoader , Dataset
import pandas as pd




def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

class CSVDataset(Dataset):
    def __init__(self, args):
        self.path = args.dataset_path
        self.data_frame = pd.read_csv(args.dataset_path)
        self.m = args.spd_size
        self.seq_length = self.m * self.m

    def __getitem__(self,idx):

        sample = self.data_frame.iloc[idx].values       
        sample = torch.from_numpy(sample)
        sample = sample.unsqueeze(0)
        
        # Normalize data to [0, 1]
        sample_min, sample_max = sample.min(), sample.max()
        if sample_max - sample_min > 1e-6:
            sample = (sample - sample_min) / (sample_max - sample_min)
        else:
            sample = torch.full_like(sample, 0.5)
            
        return sample.float()
    

    def __len__(self):
        return len(self.data_frame)



