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
        self.dataset_name = args.dataset_name
        self.path = args.dataset_path
        
        if self.dataset_name == "synthetic":
            self.data_frame = pd.read_csv(args.dataset_path)
        else:
            self.tensor = torch.load(args.dataset_path)
        self.m = args.spd_size

    def __getitem__(self,idx):

        if self.dataset_name == "synthetic":
            sample = self.data_frame.iloc[idx].values       
            sample = sample.reshape(self.m, self.m)
            return torch.from_numpy(sample)
        else:
            self.tensor[idx].fill_diagonal_(1)
            return self.tensor[idx]
            
    

    def __len__(self):
        if self.dataset_name == "synthetic":
            return len(self.data_frame)
        else:
            return len(self.tensor)



