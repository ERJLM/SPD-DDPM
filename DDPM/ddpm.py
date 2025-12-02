import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from torch.utils.data import DataLoader
import logging
import warnings
import numpy as np
import pandas as pd
import json
from datetime import datetime
import math
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
warnings.filterwarnings("ignore")

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def train(args):

    device = args.device
    dataset = CSVDataset(args) # read training data
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1
    ).to(device)

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = args.spd_size * args.spd_size,
        timesteps = 1000,
        objective = 'pred_noise'
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    epoch_start = 1
    if args.resume != '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(args.resume))

    results = {'train_loss': [],'lr':[]}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)


    for epoch in range(epoch_start,args.epochs):
        total_loss = []
        pbar = tqdm(dataloader)
        lr1 = adjust_learning_rate(optimizer, epoch, args)
        
        for i, spds in enumerate(pbar):
            spds = spds.to(device)
            optimizer.zero_grad()
            
            loss = diffusion(spds) 
           
            total_loss.append(loss.item()) 
            pbar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], loss.item()))
            
            loss.backward()
            optimizer.step()

        epoch_loss = np.mean(total_loss)
        results['train_loss'].append(epoch_loss)
        results['lr'].append(lr1)
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM"
    args.epochs = 500
    args.batch_size = 150
    args.spd_size = 8  
    args.time_size = 256 # Not used in Unet1D but kept for compatibility
    args.dataset_path = "data/uncondition/train_data.csv"
    args.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    args.lr = 8e-5
    args.results_dir = 'result/ddpm-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.resume = ""

    train(args)


if __name__ == '__main__':
   launch()