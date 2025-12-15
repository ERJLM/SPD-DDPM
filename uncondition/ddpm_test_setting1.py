from SPD_net import SPD_NET
import torch 
from ddpm import Diffusion
import pandas as pd
from support_function import *
import matplotlib.pyplot as plt

def ddpm_sample(n, dataset_center_path, path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SPD_NET(spd_size=m,time_size=256).to(device)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])
    diffusion = Diffusion(spd_size=m, device=device)
    x = diffusion.sample(dataset_center_path, model, n)
    return x

device = "cuda" if torch.cuda.is_available() else "cpu"
n = 300
m = 8
dataset_center_path = "data/uncondition/exp1_setting1_init.csv"
init = pd.read_csv(dataset_center_path)
init =  torch.tensor(init.values)
init_tensor =  init.repeat(n, 1,1).to(device) # Repeat the tensor n times
model_path = "result/spd_ddpm_un-2025-12-13-18-08-40/model_last.pth" #"result/spd_uncondition.pth"

# Unconditional sampling using the SPD-DDPM model.
# This is described on Algorithm 2 in the paper.
sample_list = ddpm_sample(n, dataset_center_path, model_path)
test_dis = spd_dis(init_tensor,sample_list).cpu() # Distances between target and generated samples by the SPD_DDPM
mask1 = torch.isnan(test_dis) == False
test_dis = test_dis[mask1]

print(f"Final test distance: {test_dis.mean()}")

vectors = sample_list.reshape(n, m*m).cpu()
df = pd.DataFrame(vectors)
df.to_csv("data/uncondition/generated_samples_spd_ddpm_reproduced2.csv",index=False)


