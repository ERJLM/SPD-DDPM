from SPD_net import SPD_NET
import torch 
from ddpm import Diffusion
import pandas as pd
from support_function import *
import time
from tqdm import tqdm

def ddpm_sample(n,path,Y,Y_size, device):
    model = SPD_NET(m,256,Y_size).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    diffusion = Diffusion(spd_size=m, device=device)
    Y = Y.to(device)
    x = diffusion.sample(model,n,Y)
    
    return x

n_sample = 1100 
m = 10  
num = 20 
device = "cuda" if torch.cuda.is_available() else "cpu"


Y = pd.read_csv("data/condition/test_y.csv")
Y = Y.iloc[0:n_sample,1:14]
a = 0

# Load model
model_path = "result/ddpm_co-2025-12-13-19-25-38/model_last.pth"
Y_size = Y.shape[1]
vectors_list = pd.DataFrame(index=range(n_sample*num), columns=range(m*m))

# The following loop implements conditional inference (sample generation) of the SPD-DDPM model.
# This is described on Algorithm 4 in the paper.
for i in tqdm(range(n_sample), desc="Generating Samples"):
    
    Y_loc = pd.DataFrame([Y.iloc[a+i]] * num, columns=Y.columns)
    Y_loc = torch.tensor(Y_loc.values).to(device)
    samples = ddpm_sample(num,model_path,Y_loc,Y_size,device)
    n_len = samples.size(0)

    samples1 = samples.float().reshape(n_len, m*m)
    samples_list = pd.DataFrame(samples1.cpu().numpy())
    vectors_list.iloc[i*num:i*num+n_len,:] = samples_list.iloc[:,:]
    vectors_list.to_csv("data/condition/ddpm_spds_list_reproduced1.csv",index=False)



