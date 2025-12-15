import pandas as pd
import numpy as np
import torch
from scipy.linalg import logm


def fro(a,b):
    dis = np.linalg.norm(a-b, 'fro')
    return dis

def spd_dis(A,B):
    matrix_ABA = torch.matmul(torch.matmul(tensor_power(A,-0.5),B),tensor_power(A,-0.5))
    S,U = torch.linalg.eigh(matrix_ABA)
    S_trans = (torch.log(S))**2
    dis = S_trans.sum(1)

    return dis

def tensor_power(A,r):
    S, U = torch.linalg.eigh(A)
    pow_S = S**r
    power_A = torch.matmul(torch.matmul(U,torch.diag_embed(pow_S)),U.transpose(1,2))  

    return(power_A)

def get_lowest_distance(A, n_samples_pp, n_test, T, m):
    """
    We sampled n matrices for each predictor Y, therefore our csv file 
    has n*number_of_predictors rows, instead of number_of_predictors. 
    This function returns the closest sample to each predictor Y.

    Args:
        A: Matrix with the samples.
        n_samples_pp: Number of samples per predictor.
        n_test: Number of predictors
        T: Matrix with the true values of predictors Y.
        m: size of the Matrix
    """
    
    spd_ddpm = A.reshape(n_test, n_samples_pp, m, m)
    Y = T.reshape(n_test, m, m)
    
    best_samples = []

    for i in range(n_test):
        true_mat = Y[i]              
        cands = spd_ddpm[i]       
        
        # Compute the Frobenius Distance 
        diffs = cands - true_mat
        fro_dists = np.linalg.norm(diffs, axis=(1, 2), ord='fro')
        
        # Compute the Affine Invariant Distance (Vectorized)
        t_tensor = torch.tensor(true_mat, dtype=torch.float32).unsqueeze(0) 
        c_tensor = torch.tensor(cands, dtype=torch.float32)                
        affine_dists = spd_dis(t_tensor, c_tensor).detach().numpy()
        

        # Set NaNs to inf
        affine_dists[np.isnan(affine_dists)] = np.inf

        # Score each sample by the sum of both distances 
        total_score = fro_dists + affine_dists
        best_idx = np.argmin(total_score)
        
        # Mark invalid scores as NaN
        if np.isinf(total_score[best_idx]) or fro_dists[best_idx] > 1000:
            best_samples.append(np.full(m*m, np.nan))
        else:
            best_samples.append(cands[best_idx].flatten())

    return pd.DataFrame(best_samples)

spds_true = pd.read_csv('data/condition/data_true.csv') # True results
spds_class_ddpm = pd.read_csv('data/condition/data_3.csv') # DDPM results
spds_ddpm = pd.read_csv('data/condition/ddpm_spds_list_reproduced.csv') # SPD-DDPM results
spds_frechet = pd.read_csv('data/condition/data_1.csv') # Frechet Regression results


n = len(spds_true)
m = 10
n_test = 1100
a = 0

# Get the true samples and the samples from each model
spds_true = spds_true.iloc[0:n_test,:]
spds_ddpm = spds_ddpm.iloc[0:n_test*20,:]
spds_ddpm = get_lowest_distance(spds_ddpm.to_numpy(), n_samples_pp=20, n_test=n_test, T=spds_true.to_numpy(), m=m)
spds_frechet = spds_frechet.iloc[0:n_test,:]
spds_class_ddpm = spds_class_ddpm.iloc[0:n_test,:]

# Drop rows with missing values
missing_rows = spds_ddpm[spds_ddpm.isnull().any(axis=1)].index
spds_ddpm= spds_ddpm.drop(missing_rows, errors='ignore')
spds_frechet = spds_frechet.drop(missing_rows, errors='ignore')
spds_true = spds_true.drop(missing_rows, errors='ignore')
spds_class_ddpm = spds_class_ddpm.drop(missing_rows, errors='ignore')

spds_true = spds_true.to_numpy()
spds_ddpm = spds_ddpm.to_numpy()
spds_frechet = spds_frechet.to_numpy()
spds_class_ddpm = spds_class_ddpm.to_numpy()

n_test2 = len(spds_ddpm)
true = spds_true.reshape(n_test2, m, m)
spd_ddpm = spds_ddpm.reshape(n_test2, m, m)
frechet = spds_frechet.reshape(n_test2, m, m)
class_ddpm = spds_class_ddpm.reshape(n_test2, m, m)


""" Frechet Regression Distances """
f_dis_frechet = 0
for i in range(0,n_test2):
    f_dis_frechet = f_dis_frechet + fro(true[i,:,:],frechet[i,:,:])
f_dis_frechet = f_dis_frechet/n_test2
a_dis_frechet = spd_dis(torch.tensor(true),torch.tensor(frechet)).mean()

""" SPD-DDPM Distances """
f_dis_ddpm = 0
for i in range(0,n_test2):
    f_dis_ddpm = f_dis_ddpm + fro(true[i,:,:],spd_ddpm[i,:,:])
f_dis_ddpm = f_dis_ddpm/n_test2
a_dis_ddpm = spd_dis(torch.tensor(true),torch.tensor(spd_ddpm)).mean()

""" DDPM Distances """
f_dis_class = 0
for i in range(0,n_test2):
    f_dis_class = f_dis_class + fro(true[i,:,:],class_ddpm[i,:,:])
f_dis_class= f_dis_class/n_test2
a_dis_class_list = spd_dis(torch.tensor(true),torch.tensor(class_ddpm))
cleaned_data = a_dis_class_list [~torch.isnan(a_dis_class_list )]
cleaned_data = cleaned_data[~torch.isinf(cleaned_data)]
a_dis_class = cleaned_data.mean()


""" Results """
print("Affine Invariant (Frechet Reg.):", a_dis_frechet)
print("Frobenius (Frechet Reg.):",f_dis_frechet)
print()
print("Affine Invariant (SPD-DDPM):", a_dis_ddpm)
print("Frobenius (SPD-DDPM):",f_dis_ddpm)
print()
print("Affine Invariant (DDPM):", a_dis_class)
print("Frobenius (DDPM):", f_dis_class)




