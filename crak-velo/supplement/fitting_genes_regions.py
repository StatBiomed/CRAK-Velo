import numpy as np
import pandas as pd
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvelo as scv
scv.settings.verbosity = 0
# import unitvelo as utv
# import gseapy as gp
#import multivelo as mv

import warnings
warnings.filterwarnings("ignore")


def average_over_intervals(Y, T, L):
    """
    Calculate the average of Y over time intervals of length L.

    Parameters:
    Y (list of floats): List of values.
    T (list of floats): List of corresponding time points.
    L (float): Length of the time intervals.

    Returns:
    tuple of lists: Two lists - one for interval start times and one for averages of Y in those intervals.
    """
    if len(Y) != len(T):
        raise ValueError("Y and T must be the same length")
    
    # Sort by time
    combined = sorted(zip(T, Y))
    T_sorted, Y_sorted = zip(*combined)

    interval_starts = []
    averages = []
    start = T_sorted[0]
    current_sum = 0
    count = 0

    for i in range(len(T_sorted)):
        if T_sorted[i] < start + L:
            current_sum += Y_sorted[i]
            count += 1
        else:
            if count > 0:
                average = current_sum / count
                interval_starts.append(start)
                averages.append(average)
            # Move to the next interval
            start = start + L
            current_sum = Y_sorted[i]
            count = 1

            # Handle the case where time jumps over multiple intervals
            while T_sorted[i] >= start + L:
                interval_starts.append(start)
                averages.append(None)  # None for intervals with no data
                start += L

    # Last interval
    if count > 0:
        average = current_sum / count
        interval_starts.append(start)
        averages.append(average)
    
    return interval_starts, averages



def min_max_normalize(signal, min_val=0, max_val=1):
    signal_min = np.min(signal,axis=0)
    signal_max = np.max(signal,axis=0)
    return (signal - signal_min) / (signal_max - signal_min) * (max_val - min_val) + min_val


def compute_alpha_atac(adata,adata_atac, B):
    w = np.multiply(adata.varm["fit_region_weights"],B.T )
    c_ = np.where(w!=0)[1]
    r_ = np.where(w!=0)[0]

   
    alpha_atac = np.zeros([adata.shape[0], adata.shape[1]])
    gene_names = adata.var_names
    counter = 0
    adata_atac.obsm["cisTopic"] =  min_max_normalize(adata_atac.obsm["cisTopic"])

    for i, gene_name in enumerate(gene_names):
        gene_number = np.where(adata.var_names == gene_name)[0][0]
        r_g = adata.varm["fit_region_weights"][gene_number, c_[r_ == gene_number]]
       # r_g[r_g!=0] = 1
        if r_g.shape[0] == 0:
            counter += 1
        phi_r = adata_atac.obsm["cisTopic"][:, c_[r_ == gene_number]]
        w_r = np.multiply(phi_r,r_g).sum(axis=1)
        etta = adata.var['fit_etta'][gene_number]
        alpha_atac[:,i] = etta *w_r 
       
    adata.layers["ATAC"] = alpha_atac
    return alpha_atac, adata   


def region_unspliced_kinetics(adata, adata_atac, B, gene_name, interval = 0.05):

    w = np.multiply(adata.varm["fit_region_weights"],B.T )
    c = np.where(w!=0)[1]
    r = np.where(w!=0)[0]

    gene_number = np.where(adata.var_names == gene_name)[0][0]
    r_g = adata.varm["fit_region_weights"][gene_number, c[r == gene_number]]

    phi_r =  min_max_normalize(adata_atac.obsm["cisTopic"])[:, c[r == gene_number]]

    if phi_r.shape[1] != 0:

        time = np.zeros([phi_r.shape[1],int(1/interval)])
        c = np.zeros([phi_r.shape[1],int(1/interval)])

        for i in np.arange(0,phi_r.shape[1]):
            time[i,:], c[i,:] = average_over_intervals(r_g[i]*phi_r[:,i],adata.obs["latent_time"], interval)

        t, u_avg = average_over_intervals(adata.layers["Mu"][:,gene_number],
                                                adata.obs["latent_time"], interval)
        
        return time, c, u_avg
    
    else:
        print('No regions associated to '+str(gene_name)+" is found in the dataset")



