#%%
import numpy as np
from kdg.utils import generate_gaussian_parity, generate_ellipse, generate_spirals, generate_sinewave, generate_polynomial
from kdg.utils import plot_2dsim
from kdg import kdf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import pickle
# %%
def unpickle(filename):
    with open(filename, 'rb') as f:
        df = pickle.load(f)
    return df

# %%
datasets = ['CIFAR10-0,1', 'CIFAR10-2,3', 'CIFAR10-4,5', 'CIFAR10-6,7', 'CIFAR10-8,9']
sample_size = [10, 100, 1000, 10000]
r = np.arange(0,20.5,1)
r = r = np.arange(0,10.5,.5)
linewidth = [6,3]

#sns.set_context('talk')
ticksize = 45
labelsize = 50
fig, ax = plt.subplots(3, 5, figsize=(55, 38))
sns.set_context('talk')
for jj, dataset in enumerate(datasets):
    filename = '/Users/jayantadey/kdg/benchmarks/cifar10_experiments/results/Task'+str(jj+1)+'.pickle'
    df = unpickle(filename)

    filename = '/Users/jayantadey/kdg/benchmarks/cifar10_experiments/results/Task'+str(jj+1)+'_OOD.pickle'
    df_ood = unpickle(filename)

    ax[0,jj].set_title(dataset, fontsize=70)

    if jj==0:
        ax[0,jj].set_ylabel('classification error', fontsize=labelsize+12)
        ax[1,jj].set_ylabel('ECE', fontsize=labelsize+12)
        ax[2,jj].set_ylabel('Mean Max Conf.', fontsize=labelsize+12)

    if jj==2:
        ax[0,jj].set_xlabel('sample size (log)', fontsize=labelsize+12)
        ax[1,jj].set_xlabel('sample size (log)', fontsize=labelsize+12)
        ax[2,jj].set_xlabel('Distance', fontsize=labelsize+12)

    ax[0,jj].plot(sample_size, df['err_kdn_geod_med'], linewidth=6, c='r', label='KDN-Geodesic')    
    ax[0,jj].fill_between(sample_size, df['err_kdn_geod_25'], df['err_kdn_geod_75'], facecolor='r', alpha=.3)
    ax[0,jj].plot(sample_size, df['err_kdn_euc_med'], linewidth=4, c='b', label='KDN-Euclidean')    
    ax[0,jj].fill_between(sample_size, df['err_kdn_euc_25'], df['err_kdn_euc_75'], facecolor='b', alpha=.3)
    ax[0,jj].plot(sample_size, df['err_dn_med'], linewidth=4, c='k', label='DN')    
    ax[0,jj].fill_between(sample_size, df['err_dn_25'], df['err_dn_75'], facecolor='k', alpha=.3)
    ax[0,jj].set_yticks([0.1,0.5])
    ax[0,jj].set_ylim([0.05,0.5])

    ax[0,jj].set_xscale("log")
    ax[0,jj].set_xticks([])
    ax[0,jj].tick_params(labelsize=60)

    right_side = ax[0,jj].spines["right"]
    right_side.set_visible(False)
    top_side = ax[0,jj].spines["top"]
    top_side.set_visible(False)

    ax[1,jj].plot(sample_size, df['ece_kdn_euc_med'], linewidth=4, c='b')    
    ax[1,jj].fill_between(sample_size, df['ece_kdn_euc_25'], df['ece_kdn_euc_75'], facecolor='b', alpha=.3)
    ax[1,jj].plot(sample_size, df['ece_dn_med'], linewidth=4, c='k')    
    ax[1,jj].fill_between(sample_size, df['ece_dn_25'], df['ece_dn_75'], facecolor='k', alpha=.3)
    ax[1,jj].plot(sample_size, df['ece_kdn_geod_med'], linewidth=6, c='r')    
    ax[1,jj].fill_between(sample_size, df['ece_kdn_geod_25'], df['ece_kdn_geod_75'], facecolor='r', alpha=.3)
    ax[1,jj].set_yticks([0.5,0.1])

    ax[1,jj].set_xscale("log")
    ax[1,jj].tick_params(labelsize=60)

    right_side = ax[1,jj].spines["right"]
    right_side.set_visible(False)
    top_side = ax[1,jj].spines["top"]
    top_side.set_visible(False)

    ax[2,jj].plot(r, df_ood['mmcOut_kdn_euc_med'], linewidth=4, c='b')    
    ax[2,jj].fill_between(r, df_ood['mmcOut_kdn_euc_25'], df_ood['mmcOut_kdn_euc_75'], facecolor='b', alpha=.3)
    ax[2,jj].plot(r, df_ood['mmcOut_dn_med'], linewidth=4, c='k')    
    ax[2,jj].fill_between(r, df_ood['mmcOut_dn_25'], df_ood['mmcOut_dn_75'], facecolor='k', alpha=.3)
    ax[2,jj].plot(r, df_ood['mmcOut_kdn_geod_med'], linewidth=6, c='r')    
    ax[2,jj].fill_between(r, df_ood['mmcOut_kdn_geod_25'], df_ood['mmcOut_kdn_geod_75'], facecolor='r', alpha=.3)
    ax[2,jj].set_yticks([0.5,1])

    ax[2,jj].tick_params(labelsize=60)

    right_side = ax[2,jj].spines["right"]
    right_side.set_visible(False)
    top_side = ax[2,jj].spines["top"]
    top_side.set_visible(False)

    if jj==0:
            leg = ax[0,jj].legend(bbox_to_anchor=(0.5, 0.08), bbox_transform=plt.gcf().transFigure,
                        ncol=3, loc='upper center', fontsize=labelsize+10)
            leg.get_frame().set_linewidth(0.0)


#plt.tight_layout()
plt.savefig('../plots/cifar10_lenet_res.pdf')
# %%
