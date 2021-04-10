# %%
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np
import openml
from scipy.interpolate import interp1d

'''tasks = [3,11,12,14,16,18,22,23,
        28,31,32,37,43,45,49,53,
        2074,3022,3549,3560,3902,
        3903,3913,3917,3918,9946,
        9952,9957,9960,9971,9978,
        10093,10101,125922,146817,
        146819,146820,146821,146822,
        146824,167140,167141]'''

tasks = [3,6,11,12,14,16,22,23,28,31,32,37,43,45,
49,2074,3481,3902,3903,3913,3917,9946,9952,9964,9977,
9978,146195,14970,167120,167141]
reps = 5
samples = []
samples_normalized = []
delta_kappas = []
delta_eces = []
kappa_over_dataset = [] 
ece_over_dataset = []
kappa_normalized_over_dataset = []
ece_normalized_over_dataset = []

sns.set_context('talk')
fig, ax = plt.subplots(1,2, figsize=(16,8))
#minimum = 0
#maximum = 1e10

for task in tasks:
    task_ = openml.tasks.get_task(task)
    X, y = task_.get_X_and_y()
    p = X.shape[1]

    df = pd.read_csv('amalgamated_activation2/openML_cc18_task_'+str(task)+'.csv')
    sample_ = list(np.unique(df['sample']))
    samples.extend(sample_)
    #sample_ = list(np.unique(df['sample'])/p)
    #samples_normalized.extend(sample_)

samples = np.sort(np.unique(samples))
#samples_normalized = np.sort(np.unique(samples_normalized))

for task in tasks:
    X, y = task_.get_X_and_y()
    p = X.shape[1]

    df = pd.read_csv('amalgamated_activation/openML_cc18_task_'+str(task)+'.csv')
    sample_ = list(np.unique(df['sample']))
    sample__ = list(np.unique(df['sample'])/p)

    kappa_kdf = np.zeros((len(sample_),reps), dtype=float)
    kappa_rf = np.zeros((len(sample_),reps), dtype=float)
    delta_kappa = np.zeros((len(sample_),reps), dtype=float)
    delta_ece = np.zeros((len(sample_),reps), dtype=float)
    ece_kdf = np.zeros((len(sample_),reps), dtype=float)
    ece_rf = np.zeros((len(sample_),reps), dtype=float)

    for ii in range(reps):
        kappa_kdf[:,ii] = df['kappa_kdf'][df['fold']==ii]
        kappa_rf[:,ii] = df['kappa_rf'][df['fold']==ii]
        delta_kappa[:,ii] = kappa_kdf[:,ii] - kappa_rf[:,ii]

        #ax[0].plot(sample_size, delta_kappa[:,ii], c='k', alpha=0.5, lw=1)

    mean_delta_kappa = np.mean(delta_kappa,axis=1)
    interp_func_kappa = interp1d(sample_, mean_delta_kappa)
    interpolated_kappa = np.array([np.nan]*len(samples))
    interpolated_kappa_ = interp_func_kappa(samples[np.where((samples>=sample_[0]) & (samples<=sample_[-1]))[0]])
    interpolated_kappa[np.where((samples>=sample_[0]) & (samples<=sample_[-1]))[0]] = interpolated_kappa_
    kappa_over_dataset.append(interpolated_kappa)

    '''interp_func_kappa_normalized = interp1d(sample__, mean_delta_kappa)
    interpolated_kappa_normalized = np.array([np.nan]*len(samples_normalized))
    interpolated_kappa_ = interp_func_kappa_normalized(samples_normalized[np.where((samples_normalized>=sample__[0]) & (samples_normalized<=sample__[-1]))[0]])
    interpolated_kappa_normalized[np.where((samples_normalized>=sample__[0]) & (samples_normalized<=sample__[-1]))[0]] = interpolated_kappa_
    kappa_normalized_over_dataset.append(interpolated_kappa_normalized)'''

    ax[0].plot(sample_, mean_delta_kappa, c='r', alpha=0.3, lw=.5)
    #ax.fill_between(sample_size, mean_kdf-1.96*var_kdf, mean_kdf+1.96*var_kdf, facecolor='r', alpha=0.5)
    #ax[0].plot(sample_size, np.mean(kappa_rf,axis=1), label='RF', c='k', lw=3)
    #ax.fill_between(sample_size, mean_rf-1.96*var_kdf, mean_rf+1.96*var_kdf, facecolor='k', alpha=0.5)

    ax[0].set_xlabel('Sample size')
    ax[0].set_ylabel('kappa_kdf - kappa_rf')
    ax[0].set_xscale('log')
    #ax[0][0].legend(frameon=False)
    ax[0].set_title('Delta Kappa', fontsize=24)
    #ax[0][0].set_yticks([0,.2,.4,.6,.8,1])
    right_side = ax[0].spines["right"]
    right_side.set_visible(False)
    top_side = ax[0].spines["top"]
    top_side.set_visible(False)

    '''ax[1][0].plot(sample__, mean_delta_kappa, c='r', alpha=0.3, lw=.5)
    ax[1][0].set_xlabel('Normalized Sample Size (n/p)')
    ax[1][0].set_ylabel('kappa_kdf - kappa_rf')
    ax[1][0].set_xscale('log')
    #ax[1][0].legend(frameon=False)
    ax[1][0].set_title('Delta Kappa', fontsize=24)
    #ax[1][0].set_yticks([0,.2,.4,.6,.8,1])
    right_side = ax[1][0].spines["right"]
    right_side.set_visible(False)
    top_side = ax[1][0].spines["top"]
    top_side.set_visible(False)'''

    for ii in range(reps):
        ece_kdf[:,ii] = df['ece_kdf'][df['fold']==ii]
        ece_rf[:,ii] = df['ece_rf'][df['fold']==ii]
        delta_ece[:,ii] = ece_kdf[:,ii] - ece_rf[:,ii]

        #ax[1].plot(sample_size, ece_kdf[:,ii], c='r', alpha=0.5, lw=1)
        #ax[1].plot(sample_size, delta_ece[:,ii], c='k', alpha=0.5, lw=1)

    mean_delta_ece = np.mean(delta_ece,axis=1)
    interp_func_ece = interp1d(sample_, mean_delta_ece)
    interpolated_ece = np.array([np.nan]*len(samples))
    interpolated_ece_ = interp_func_ece(samples[np.where((samples>=sample_[0]) & (samples<=sample_[-1]))[0]])
    interpolated_ece[np.where((samples>=sample_[0]) & (samples<=sample_[-1]))[0]] = interpolated_ece_
    ece_over_dataset.append(interpolated_ece)

    '''interp_func_ece_normalized = interp1d(sample__, mean_delta_ece)
    interpolated_ece_normalized = np.array([np.nan]*len(samples_normalized))
    interpolated_ece_ = interp_func_ece_normalized(samples_normalized[np.where((samples_normalized>=sample__[0]) & (samples_normalized<=sample__[-1]))[0]])
    interpolated_ece_normalized[np.where((samples_normalized>=sample__[0]) & (samples_normalized<=sample__[-1]))[0]] = interpolated_ece_
    ece_normalized_over_dataset.append(interpolated_ece_normalized)
    #interpolated_ece[np.where(samples<sample_[0] & samples>sample_[-1])] = np.nan'''

    ax[1].plot(sample_, mean_delta_ece, c='r', alpha=0.3, lw=.5)
    #ax.fill_between(sample_size, mean_kdf-1.96*var_kdf, mean_kdf+1.96*var_kdf, facecolor='r', alpha=0.5)
    #ax[1].plot(sample_size, np.mean(ece_rf,axis=1), label='RF', c='k', lw=3)
    #ax.fill_between(sample_size, mean_rf-1.96*var_kdf, mean_rf+1.96*var_kdf, facecolor='k', alpha=0.5)

    ax[1].set_xlabel('Sample size')
    ax[1].set_ylabel('ECE_kdf - ECE_rf')
    ax[1].set_xscale('log')
    #ax[1].legend(frameon=False)
    ax[1].set_title('Delta ECE',fontsize=24)
    #ax[1].set_yticks([0,.2,.4,.6,.8,1])
    right_side = ax[1].spines["right"]
    right_side.set_visible(False)
    top_side = ax[1].spines["top"]
    top_side.set_visible(False)

    '''ax[1][1].plot(sample__, mean_delta_ece, c='r', alpha=0.3, lw=.5)
    ax[1][1].set_xlabel('Normalized Sample Size (n/p)')
    ax[1][1].set_ylabel('kappa_kdf - kappa_rf')
    ax[1][1].set_xscale('log')
    #ax[1][0].legend(frameon=False)
    ax[1][1].set_title('Delta ECE', fontsize=24)
    #ax[1][0].set_yticks([0,.2,.4,.6,.8,1])
    right_side = ax[1][1].spines["right"]
    right_side.set_visible(False)
    top_side = ax[1][1].spines["top"]
    top_side.set_visible(False)'''


ax[0].hlines(0, 4, np.max(samples), colors="k", linestyles="dashed", linewidth=1.5)
ax[1].hlines(0, 4, np.max(samples), colors="k", linestyles="dashed", linewidth=1.5)

qunatiles = np.nanquantile(kappa_over_dataset,[.25,.75],axis=0)
ax[0].fill_between(samples, qunatiles[0], qunatiles[1], facecolor='r', alpha=.3)
ax[0].plot(samples, np.nanmean(kappa_over_dataset, axis=0), c='r', lw=3)

qunatiles = np.nanquantile(ece_over_dataset,[.25,.75],axis=0)
ax[1].fill_between(samples, qunatiles[0], qunatiles[1], facecolor='r', alpha=.3)
ax[1].plot(samples, np.nanmean(ece_over_dataset, axis=0), c='r', lw=3)

'''qunatiles = np.nanquantile(kappa_normalized_over_dataset,[.25,.75],axis=0)
ax[1][0].fill_between(samples_normalized, qunatiles[0], qunatiles[1], facecolor='r', alpha=.3)
ax[1][0].hlines(0, np.min(samples_normalized), np.max(samples_normalized), colors="k", linestyles="dashed", linewidth=1.5)
ax[1][0].plot(samples_normalized, np.nanmean(kappa_normalized_over_dataset, axis=0), c='r', lw=3)

qunatiles = np.nanquantile(ece_normalized_over_dataset,[.25,.75],axis=0)
ax[1][1].fill_between(samples_normalized, qunatiles[0], qunatiles[1], facecolor='r', alpha=.3)
ax[1][1].hlines(0, np.min(samples_normalized), np.max(samples_normalized), colors="k", linestyles="dashed", linewidth=1.5)
ax[1][1].plot(samples_normalized, np.nanmean(ece_normalized_over_dataset, axis=0), c='r', lw=3)'''
plt.savefig('plots/openML_cc18_all_amalgamated_activation.pdf')
#plt.show()
