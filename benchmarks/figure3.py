#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import numpy as np
import openml
from scipy.interpolate import interp1d

# %%
res_folder_kdn = 'openml_res/openml_kdn_res'
res_folder_kdf = 'openml_res/openml_kdf_res'
files = os.listdir(res_folder_kdf)
files.remove('.DS_Store')
# %%
def plot_summary_error(files, folder, model='kdf', parent='rf', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharey=True, sharex=True, constrained_layout=True)

    sample_combined = []
    for file in files:
        df = pd.read_csv(folder+'/'+file)
        sample_combined.extend(np.unique(df['samples']))

    sample_combined = np.unique(
            sample_combined
        )
    
    err_diff_ = []
    for file in files:
        df = pd.read_csv(folder+'/'+file)
        samples = np.unique(df['samples'])
        err_kdx_med = []
        err_x_med = []

        
        
        for sample in samples:
            kdx = df['err_'+model][df['samples']==sample]
            x = df['err_'+parent][df['samples']==sample]

            err_kdx_med.append(
                np.median(kdx)
            )

            err_x_med.append(
                np.median(x)
            )
            
        err_diff = np.array(err_x_med) - np.array(err_kdx_med)
        idx = np.where(sample_combined<=samples[-1])[0]
        f = interp1d(samples, err_diff, kind='linear')
        tmp_diff = list(f(sample_combined[idx]))
        tmp_diff.extend((len(sample_combined)-len(idx))*[np.nan])
        err_diff_.append(
            tmp_diff
        )
        
        ax.plot(samples, err_diff, linewidth=4, c='r', alpha=.1)
    
    ax.plot(sample_combined, np.nanmean(np.array(err_diff_), axis=0), linewidth=4, c='r')


def plot_summary_ece(files, folder, model='kdf', parent='rf', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharey=True, sharex=True, constrained_layout=True)

    sample_combined = []
    for file in files:
        df = pd.read_csv(folder+'/'+file)
        sample_combined.extend(np.unique(df['samples']))

    sample_combined = np.unique(
            sample_combined
        )
    
    err_diff_ = []
    for file in files:
        df = pd.read_csv(folder+'/'+file)
        samples = np.unique(df['samples'])
        err_kdx_med = []
        err_x_med = []

        
        
        for sample in samples:
            kdx = df['ece_'+model][df['samples']==sample]
            x = df['ece_'+parent][df['samples']==sample]

            err_kdx_med.append(
                np.median(kdx)
            )

            err_x_med.append(
                np.median(x)
            )
            
        err_diff = np.array(err_x_med) - np.array(err_kdx_med)
        idx = np.where(sample_combined<=samples[-1])[0]
        f = interp1d(samples, err_diff, kind='linear')
        tmp_diff = list(f(sample_combined[idx]))
        tmp_diff.extend((len(sample_combined)-len(idx))*[np.nan])
        err_diff_.append(
            tmp_diff
        )
        
        ax.plot(samples, err_diff, linewidth=4, c='r', alpha=.1)
    
    ax.plot(sample_combined, np.nanmean(np.array(err_diff_), axis=0), linewidth=4, c='r')


#%%
sns.set(
    color_codes=True, palette="bright", style="white", context="talk", font_scale=1.5
)

fig, ax = plt.subplots(2, 2, figsize=(14,14), constrained_layout=True, sharex=True)

plot_summary_error(files, res_folder_kdf, ax=ax[0][0])
plot_summary_ece(files, res_folder_kdf, ax=ax[0][1])
plot_summary_error(files, res_folder_kdn, model='kdn', parent='dn', ax=ax[1][0])
plot_summary_ece(files, res_folder_kdn, model='kdn', parent='dn', ax=ax[1][1])

ax[0][0].set_xscale("log")
ax[0][0].set_ylim([-0.35, .2])
ax[0][0].set_yticks([-.3,0,.2])
ax[0][0].set_ylabel('RF error - KDF error', fontsize=35)
ax[0][0].text(100, .1, 'KDF wins')
ax[0][0].text(100, -.2, 'KDF loses')

ax[0][1].set_xscale("log")
ax[0][1].set_ylim([-0.3, .45])
ax[0][1].set_yticks([-.2,0,.4])
ax[0][1].set_ylabel('RF ECE - KDF ECE', fontsize=35)
ax[0][1].text(100, .2, 'KDF wins')
ax[0][1].text(100, -.1, 'KDF loses')

ax[1][0].set_xscale("log")
ax[1][0].set_ylim([-0.44, .2])
ax[1][0].set_yticks([-.3,0,.2])
ax[1][0].set_ylabel('DN error - KDN error', fontsize=35)
ax[1][0].text(100, .1, 'KDN wins')
ax[1][0].text(100, -.2, 'KDN loses')

ax[1][1].set_xscale("log")
ax[1][1].set_ylim([-0.15, .1])
ax[1][1].set_yticks([-.1,0,.1])
ax[1][1].set_ylabel('DN ECE - KDN ECE', fontsize=35)
ax[1][1].text(100, .06, 'KDN wins')
ax[1][1].text(100, -.06, 'KDN loses')

for j in range(2):
    for i in range(2):
        ax[j][i].hlines(0, 10,1e5, colors='grey', linestyles='dashed',linewidth=4)

        ax[j][i].tick_params(labelsize=30)
        right_side = ax[j][i].spines["right"]
        right_side.set_visible(False)
        top_side = ax[j][i].spines["top"]
        top_side.set_visible(False)
fig.text(0.53, -0.04, "Number of Training Samples", ha="center", fontsize=35)

plt.savefig('plots/openml_summary.pdf')
# %%
