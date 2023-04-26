#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import numpy as np
import openml
# %%
res_folder_kdn = 'openml_res/openml_kdn_res'
res_folder_kdf = 'openml_res/openml_kdf_res'
files = os.listdir(res_folder_kdf)
files.remove('.DS_Store')
# %%
def plot_summary_error(file, folder, model='kdf', parent='rf', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharey=True, sharex=True, constrained_layout=True)

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

    samples = np.unique(df['samples'])
    err_diff = np.array(err_x_med) - np.array(err_kdx_med)
    
    ax.plot(samples, err_diff, linewidth=4, c='r')

    return False

def plot_summary_ece(file, folder, model='kdf', parent='rf', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharey=True, sharex=True, constrained_layout=True)

    df = pd.read_csv(folder+'/'+file)
    samples = np.unique(df['samples'])
    ece_kdx_med = []
    ece_x_med = []

    for sample in samples:
        kdx = df['ece_'+model][df['samples']==sample]
        x = df['ece_'+parent][df['samples']==sample]
        
        ece_kdx_med.append(
            np.median(kdx)
        )

        ece_x_med.append(
            np.median(x)
        )

    samples = np.unique(df['samples'])
    ece_diff = np.array(ece_x_med) - np.array(ece_kdx_med)
    
    ax.plot(samples, ece_diff, linewidth=4, c='r')

    return False

#%%
sns.set(
    color_codes=True, palette="bright", style="white", context="talk", font_scale=1.5
)

fig, ax = plt.subplots(2, 2, figsize=(14,14), constrained_layout=True, sharex=True)

for file in files:
    plot_summary_error(file, res_folder_kdf, ax=ax[0][0])
    plot_summary_ece(file, res_folder_kdf, ax=ax[0][1])
    plot_summary_error(file, res_folder_kdn, model='kdn', parent='dn', ax=ax[1][0])
    plot_summary_ece(file, res_folder_kdn, model='kdn', parent='dn', ax=ax[1][1])

ax[0][0].set_xscale("log")
ax[0][0].set_ylim([-0.35, .2])
ax[0][0].set_yticks([-.3,0,.2])
ax[0][0].set_ylabel('RF error - KDF error', fontsize=35)

ax[0][1].set_xscale("log")
ax[0][1].set_ylim([-0.3, .45])
ax[0][1].set_yticks([-.2,0,.4])
ax[0][1].set_ylabel('RF ECE - KDF ECE', fontsize=35)

ax[1][0].set_xscale("log")
ax[1][0].set_ylim([-0.44, .2])
ax[1][0].set_yticks([-.3,0,.2])
ax[1][0].set_ylabel('DN error - KDN error', fontsize=35)

ax[1][1].set_xscale("log")
ax[1][1].set_ylim([-0.15, .1])
ax[1][1].set_yticks([-.1,0,.1])
ax[1][1].set_ylabel('DN ECE - KDN ECE', fontsize=35)

for j in range(2):
    for i in range(2):
        #ax[i][j].axhline(y=0,xmin=1, xmax=1e4, c='k', linewidth=4, linestyle='dashed')
        ax[j][i].tick_params(labelsize=30)
        right_side = ax[j][i].spines["right"]
        right_side.set_visible(False)
        top_side = ax[j][i].spines["top"]
        top_side.set_visible(False)
fig.text(0.53, -0.04, "Number of Training Samples", ha="center", fontsize=35)

plt.savefig('plots/openml_summary.pdf')
# %%
