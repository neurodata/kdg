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
res_folder_kdn_ood = 'openml_res/openml_kdn_res_ood'
res_folder_kdf_ood = 'openml_res/openml_kdf_res_ood'
files = os.listdir(res_folder_kdn)
files.remove('.DS_Store')
# %%
id_done_ood = [6,11,12,14,16,18,22,28,32,37,44,54,182,300,458, 554,1049,1050,1063,1067,1068, 1462, 1464, 1468, 1475, 1478, 1485, 1487, 1489, 1494, 1497, 1501, 1510, 4134, 4538, 40499, 40979, 40982, 40983, 40984, 40994, 40996]
#%%
def plot_summary_ood(files, folder, model='kdf', parent='rf', color='r', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharey=True, sharex=True, constrained_layout=True)

    r = np.arange(0,5,.5)
    r[1:] += .5

    err_diff_ = []
    for file in files:
        if os.path.exists(folder+'/'+file):
            df = pd.read_csv(folder+'/'+file)

            data_id = file[:-4]
            data_id = int(data_id[8:])

            dataset = openml.datasets.get_dataset(data_id)
            X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )
            _, counts = np.unique(y, return_counts=True)
            mean_max_ood = np.max(counts)/np.sum(counts)
            #print(mean_max_ood)
            err_kdx_med = []
            err_x_med = []

            for dist in r:
                kdx = np.abs(df['conf_'+model][df['distance']==dist] - mean_max_ood)
                x = np.abs(df['conf_'+parent][df['distance']==dist] - mean_max_ood)

                err_kdx_med.append(
                    np.median(kdx)
                )

                err_x_med.append(
                    np.median(x)
                )
            
            err_diff_.append(
                np.array(err_x_med) - np.array(err_kdx_med)
            )

    qunatiles = np.nanquantile(np.array(err_diff_),[.25,.75],axis=0)
    ax.plot(r[1:], np.nanmedian(np.array(err_diff_), axis=0)[1:], linewidth=4, c=color)
    ax.fill_between(r[1:], qunatiles[0][1:], qunatiles[1][1:], facecolor=color, alpha=.3)


def plot_summary_error(files, folder, model='kdf', parent='rf', color='r', ax=None):
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
        
       # ax.plot(samples, err_diff, linewidth=4, c='r', alpha=.1)
    qunatiles = np.nanquantile(np.array(err_diff_),[.25,.75],axis=0)
    ax.plot(sample_combined, np.nanmedian(np.array(err_diff_), axis=0), linewidth=4, c=color)    
    ax.fill_between(sample_combined, qunatiles[0], qunatiles[1], facecolor=color, alpha=.3)


def plot_summary_ece(files, folder, model='kdf', parent='rf', color='r', ax=None):
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
        
        #ax.plot(samples, err_diff, linewidth=4, c='r', alpha=.1)
    qunatiles = np.nanquantile(np.array(err_diff_),[.25,.75],axis=0)
    ax.plot(sample_combined, np.nanmedian(np.array(err_diff_), axis=0), linewidth=4, c=color)
    ax.fill_between(sample_combined, qunatiles[0], qunatiles[1], facecolor=color, alpha=.3)


#%%
sns.set(
    color_codes=True, palette="bright", style="white", context="talk", font_scale=1.5
)

fig, ax = plt.subplots(2, 3, figsize=(21,14))

plot_summary_error(files, res_folder_kdf, model='kdf_geod', ax=ax[0][0])
plot_summary_error(files, res_folder_kdf, color='b', ax=ax[0][0])
plot_summary_ece(files, res_folder_kdf, model='kdf_geod', ax=ax[0][1])
plot_summary_ece(files, res_folder_kdf, color='b', ax=ax[0][1])
plot_summary_error(files, res_folder_kdn, model='kdn_geod', parent='dn', ax=ax[1][0])
plot_summary_error(files, res_folder_kdn, model='kdn', parent='dn', color='b', ax=ax[1][0])
plot_summary_ece(files, res_folder_kdn, model='kdn_geod', parent='dn', ax=ax[1][1])
plot_summary_ece(files, res_folder_kdn, model='kdn', parent='dn', color='b', ax=ax[1][1])
plot_summary_ood(files, res_folder_kdf_ood, color='b', ax=ax[0][2])
plot_summary_ood(files, res_folder_kdn_ood, model='kdn', parent='dn', color='b', ax=ax[1][2])

ax[1][0].set_xlim([100, 50000])
ax[1][1].set_xlim([100, 50000])
ax[0][0].set_xlim([100, 50000])
ax[0][1].set_xlim([100, 50000])


ax[0][0].set_title('Classification Error', fontsize=40)

ax[0][0].set_xscale("log")
ax[0][0].set_ylim([-0.1, .1])
ax[0][0].set_yticks([-.1,0,.1])
ax[0][0].set_xticks([])

ax[0][0].set_ylabel('Difference', fontsize=35)
#ax[0][0].text(100, .05, 'KGF wins')
#ax[0][0].text(100, -.08, 'RF wins')

ax[0][1].set_title('ID Calibration Error', fontsize=40)

ax[0][1].set_xscale("log")
ax[0][1].set_ylim([-0.1, .1])
ax[0][1].set_yticks([-.1,0,.4])
ax[0][1].set_xticks([])
ax[0][1].set_ylabel('', fontsize=35)
#ax[0][1].text(100, .3, 'KGF wins')
#ax[0][1].text(100, -.05, 'RF wins')

ax[1][0].set_xscale("log")
ax[1][0].set_ylim([-0.05, .05])
ax[1][0].set_yticks([-.05,0,.05])
ax[1][0].set_ylabel('Difference', fontsize=35)
#ax[1][0].text(100, .05, 'KGN wins')
#ax[1][0].text(100, -.08, 'DN wins')

ax[1][1].set_xscale("log")
ax[1][1].set_ylim([-0.05, .05])
ax[1][1].set_yticks([-.05,0,.05])
ax[1][1].set_ylabel('', fontsize=35)
#ax[1][1].text(100, .05, 'KGN wins')
#ax[1][1].text(100, -.08, 'DN wins')

ax[0][2].set_ylim([-0.25, .25])
ax[0][2].set_yticks([-.2,0,.2])
ax[0][2].set_xticks([])
#ax[2][0].set_ylabel('Difference', fontsize=35)
#ax[0][2].text(2, .05, 'KGF wins')
#ax[0][2].text(2, -.08, 'RF wins')
ax[0][2].set_xlim([1, 5])
#ax[0][2].text(.25, .1, 'ID', rotation=90, fontsize=40, color='b')
#ax[0][2].text(1.5, .05, 'OOD', rotation=90, fontsize=40, color='b')
#ax[0][2].axvline(x=1, ymin=-0.2, ymax=1, color='b', linestyle='dashed',linewidth=4)

ax[0][2].set_title('OOD Calibration Error', fontsize=40)


ax[1][2].set_ylim([-0.2, .85])
ax[1][2].set_xlim([1, 5])
ax[1][2].set_yticks([-.2,0,.8])
ax[1][2].set_xticks([1,3,5])
#ax[2][0].set_ylabel('Difference', fontsize=35)
#ax[1][2].text(2, .3, 'KGN wins')
#ax[1][2].text(2, -.08, 'DN wins')
#ax[2][0].set_ylabel('Difference', fontsize=35)
#ax[1][2].text(.25, .5, 'ID', rotation=90, fontsize=40, color='b')
#ax[1][2].text(1.5, .4, 'OOD', rotation=90, fontsize=40, color='b')
#ax[1][2].axvline(x=1, ymin=-0.2, ymax=1, color='b', linestyle='dashed',linewidth=4)
#ax[1][2].set_xlabel('Distance')

for j in range(2):
    for i in range(2):
        ax[j][i].hlines(0, 10,1e5, colors='grey', linestyles='dashed',linewidth=4)

        ax[j][i].tick_params(labelsize=30)
        right_side = ax[j][i].spines["right"]
        right_side.set_visible(False)
        top_side = ax[j][i].spines["top"]
        top_side.set_visible(False)

for i in range(2):
    ax[i][2].hlines(0, 0,5, colors='grey', linestyles='dashed',linewidth=4)

    ax[i][2].tick_params(labelsize=30)
    right_side = ax[i][2].spines["right"]
    right_side.set_visible(False)
    top_side = ax[i][2].spines["top"]
    top_side.set_visible(False)

fig.text(0.43, 0.01, "Number of Training Samples (log)", ha="center", fontsize=35)
fig.text(0.83, 0.01, "Distance", ha="center", fontsize=35)
plt.tight_layout()
plt.savefig('plots/openml_summary.pdf')
# %%
