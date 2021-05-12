#%%
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np
import openml
from scipy.interpolate import interp1d
from os import listdir, getcwd 

#%%
df = pd.read_csv('simulation_res_BIC.csv')
sample_size = np.logspace(
        np.log10(10),
        np.log10(5000),
        num=10,
        endpoint=True,
        dtype=int
        )
dist_kdf_med = []
dist_kdf_25_quantile = []
dist_kdf_75_quantile = []
dist_rf_med = []
dist_rf_25_quantile = []
dist_rf_75_quantile = []

err_kdf_med = []
err_kdf_25_quantile = []
err_kdf_75_quantile = []
err_rf_med = []
err_rf_25_quantile = []
err_rf_75_quantile = []

#%%
for sample in sample_size:
    res_kdf = df['hellinger dist kdf'][df['sample']==sample]
    res_rf = df['hellinger dist rf'][df['sample']==sample]

    dist_kdf_med.append(np.median(res_kdf))
    dist_rf_med.append(np.median(res_rf))
    dist_kdf_25_quantile.append(
        np.quantile(res_kdf,[.25])[0]
    )
    dist_kdf_75_quantile.append(
        np.quantile(res_kdf,[.75])[0]
    )
    dist_rf_25_quantile.append(
        np.quantile(res_rf,[.25])[0]
    )
    dist_rf_75_quantile.append(
        np.quantile(res_rf,[.75])[0]
    )

#%%
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(sample_size, dist_kdf_med, c="r", label='KDF')
ax.plot(sample_size, dist_rf_med, c="k", label='RF')

ax.fill_between(sample_size, dist_kdf_25_quantile, dist_kdf_75_quantile, facecolor='r', alpha=.3)
ax.fill_between(sample_size, dist_rf_25_quantile, dist_rf_75_quantile, facecolor='k', alpha=.3)

ax.set_xscale('log')
ax.set_xlabel('Sample size')
ax.set_ylabel('Hellinger Distance')
ax.legend()

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
plt.savefig('plots/sim_res_spherical.pdf')
# %% plot them all
covariance_types = ['full', 'spherical', 'diag', 'AIC', 'BIC']
sample_size = np.logspace(
        np.log10(10),
        np.log10(5000),
        num=10,
        endpoint=True,
        dtype=int
        )
dist_rf_med = []
dist_rf_25_quantile = []
dist_rf_75_quantile = []

err_rf_med = []
err_rf_25_quantile = []
err_rf_75_quantile = []
clr = ["#e41a1c", "#f781bf", "#b15928", "#377eb8", "#4daf4a", "#984ea3"]
c = sns.color_palette(clr, n_colors=5)

df = pd.read_csv('simulation_res_full.csv')

for sample in sample_size:
    res_rf = df['hellinger dist rf'][df['sample']==sample]
    err_rf = df['error rf'][df['sample']==sample]

    dist_rf_med.append(np.median(res_rf))
    dist_rf_25_quantile.append(
            np.quantile(res_rf,[.25])[0]
        )
    dist_rf_75_quantile.append(
        np.quantile(res_rf,[.75])[0]
    )

    err_rf_med.append(np.median(err_rf))
    err_rf_25_quantile.append(
            np.quantile(err_rf,[.25])[0]
        )
    err_rf_75_quantile.append(
        np.quantile(err_rf,[.75])[0]
    )

sns.set_context('talk')
fig, ax = plt.subplots(2,1, figsize=(8,8))

ax.plot(sample_size, dist_rf_med, c="k", label='RF')
ax.fill_between(sample_size, dist_rf_25_quantile, dist_rf_75_quantile, facecolor='k', alpha=.3)

for ii, cov_type in enumerate(covariance_types):
    filename = 'simulation_res_' + cov_type + '.csv'
    df = pd.read_csv(filename)

    dist_kdf_med = []
    dist_kdf_25_quantile = []
    dist_kdf_75_quantile = []

    err_kdf_med = []
    err_kdf_25_quantile = []
    err_kdf_75_quantile = []    

    for sample in sample_size:
        res_kdf = df['hellinger dist kdf'][df['sample']==sample]
        err_kdf = df['error kdf'][df['sample']==sample]
        #res_rf = df['hellinger dist rf'][df['sample']==sample]

        dist_kdf_med.append(np.median(res_kdf))
        dist_rf_med.append(np.median(res_rf))
        dist_kdf_25_quantile.append(
            np.quantile(res_kdf,[.25])[0]
        )
        dist_kdf_75_quantile.append(
            np.quantile(res_kdf,[.75])[0]
        )

        err_kdf_med.append(np.median(err_kdf))
        err_kdf_25_quantile.append(
                np.quantile(err_kdf,[.25])[0]
            )
        err_kdf_75_quantile.append(
            np.quantile(err_kdf,[.75])[0]
        )

    ax[0].plot(sample_size, dist_kdf_med, c=c[ii], label=cov_type)
    ax[0].fill_between(sample_size, dist_kdf_25_quantile, dist_kdf_75_quantile, facecolor=c[ii], alpha=.3)
    ax[1].plot(sample_size, err_kdf_med, c=c[ii], label=cov_type)
    ax[1].fill_between(sample_size, err_kdf_25_quantile, err_kdf_75_quantile, facecolor=c[ii], alpha=.3)

ax[0].set_xscale('log')
ax[0].set_xlabel('Sample size')
ax[0].set_ylabel('Hellinger Distance')
ax[0].legend()

ax[1].set_xscale('log')
ax[1].set_xlabel('Sample size')
ax[1].set_ylabel('Error')
ax[1].legend()

right_side = ax[0].spines["right"]
right_side.set_visible(False)
top_side = ax[0].spines["top"]
top_side.set_visible(False)

right_side = ax[1].spines["right"]
right_side.set_visible(False)
top_side = ax[1].spines["top"]
top_side.set_visible(False)

plt.savefig('plots/sim_res_all.pdf')


# %%
