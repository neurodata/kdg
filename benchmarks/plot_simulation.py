#%%
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np
import openml
from scipy.interpolate import interp1d
from os import listdir, getcwd 

#%%
df = pd.read_csv('sim_res/simulation_res.csv')
#df_bic = pd.read_csv('simulation_res_BIC.csv')
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

dist_kdf_bic_med = []
dist_kdf_bic_25_quantile = []
dist_kdf_bic_75_quantile = []

dist_rf_med = []
dist_rf_25_quantile = []
dist_rf_75_quantile = []

err_kdf_med = []
err_kdf_25_quantile = []
err_kdf_75_quantile = []

err_kdf_bic_med = []
err_kdf_bic_25_quantile = []
err_kdf_bic_75_quantile = []

err_rf_med = []
err_rf_25_quantile = []
err_rf_75_quantile = []

#%%
for sample in sample_size:
    res_kdf = df['hellinger dist kdf'][df['sample']==sample]
    res_rf = df['hellinger dist rf'][df['sample']==sample]
    #res_kdf_bic = df_bic['hellinger dist kdf'][df_bic['sample']==sample]

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
    '''dist_kdf_bic_med.append(np.median(res_kdf_bic))
    dist_kdf_bic_25_quantile.append(
        np.quantile(res_kdf_bic,[.25])[0]
    )
    dist_kdf_bic_75_quantile.append(
        np.quantile(res_kdf_bic,[.75])[0]
    )'''

    err_rf = df['error rf'][df['sample']==sample]
    
    err_rf_med.append(np.median(err_rf))
    err_rf_25_quantile.append(
            np.quantile(err_rf,[.25])[0]
        )
    err_rf_75_quantile.append(
        np.quantile(err_rf,[.75])[0]
    )

    err_kdf = df['error kdf'][df['sample']==sample]

    err_kdf_med.append(np.median(err_kdf))
    err_kdf_25_quantile.append(
            np.quantile(err_kdf,[.25])[0]
        )
    err_kdf_75_quantile.append(
            np.quantile(err_kdf,[.75])[0]
        )

    #err_kdf_bic = df_bic['error kdf'][df_bic['sample']==sample]

    '''err_kdf_bic_med.append(np.median(err_kdf_bic))
    err_kdf_bic_25_quantile.append(
            np.quantile(err_kdf_bic,[.25])[0]
        )
    err_kdf_bic_75_quantile.append(
            np.quantile(err_kdf_bic,[.75])[0]
        )'''
#%%
sns.set_context('talk')
fig, ax = plt.subplots(1,2, figsize=(16,8))

ax[0].plot(sample_size, dist_kdf_med, c="r", label='KDF')
#ax[0].plot(sample_size, dist_kdf_bic_med, c="b", label='KDF (bic)')
ax[0].plot(sample_size, dist_rf_med, c="k", label='RF')

ax[0].fill_between(sample_size, dist_kdf_25_quantile, dist_kdf_75_quantile, facecolor='r', alpha=.3)
ax[0].fill_between(sample_size, dist_rf_25_quantile, dist_rf_75_quantile, facecolor='k', alpha=.3)
#ax[0].fill_between(sample_size, dist_kdf_bic_25_quantile, dist_kdf_bic_75_quantile, facecolor='b', alpha=.3)

ax[0].set_xscale('log')
ax[0].set_xlabel('Sample size')
ax[0].set_ylabel('Hellinger Distance')

right_side = ax[0].spines["right"]
right_side.set_visible(False)
top_side = ax[0].spines["top"]
top_side.set_visible(False)

ax[1].plot(sample_size, err_kdf_med, c="r", label='KDF')
#ax[1].plot(sample_size, err_kdf_bic_med, c="b", label='KDF (bic)')
ax[1].plot(sample_size, err_rf_med, c="k", label='RF')

ax[1].fill_between(sample_size, err_kdf_25_quantile, err_kdf_75_quantile, facecolor='r', alpha=.3)
ax[1].fill_between(sample_size, err_rf_25_quantile, err_rf_75_quantile, facecolor='k', alpha=.3)
#ax[1].fill_between(sample_size, err_kdf_bic_25_quantile, err_kdf_bic_75_quantile, facecolor='b', alpha=.3)

ax[1].set_xscale('log')
ax[1].set_xlabel('Sample size')
ax[1].set_ylabel('Generalization error')
ax[1].legend()

right_side = ax[1].spines["right"]
right_side.set_visible(False)
top_side = ax[1].spines["top"]
top_side.set_visible(False)
plt.savefig('plots/sim_res.pdf')
# %% plot them all
covariance_types = ['full']
sample_size = np.logspace(
        np.log10(10),
        np.log10(5000),
        num=10,
        endpoint=True,
        dtype=int
        )
dist_nn_med = []
dist_nn_25_quantile = []
dist_nn_75_quantile = []

err_nn_med = []
err_nn_25_quantile = []
err_nn_75_quantile = []

dist_kdn_med = []
dist_kdn_25_quantile = []
dist_kdn_75_quantile = []

err_kdn_med = []
err_kdn_25_quantile = []
err_kdn_75_quantile = []

clr = [ "#4daf4a", "#984ea3"]
c = sns.color_palette(clr, n_colors=5)

df = pd.read_csv('sim_res/simulation_res_nn_full.csv')

for sample in sample_size:
    res_nn = df['hellinger dist nn'][df['sample']==sample]
    err_nn = df['error nn'][df['sample']==sample]

    dist_nn_med.append(np.median(res_nn))
    dist_nn_25_quantile.append(
            np.quantile(res_nn,[.25])[0]
        )
    dist_nn_75_quantile.append(
        np.quantile(res_nn,[.75])[0]
    )

    err_nn_med.append(np.median(err_nn))
    err_nn_25_quantile.append(
            np.quantile(err_nn,[.25])[0]
        )
    err_nn_75_quantile.append(
        np.quantile(err_nn,[.75])[0]
    )

sns.set_context('talk')
fig, ax = plt.subplots(1,2, figsize=(16,8))

ax[0].plot(sample_size, dist_kdf_med, c="r", label='KDF')
#ax[0].plot(sample_size, dist_kdf_bic_med, c="b", label='KDF (bic)')
ax[0].plot(sample_size, dist_rf_med, c="k", label='RF')

ax[0].fill_between(sample_size, dist_kdf_25_quantile, dist_kdf_75_quantile, facecolor='r', alpha=.3)
ax[0].fill_between(sample_size, dist_rf_25_quantile, dist_rf_75_quantile, facecolor='k', alpha=.3)
#ax[0].fill_between(sample_size, dist_kdf_bic_25_quantile, dist_kdf_bic_75_quantile, facecolor='b', alpha=.3)

ax[0].set_xscale('log')
ax[0].set_xlabel('Sample size')
ax[0].set_ylabel('Hellinger Distance')

ax[1].plot(sample_size, err_kdf_med, c="r", label='KDF')
#ax[1].plot(sample_size, err_kdf_bic_med, c="b", label='KDF (bic)')
ax[1].plot(sample_size, err_rf_med, c="k", label='RF')

ax[1].fill_between(sample_size, err_kdf_25_quantile, err_kdf_75_quantile, facecolor='r', alpha=.3)
ax[1].fill_between(sample_size, err_rf_25_quantile, err_rf_75_quantile, facecolor='k', alpha=.3)

ax[0].plot(sample_size, dist_nn_med, c=c[0], label='NN')
ax[0].fill_between(sample_size, dist_nn_25_quantile, dist_nn_75_quantile, facecolor=c[0], alpha=.3)

ax[1].plot(sample_size, err_nn_med, c=c[0], label='NN')
ax[1].fill_between(sample_size, err_nn_25_quantile, err_nn_75_quantile, facecolor=c[0], alpha=.3)

for ii, cov_type in enumerate(covariance_types):
    filename = 'sim_res/simulation_res_nn_' + cov_type + '.csv'
    df = pd.read_csv(filename)   

    for sample in sample_size:
        res_kdn = df['hellinger dist kdn'][df['sample']==sample]
        err_kdn = df['error kdn'][df['sample']==sample]
        #res_rf = df['hellinger dist rf'][df['sample']==sample]

        dist_kdn_med.append(np.median(res_kdn))
        #dist_rf_med.append(np.median(res_rf))
        dist_kdn_25_quantile.append(
            np.quantile(res_kdn,[.25])[0]
        )
        dist_kdn_75_quantile.append(
            np.quantile(res_kdn,[.75])[0]
        )

        err_kdn_med.append(np.median(err_kdn))
        err_kdn_25_quantile.append(
                np.quantile(err_kdn,[.25])[0]
            )
        err_kdn_75_quantile.append(
            np.quantile(err_kdn,[.75])[0]
        )

    ax[0].plot(sample_size, dist_kdn_med, c=c[1], label='kdn')
    ax[0].fill_between(sample_size, dist_kdn_25_quantile, dist_kdn_75_quantile, facecolor=c[1], alpha=.3)
    ax[1].plot(sample_size, err_kdn_med, c=c[1], label='kdn')
    ax[1].fill_between(sample_size, err_kdn_25_quantile, err_kdn_75_quantile, facecolor=c[1], alpha=.3)

ax[0].set_xscale('log')
ax[0].set_xlabel('Sample size')
ax[0].set_ylabel('Hellinger Distance')
#ax[0].legend()

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

plt.savefig('plots/sim_res_kdn_kdf.pdf')


# %%

