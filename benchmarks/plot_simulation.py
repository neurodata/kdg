#%%
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np
import openml
from scipy.interpolate import interp1d
from os import listdir, getcwd 

#%%
df = pd.read_csv('simulation_res_10000.csv')
sample_size = np.logspace(
        np.log10(10),
        np.log10(10000),
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
plt.savefig('plots/sim_res.pdf')
# %%
