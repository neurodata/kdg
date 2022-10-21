#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
# %%
filename0 = 'high_dim_res_kdf_gaussian_empirical_cov.csv'

df = pd.read_csv(filename0)
err_rf_med = []
err_kdf_med = []
sample_size = [1000,5000,10000]
kdf_res = []

for sample in sample_size:
    err_rf = 1 - df['accuracy rf'][df['sample']==sample]
    err_kdf = 1 - df['accuracy kdf'][df['sample']==sample]

    err_rf_med.append(np.median(err_rf))
    err_kdf_med.append(np.median(err_kdf))

kdf_res.append(err_kdf_med)

for i in range(2,6):
    filename = 'high_dim_res_kdf_gaussian_' + str(i) + 'tree.csv'

    df = pd.read_csv(filename)
    err_kdf_med = []

    for sample in sample_size:
        err_kdf = 1 - df['accuracy kdf'][df['sample']==sample]
        err_kdf_med.append(np.median(err_kdf))
    kdf_res.append(err_kdf_med)
# %%
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(sample_size, err_rf_med, c="k", label='RF')

for ii,res in enumerate(kdf_res):
    ax.plot(sample_size, res, label='Threshold '+str(ii+1))

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xscale('log')
ax.set_yticks([.15,.30,.5])
ax.set_xlabel('Sample size')
ax.set_ylabel('error')
ax.legend(frameon=False)

plt.savefig('plots/high_dim_gaussian_robust_cov.pdf')

# %%
methods = ['empirical_cov', 'LedoitWolf', 'min_cov', 'oas']
sample_size = [1000,5000,10000]
res = []

for method in methods:
    filename = 'high_dim_res_kdf_gaussian_' + method + '.csv'
    df = pd.read_csv(filename)
    err_rf_med = []
    err_kdf_med = []

    for sample in sample_size:
        err_rf = 1 - df['accuracy rf'][df['sample']==sample]
        err_kdf = 1 - df['accuracy kdf'][df['sample']==sample]

        err_rf_med.append(np.median(err_rf))
        err_kdf_med.append(np.median(err_kdf))

    res.append(err_kdf_med)

#%%
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(sample_size, err_rf_med, c="k", label='RF')

for ii,kdf_res in enumerate(res):
    ax.plot(sample_size, kdf_res, label=methods[ii])

ax.plot([1000, 5000], [.499, 0.1715], label='Eliptical Envelope')
ax.plot([1000, 5000], [.496, 0.396], label='Graphical Lasso')

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xscale('log')
ax.set_yticks([.15,.30,.5])
ax.set_xlabel('Sample size')
ax.set_ylabel('error')
ax.legend(frameon=False)

plt.savefig('plots/high_dim_gaussian_all_cov.pdf')
# %%
