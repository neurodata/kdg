#%%
from kdg import kdf
from kdg.utils import get_ece
import openml
import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import cohen_kappa_score
import os
from kdg.utils import generate_gaussian_parity, pdf, hellinger
# %%
reps = 100
n_estimators = 500
sample_size = np.logspace(
        np.log10(10),
        np.log10(5000),
        num=10,
        endpoint=True,
        dtype=int
)

#%%
def experiment_kdf(sample, n_estimators=500):
    X, y = generate_gaussian_parity(sample, cluster_std=0.5)
    X_test, y_test = generate_gaussian_parity(1000, cluster_std=0.5)
    p = np.arange(-1,1,step=0.006)
    q = np.arange(-1,1,step=0.006)
    xx, yy = np.meshgrid(p,q)
    grid_samples = np.concatenate(
            (
                xx.reshape(-1,1),
                yy.reshape(-1,1)
            ),
            axis=1
    ) 
    model_kdf = kdf(kwargs={'n_estimators':n_estimators})
    model_kdf.fit(X, y)
    proba_kdf = model_kdf.predict_proba(grid_samples)
    true_pdf_class1 = np.array([pdf(x, cov_scale=0.5) for x in grid_samples]).reshape(-1,1)
    true_pdf = np.concatenate([true_pdf_class1, 1-true_pdf_class1], axis = 1)

    error = 1 - np.mean(model_kdf.predict(X_test)==y_test)
    return hellinger(proba_kdf, true_pdf), error

def experiment_rf(sample, n_estimators=500):
    X, y = generate_gaussian_parity(sample, cluster_std=0.5)
    X_test, y_test = generate_gaussian_parity(1000, cluster_std=0.5)
    p = np.arange(-1,1,step=0.006)
    q = np.arange(-1,1,step=0.006)
    xx, yy = np.meshgrid(p,q)
    grid_samples = np.concatenate(
            (
                xx.reshape(-1,1),
                yy.reshape(-1,1)
            ),
            axis=1
    ) 
    model_rf = rf(n_estimators=n_estimators).fit(X, y)
    proba_rf = model_rf.predict_proba(grid_samples)
    true_pdf_class1 = np.array([pdf(x, cov_scale=0.5) for x in grid_samples]).reshape(-1,1)
    true_pdf = np.concatenate([true_pdf_class1, 1-true_pdf_class1], axis = 1)

    error = 1 - np.mean(model_rf.predict(X_test)==y_test)
    return hellinger(proba_rf, true_pdf), error

#%%
df = pd.DataFrame()
hellinger_dist_kdf = []
hellinger_dist_rf = []
err_kdf = []
err_rf = []
sample_list = []
    
for sample in sample_size:
    print('Doing sample %d'%sample)

    res_kdf = Parallel(n_jobs=-1)(
                delayed(experiment_kdf)(
                sample
                ) for _ in range(reps)
            )

    res_rf = Parallel(n_jobs=-1)(
        delayed(experiment_rf)(
                sample
                ) for _ in range(reps)
            )
    
    for ii in range(reps):
        hellinger_dist_kdf.append(
                res_kdf[ii][0]
            )
        err_kdf.append(
                res_kdf[ii][1]
            )

        hellinger_dist_rf.append(
                res_rf[ii][0]
            )
        err_rf.append(
                res_rf[ii][1]
            )

    sample_list.extend([sample]*reps)

df['hellinger dist kdf'] = hellinger_dist_kdf
df['hellinger dist rf'] = hellinger_dist_rf
df['error kdf'] = err_kdf
df['error rf'] = err_rf
df['sample'] = sample_list
df.to_csv('simulation_res.csv')
# %%
import seaborn as sns
import matplotlib.pyplot as plt

p = np.arange(-1,1,step=0.06)
q = np.arange(-1,1,step=0.06)
xx, yy = np.meshgrid(p,q)
tmp = np.ones(xx.shape)

grid_samples = np.concatenate(
            (
                xx.reshape(-1,1),
                yy.reshape(-1,1),
                tmp.reshape(-1,1),
                tmp.reshape(-1,1),
                tmp.reshape(-1,1)
            ),
            axis=1
    ) 
#model_kdf = kdf(kwargs={'n_estimators':500})
#model_kdf.fit(X, y)
proba_kdf = model_kdf.rf_model.predict_proba(grid_samples)

data = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':proba_kdf[:,0]})
data = data.pivot(index='x', columns='y', values='z')


sns.set_context("talk")
fig, ax = plt.subplots(1,1, figsize=(8,8))
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
ax1.set_xticklabels(['-1','' , '', '', '', '', '','','','','0','','','','','','','','','1'])
ax1.set_yticklabels(['-1','' , '', '', '', '', '','','','','','','0','','','','','','','','','','','','','1'])
#ax1.set_yticklabels(['-1','' , '', '', '', '', '','','','' , '', '', '', '', '', '','','','','', '0','','' , '', '', '', '', '','','','','','','','','','','','','1'])
ax.set_title('Estimated PDF of xor-nxor simulation data',fontsize=24)
ax.invert_yaxis()
# %%
