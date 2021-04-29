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
reps = 1000
n_estimators = 500
sample_size = np.logspace(
        np.log10(10),
        np.log10(10000),
        num=10,
        endpoint=True,
        dtype=int
        )
delta = 0.01
p = np.arange(-2,2,step=delta)
q = np.arange(-2,2,step=delta)
xx, yy = np.meshgrid(p,q)
grid_samples = np.concatenate(
            (
                xx.reshape(-1,1),
                yy.reshape(-1,1)
            ),
            axis=1
) 
true_pdf_class1 = np.array([pdf(x) for x in grid_samples]).reshape(-1,1)
true_pdf = np.concatenate([true_pdf_class1, 1-true_pdf_class1], axis = 1)
#%%
df = pd.DataFrame()
hellinger_dist_kdf = []
hellinger_dist_rf = []
sample_list = []

for sample in sample_size:
    for i in range(reps):
        X, y = generate_gaussian_parity(sample)
        model_kdf = kdf({'n_estimators':n_estimators})
        model_kdf.fit(X, y)
        model_rf = rf(n_estimators=n_estimators).fit(X, y)

        proba_kdf = model_kdf.predict_proba(grid_samples)
        proba_rf = model_rf.predict_proba(grid_samples)

        hellinger_dist_kdf.append(hellinger(proba_kdf, true_pdf))
        hellinger_dist_rf.append(hellinger(proba_rf, true_pdf))
        sample_list.append(sample)

df['hellinger dist kdf'] = hellinger_dist_kdf
df['hellinger dist rf'] = hellinger_dist_rf
df['sample'] = sample_list

        