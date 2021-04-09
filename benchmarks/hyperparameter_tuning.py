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
import matplotlib.pyplot as plt 
import seaborn as sns
# %%
task = openml.tasks.get_task(6)
X, y = task.get_X_and_y()
ntrees = range(1,42,2)
skf = StratifiedKFold(n_splits=5)
err = []

for trees in ntrees:
    print('Doing %d trees'%trees)
    err_ = []
    for train_index, test_index in skf.split(X, y):
        model_kdf = kdf({'n_estimators':trees})
        model_kdf.fit(X[train_index], y[train_index])

        err_.append(
            1-np.mean(y[test_index]==model_kdf.predict(X[test_index]))
        )
    err.append(
        np.mean(err_)
    )

sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(ntrees,err)
ax.set_xlabel('ntrees')
ax.set_ylabel('Generalization Error')
plt.savefig('plots/tree_vs_error.pdf')
# %%
