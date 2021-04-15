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
import matplotlib.pyplot as plt
# %%
task = openml.tasks.get_task(3)
X, y = task.get_X_and_y()

np.random.seed(12345)
skf = StratifiedKFold(n_splits=5)
error = np.zeros(5, dtype=float)

for ii, [train_index, test_index] in enumerate(skf.split(X, y)):
    np.random.seed(12345)
    model_kdf = kdf({'n_estimators':500,'max_depth':2})
    model_kdf.fit(X[train_index], y[train_index])
    error[ii] = 1 - np.mean(y[test_index]==model_kdf.predict(X[test_index]))

# %%
error_will = np.array([0.1       , 0.09546166, 0.3114241 , 0.15649452, 0.52738654])
error_jd = np.array([0.0921875 , 0.08920188, 0.33020344, 0.14084507, 0.52269171])

error_diff = error_jd - error_will
plt.plot([1,2,3,4,5],error_diff,marker='.')
plt.ylabel('error_jd - error_will')
plt.xlabel('fold')
plt.xticks([1,2,3,4,5])
# %%
