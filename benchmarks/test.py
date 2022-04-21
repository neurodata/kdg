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
from sklearn import preprocessing

# %%
#task = openml.tasks.get_task(14969)
#X, y = task.get_X_and_y()
#X = X[:,4:5]
df = pd.read_csv('uci_dataset/spambase.data')
df_ = np.array(df)
X, y = np.double(df_[:,:2]), df[[' Class']]
_, y = np.unique(y, return_inverse=True)
#scaler = preprocessing.StandardScaler().fit(X)
#X = scaler.transform(X)

skf = StratifiedKFold(n_splits=3)
train_index, test_index = list(skf.split(X, y))[0]

model_kdf = kdf(k=10000000, kwargs={'n_estimators':500})
model_kdf.fit(X[train_index], y[train_index])

#model_rf = rf(n_estimators=500).fit(X[train_index], y[train_index])
# %%
test_acc = np.mean(model_kdf.predict(X[test_index])==y[test_index])
test_acc2 = np.mean(model_kdf.rf_model.predict(X[test_index])==y[test_index])
# %%
