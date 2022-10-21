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
# %%
cv = 5
offset = 1e3
task_id = 11
delta = 1e1
# %%
skf = StratifiedKFold(n_splits=cv)
task = openml.tasks.get_task(task_id)
X, y = task.get_X_and_y()
acc = []
mean_max_proba = []
acc_rf = []
mean_max_proba_rf = []

mean_max_proba_ood = []
mean_max_proba_rf_ood = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model_kdf = kdf(k=1e-3,kwargs={'n_estimators':500})
    model_kdf.fit(X_train, y_train)

    acc.append(
        np.mean(model_kdf.predict(X_test)==y_test)
    )
    mean_max_proba.append(
        np.mean(np.max(model_kdf.predict_proba(X_test), axis=1))
    )

    acc_rf.append(
        np.mean(model_kdf.rf_model.predict(X_test)==y_test)
    )
    mean_max_proba_rf.append(
        np.mean(np.max(model_kdf.rf_model.predict_proba(X_test), axis=1))
    )

    X_test[:,0] = np.max(X_test[:,0]) + delta

    mean_max_proba_ood.append(
        np.mean(np.max(model_kdf.predict_proba(X_test), axis=1))
    )
    mean_max_proba_rf_ood.append(
        np.mean(np.max(model_kdf.rf_model.predict_proba(X_test), axis=1))
    )
# %%
