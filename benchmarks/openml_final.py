#%%
from kdg import kdf
from kdg.utils import get_ece
import openml
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import StratifiedKFold as sk
from sklearn.metrics import cohen_kappa_score
from kdg.utils import get_ece
import os
from os import listdir, getcwd 
# %%
def count_rf_param(rf_model):
    total_param = 0
    for tree in rf_model.estimators_:
        nodes = tree.tree_.node_count
        leaf_node = np.sum(tree.tree_.children_left)
        total_param += tree.tree_.value.shape[2]*leaf_node\
            + (nodes-leaf_node)*2 + nodes
    return total_param

def count_kdf_param(kdf_model):
    total_param = 0

    for label in kdf_model.labels:
        total_param += len(kdf_model.polytope_cardinality[label])
    
    total_param += len(kdf_model.polytope_cardinality[label])\
            *(kdf_model.feature_dim*2)

    return total_param


def experiment(dataset_id, cv_fold = 5, n_estimators=500):
    #print(dataset_id)
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )

    if np.mean(is_categorical) >0:
        return

    if np.isnan(np.sum(y)):
        return

    if np.isnan(np.sum(X)):
        return

    skf = sk(n_splits=cv_fold)
    err_kdf = 0
    err_rf = 0
    ece_kdf = 0
    ece_rf = 0
    param_kdf = 0
    param_rf = 0
    for train_idx, test_idx in skf.split(X, y):
        model_kdf = kdf(kwargs={'n_estimators':n_estimators})
        model_kdf.fit(X[train_idx], y[train_idx])
        proba_kdf = model_kdf.predict_proba(X[test_idx])
        proba_rf = model_kdf.rf_model.predict_proba(X[test_idx])
        predicted_label_kdf = np.argmax(proba_kdf, axis = 1)
        predicted_label_rf = np.argmax(proba_rf, axis = 1)

        err_kdf += 1 - np.mean(
                        predicted_label_kdf==y[test_idx]
                    )
        err_rf += 1 - np.mean(
                        predicted_label_rf==y[test_idx]
                    )
        ece_kdf += get_ece(proba_kdf, predicted_label_kdf, y[test_idx])
        ece_rf += get_ece(proba_rf, predicted_label_rf, y[test_idx])
        param_kdf += count_kdf_param(model_kdf)
        param_rf += count_rf_param(model_kdf.rf_model)

    
    return (err_rf-err_kdf)/cv_fold, (ece_rf-ece_kdf)/cv_fold, (param_rf/param_kdf)

#%%
benchmark_suite = openml.study.get_suite('OpenML-CC18')
res = Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment)(
                dataset_id
                ) for dataset_id in openml.study.get_suite("OpenML-CC18").data
            )

# %%
