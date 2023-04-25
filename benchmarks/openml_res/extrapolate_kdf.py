#%%
from kdg import kdf
from kdg.utils import get_ece
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import numpy as np
import openml
from sklearn.metrics import cohen_kappa_score
from kdg.utils import get_ece, plot_reliability
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
# %%
def experiment(dataset_id):
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
    
    total_sample = len(y)
    max_norm = np.max(
        np.linalg.norm(X, 2, axis=1)
    )
    X /= max_norm
    norms = np.linalg.norm(X, 2, axis=1)
    sorted_id = np.argsort(norms)

    test_percentile = np.arange(.5,1.01,.1)
    train_sample = int(total_sample*0.5)
    test_sample = [int(total_sample*percentile) for percentile in test_percentile]

    model_kdf = kdf(kwargs={'n_estimators':500})
    model_kdf.fit(X[sorted_id[:train_sample]], y[sorted_id[:train_sample]])

    ECE_rf = []
    ECE_kdf = []
    error_rf = []
    error_kdf = []
    mean_max_conf_rf = []
    mean_max_conf_kdf = []

    prev_id = 0
    for sample in test_sample:
        predicted_proba_kdf = model_kdf.predict_proba(
            X[sorted_id[prev_id:sample]]
        )
        predicted_proba_rf = model_kdf.rf_model.predict_proba(
            X[sorted_id[prev_id:sample]]
        )
        predicted_label_kdf = np.argmax(predicted_proba_kdf, axis=1)
        predicted_label_rf = np.argmax(predicted_proba_rf, axis=1)

        ECE_rf.append(
            get_ece(predicted_proba_rf,
                    predicted_label_rf,
                    y[prev_id:sample])
        )
        ECE_kdf.append(
            get_ece(predicted_proba_kdf,
                    predicted_proba_kdf,
                    y[prev_id:sample])
        )
        error_rf.append(
            1-np.mean(predicted_label_rf==y[prev_id:sample])
        )
        error_kdf.append(
            1-np.mean(predicted_label_kdf==y[prev_id:sample])
        )

        mean_max_conf_rf.append(
            np.mean(np.max(predicted_proba_rf, axis=1))
        )
        mean_max_conf_kdf.append(
            np.mean(np.max(predicted_proba_kdf, axis=1))
        )

        prev_id = sample

    return ECE_rf, ECE_kdf, error_rf, error_kdf, mean_max_conf_rf, mean_max_conf_kdf

#%%
benchmark_suite = openml.study.get_suite('OpenML-CC18')
res = Parallel(n_jobs=-1,verbose=1)(
            delayed(experiment)(
                    dataset_id,
                    ) for dataset_id in benchmark_suite.data
                )
# %%
total_datasets = len(res)
ECE_rf = np.zeros(6, dtype=float)
ECE_kdf = np.zeros(6, dtype=float)
error_rf = np.zeros(6, dtype=float)
error_kdf = np.zeros(6, dtype=float)
mean_max_conf_rf = np.zeros(6, dtype=float)
mean_max_conf_kdf = np.zeros(6, dtype=float)

for ii in range(total_datasets):
    ECE_rf += res[ii][0]
    ECE_kdf += res[ii][1]
    error_rf += res[ii][2]
    error_kdf += res[ii][3]
    mean_max_conf_rf += res[ii][4]
    mean_max_conf_kdf += res[ii][5]

ECE_rf /= total_datasets
ECE_kdf /= total_datasets
error_rf /= total_datasets
error_kdf /= total_datasets
mean_max_conf_rf /= total_datasets
mean_max_conf_kdf /= total_datasets

#%%
test_percentile = np.arange(.5,1.01,.1)
sns.set_context('talk')
fig1, ax = plt.subplots(1, 3, figsize=(24, 8))

ax[0].plot(test_percentile, mean_max_conf_rf, c='k', label='RF')
ax[0].plot(test_percentile, mean_max_conf_kdf, c='r', label='KDF')
ax[0].set_ylabel('Mean Max Confidence')
ax[0].set_xlabel('Data Percentile')

ax[1].plot(test_percentile, ECE_rf, c='k', label='RF')
ax[1].plot(test_percentile, ECE_kdf, c='r', label='KDF')
ax[1].set_ylabel('ECE')
ax[1].set_xlabel('Data Percentile')


ax[2].plot(test_percentile, error_rf, c='k', label='RF')
ax[2].plot(test_percentile, error_kdf, c='r', label='KDF')
ax[2].set_ylabel('Generalization Error')
ax[2].set_xlabel('Data Percentile')
# %%
