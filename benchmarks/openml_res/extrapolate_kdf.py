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
# %% load the data
dataset_id = 12

dataset = openml.datasets.get_dataset(dataset_id)
X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )
# %% normalize and sort the data
max_norm = np.max(
    np.linalg.norm(X, 2, axis=1)
)
X /= max_norm
norms = np.linalg.norm(X, 2, axis=1)

idx_to_train = np.where(norms<=0.9)[0]
idx_to_test_subsample = np.where(norms>0.9)[0]
# %%
labels = np.unique(y)
model_kdf = kdf(kwargs={'n_estimators':500})
model_kdf.fit(X[idx_to_train], y[idx_to_train], epsilon=1e-6)

# %%
#model_kdf.global_bias = 10
proba = model_kdf.predict_proba(X[idx_to_test_subsample])
max_proba = np.max(proba, axis=1)
predicted_label = np.argmax(proba, axis=1)

proba_rf = model_kdf.rf_model.predict_proba(X[idx_to_test_subsample])
max_proba_rf = np.max(proba_rf, axis=1)
predicted_label_rf = np.argmax(proba_rf, axis=1)

print("ECE KDF:", get_ece(proba, predicted_label, y[idx_to_test_subsample]))
print("ECE RF:", get_ece(proba_rf, predicted_label_rf, y[idx_to_test_subsample]))

norms_test = np.linalg.norm(X[idx_to_test_subsample], 2, axis=1)
sorted_idx = np.argsort(norms_test)

sns.set_context('talk')
plt.plot(norms_test[sorted_idx], max_proba[sorted_idx])
plt.xlabel('Norm')
plt.ylabel('Max Conf')
# %%
labels_to_train = [0,1]
idx_to_train = []
labels_to_test = [4,5]
idx_to_test = []

for label in labels_to_train:
    idx_to_train.extend(
        np.where(y==label)[0]
    )

for label in labels_to_test:
    idx_to_test.extend(
        np.where(y==label)[0]
    )
# %%
model_kdf = kdf(kwargs={'n_estimators':500})
model_kdf.fit(X[idx_to_train], y[idx_to_train], epsilon=1e-6)

# %%
