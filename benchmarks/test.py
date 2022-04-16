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
from kdg.utils import get_ece
#%%
dataset_id = 12
dataset = openml.datasets.get_dataset(dataset_id)
X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )
#%%
unique_classes, counts = np.unique(y, return_counts=True)

test_sample = min(counts)//3

indx = []
for label in unique_classes:
    indx.append(
        np.where(
            y==label
        )[0]
    )

max_sample = min(counts) - test_sample
train_samples = np.logspace(
    np.log10(2),
    np.log10(max_sample),
    num=10,
    endpoint=True,
    dtype=int
    )

train_sample = train_samples[-1]
indx_to_take_train = []
indx_to_take_test = []

for ii, _ in enumerate(unique_classes):
    np.random.shuffle(indx[ii])
    indx_to_take_train.extend(
        list(
                indx[ii][:train_sample]
        )
    )
    indx_to_take_test.extend(
        list(
                indx[ii][-test_sample:counts[ii]]
        )
    )

model_kdf = kdf(k=1e6,kwargs={'n_estimators':500, 'min_samples_leaf':30})
model_kdf.fit(X[indx_to_take_train], y[indx_to_take_train])
# %%
def compute_pdf_1d(X, location, cov):
    return np.exp(-(X-location)**2/(2*cov))/(np.sqrt(2*np.pi*cov))
# %%
val = 1
pow = 0
for dim in range(X.shape[1]):
    location = model_kdf.polytope_means[0][0][dim]
    cov = model_kdf.polytope_cov[0][0][dim]
    
    val *= np.exp(model_kdf.pow_exp)*compute_pdf_1d(X[:1,dim], location, cov)


    print(val, pow)
# %%
np.mean(model_kdf.predict(X[indx_to_take_test])==y[indx_to_take_test])
# %%
