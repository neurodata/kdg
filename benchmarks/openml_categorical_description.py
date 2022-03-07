#%%
from kdg import kdf
from kdg.utils import get_ece
import openml
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import cohen_kappa_score
from kdg.utils import get_ece
import os
from os import listdir, getcwd 
# %%
benchmark_suite = openml.study.get_suite('OpenML-CC18')
df = pd.DataFrame()
task_ids = []
total_features = []
number_of_categorical = []

for task_id in openml.study.get_suite("OpenML-CC18").data:
    try:
        dataset = openml.datasets.get_dataset(task_id)
        X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )
        task_ids.append(
            task_id
        )
        total_features.append(
            X.shape[1]
        )
        number_of_categorical.append(
            np.mean(is_categorical)*total_features[-1]
        )
    except:
        print('Could not load ', task_id)
    else:
        print('Loaded successfully!')
# %%
df['Task_id'] = task_ids
df['total_features'] = total_features
df['categorical_features'] = number_of_categorical
df.to_csv('openml_res/openML_cc18_details.csv')
# %%
