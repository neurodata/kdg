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
from kdg.utils import get_ece, sample_unifrom_circle
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
# %%
root_dir = "openml_kdf_res_ood"

try:
    os.mkdir(root_dir)
except:
    print("directory already exists!!!")
# %%
def experiment(dataset_id, n_estimators=500, reps=10, random_state=42):
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
    
    X /= np.max(
        np.linalg.norm(X, 2, axis=1)
    )
    _, y = np.unique(y, return_inverse=True)
    
        
    total_sample = X.shape[0]
    test_sample = total_sample//3
    train_sample = total_sample-test_sample

    r = []    
    conf_rf = []
    conf_kdf = []
    distances = np.arange(0, 11, 1)

    for rep in range(reps):
        X_train, _, y_train, _ = train_test_split(
                     X, y, test_size=test_sample, train_size=train_sample, random_state=random_state+rep)
        model_kdf = kdf(kwargs={'n_estimators':n_estimators})
        model_kdf.fit(X_train, y_train)
        
        for distance in distances:
            X_ood = sample_unifrom_circle(1000, distance)
            proba_kdf = model_kdf.predict_proba(X_ood)
            proba_rf = model_kdf.rf_model.predict_proba(X_ood)
            

            conf_rf.append(
                np.mean(
                    np.max(proba_rf, axis=1)
                )
            )
            conf_kdf.append(
                np.mean(
                    np.mean(
                        np.max(proba_kdf, axis=1)
                    )
                )
            )
            r.append(
                distance
            )
            

    df = pd.DataFrame() 
    df['conf_kdf'] = conf_kdf
    df['conf_rf'] = conf_rf
    df['distance'] = r

    filename = 'Dataset_' + str(dataset_id) + '.csv'
    df.to_csv(os.path.join(root_dir, filename))

# %%
benchmark_suite = openml.study.get_suite('OpenML-CC18')

Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment)(
                dataset_id,
                ) for dataset_id in openml.study.get_suite("OpenML-CC18").data
            )
