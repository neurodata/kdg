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
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
# %%
root_dir = "openml_kdf_res"

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
    
    min_val = np.min(X,axis=0)
    max_val = np.max(X, axis=0)
    
    X = (X-min_val)/(max_val-min_val+1e-12)
    _, y = np.unique(y, return_inverse=True)
    
    '''for ii in range(X.shape[1]):
        unique_val = np.unique(X[:,ii])
        if len(unique_val) < 10:
            return'''
        
    total_sample = X.shape[0]
    test_sample = total_sample//3
    train_samples = np.logspace(
            np.log10(10),
            np.log10(total_sample-test_sample),
            num=5,
            endpoint=True,
            dtype=int
        )
    err = []
    err_rf = []
    ece = []
    ece_rf = []
    mc_rep = []
    samples = []

    for train_sample in train_samples:
        for rep in range(reps):
            X_train, X_test, y_train, y_test = train_test_split(
                     X, y, test_size=test_sample, train_size=train_sample, random_state=random_state+rep)
            
            
            model_kdf = kdf(k=1, kwargs={'n_estimators':n_estimators})
            model_kdf.fit(X_train, y_train, epsilon=1e-2)
            proba_kdf = model_kdf.predict_proba(X_test)
            proba_rf = model_kdf.rf_model.predict_proba(X_test)
            predicted_label_kdf = np.argmax(proba_kdf, axis = 1)
            predicted_label_rf = np.argmax(proba_rf, axis = 1)

            err.append(
                1 - np.mean(
                        predicted_label_kdf==y_test
                    )
            )
            err_rf.append(
                1 - np.mean(
                    predicted_label_rf==y_test
                )
            )
            ece.append(
                get_ece(proba_kdf, predicted_label_kdf, y_test)
            )
            ece_rf.append(
                get_ece(proba_rf, predicted_label_rf, y_test)
            )
            samples.append(
                train_sample
            )
            mc_rep.append(rep)

    df = pd.DataFrame() 
    df['err_kdf'] = err
    df['err_rf'] = err_rf
    df['ece_kdf'] = ece
    df['ece_rf'] = ece_rf
    df['rep'] = mc_rep
    df['samples'] = samples

    filename = 'Dataset_' + str(dataset_id) + '.csv'
    df.to_csv(os.path.join(root_dir, filename))

# %%
benchmark_suite = openml.study.get_suite('OpenML-CC18')

'''Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment)(
                dataset_id,
                ) for dataset_id in openml.study.get_suite("OpenML-CC18").data
            )'''

data_id_not_done = [28, 554, 1485, 40996, 41027, 23517, 40923, 40927]

for dataset_id in data_id_not_done:
    print("Doing ", dataset_id)
    experiment(dataset_id)