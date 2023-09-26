#%%
from kdg import kdf
from kdg.utils import get_ece, get_ace
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV as calcv
# %%
root_dir = "openml_kdn_res_baseline"

try:
    os.mkdir(root_dir)
except:
    print("directory already exists!!!")

# %%
def experiment(dataset_id, n_estimators=500, reps=10, random_state=42):
    filename = 'Dataset_' + str(dataset_id) + '.csv'
    if os.path.exists(os.path.join(root_dir, filename)):
        return
    
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
    
    print(dataset_id)
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
            np.log10(100),
            np.log10(total_sample-test_sample),
            num=4,
            endpoint=True,
            dtype=int
        )
    err_isotonic = []
    ece_isotonic = []
    err_sigmoid = []
    ece_sigmoid = []
    mc_rep = []
    samples = []

    for train_sample in train_samples:
        for rep in range(reps):
            X_train, X_test, y_train, y_test = train_test_split(
                     X, y, test_size=test_sample, train_size=train_sample, random_state=random_state+rep, stratify=y)
            
            if train_sample >= 1000:
                X_train, X_cal, y_train, y_cal = train_test_split(
                X_train, y_train, train_size=0.6666666666666666, random_state=random_state+rep, stratify=y_train)
            else:
                X_train, X_cal, y_train, y_cal = train_test_split(
                X_train, y_train, train_size=0.8, random_state=random_state+rep)

            uncalibrated_rf = RandomForestClassifier(random_state=random_state+rep, n_estimators=500)
            uncalibrated_rf.fit(X_train, y_train)
            calibrated_rf_isotonic = calcv(uncalibrated_rf, method = 'isotonic', ensemble=False, cv='prefit')
            calibrated_rf_isotonic.fit(X_cal, y_cal)

            calibrated_rf_sigmoid = calcv(uncalibrated_rf, method = 'sigmoid', ensemble=False, cv='prefit')
            calibrated_rf_sigmoid.fit(X_cal, y_cal)

            y_proba_isotonic = calibrated_rf_isotonic.predict_proba(X_test)
            y_hat_isotonic = np.argmax(y_proba_isotonic, axis=1)

            y_proba_sigmoid = calibrated_rf_sigmoid.predict_proba(X_test)
            y_hat_sigmoid = np.argmax(y_proba_sigmoid, axis=1)


            err_isotonic.append(
                1 - np.mean(
                        y_hat_isotonic==y_test
                    )
            )
            ece_isotonic.append(
                get_ece(y_proba_isotonic, y_test)
            )
            err_sigmoid.append(
                1 - np.mean(
                        y_hat_sigmoid==y_test
                    )
            )
            ece_sigmoid.append(
                get_ece(y_proba_sigmoid, y_test)
            )
            samples.append(
                train_sample
            )
            mc_rep.append(rep)  

    df = pd.DataFrame() 
    df['err_isotonic'] = err_isotonic
    df['ece_isotonic'] = ece_isotonic
    df['err_sigmoid'] = err_sigmoid
    df['ece_sigmoid'] = ece_sigmoid
    df['rep'] = mc_rep
    df['samples'] = samples

    filename = 'Dataset_' + str(dataset_id) + '.csv'
    df.to_csv(os.path.join(root_dir, filename))  

#%%
benchmark_suite = openml.study.get_suite('OpenML-CC18')
#data_id_not_done = [28, 554, 1485, 40996, 41027, 23517, 40923, 40927]

Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment)(
                dataset_id,
                ) for dataset_id in benchmark_suite.data
            )              