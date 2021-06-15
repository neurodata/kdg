#%%
from kdg import kdn
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
import keras
from keras import layers

#%%
def get_stratified_samples(y, samples_to_take):
    labels = np.unique(y)
    sample_per_class = int(np.floor(samples_to_take/len(labels)))

    if sample_per_class < len(np.where(y==labels[0])[0]):
        stratified_indices = np.random.choice(
            (
            np.where(y==labels[0])[0]
            ), 
            sample_per_class,
            replace = False
        )
    else:
        stratified_indices = np.random.choice(
            (
            np.where(y==labels[0])[0]
            ), 
            sample_per_class,
            replace = True
        )

    for lbl in labels[1:]:
        if sample_per_class < len(np.where(y==lbl)[0]):
            _stratified_indices = np.random.choice(
                (
                np.where(y==lbl)[0]
                ), 
                sample_per_class,
                replace = False
            )
        else:
            _stratified_indices = np.random.choice(
                (
                np.where(y==lbl)[0]
                ), 
                sample_per_class,
                replace = True
            )

        stratified_indices = np.concatenate(
            (stratified_indices, _stratified_indices),
            axis=0
        )
    return stratified_indices

# %%
def experiment(task_id, cov_type, criterion, folder, n_estimators=500, cv=5, reps=10):
    df = pd.DataFrame() 
    #task_id = 14
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()

    if np.isnan(np.sum(y)):
        return

    if np.isnan(np.sum(X)):
        return
    
    dim = X.shape[1]
    max_class = len(np.unique(y))
    max_sample = min(np.floor(len(y)*(cv-1.1)/cv),10000)
    sample_size = np.logspace(
        np.log10(max_class*2),
        np.log10(max_sample),
        num=10,
        endpoint=True,
        dtype=int
        )

    mean_nn = np.zeros((len(sample_size),cv), dtype=float)
    mean_kdn = np.zeros((len(sample_size),cv), dtype=float)
    mean_ece_nn = np.zeros((len(sample_size),cv), dtype=float)
    mean_ece_kdn = np.zeros((len(sample_size),cv), dtype=float)
    mean_kappa_nn = np.zeros((len(sample_size),cv), dtype=float)
    mean_kappa_kdn = np.zeros((len(sample_size),cv), dtype=float)
    folds = []
    samples = []
    dims = []

    error_nn = np.zeros((len(sample_size),reps), dtype=float)
    error_kdn = np.zeros((len(sample_size),reps), dtype=float)
    ece_nn = np.zeros((len(sample_size),reps), dtype=float)
    ece_kdn = np.zeros((len(sample_size),reps), dtype=float)
    kappa_nn = np.zeros((len(sample_size),reps), dtype=float)
    kappa_kdn = np.zeros((len(sample_size),reps), dtype=float)

    skf = StratifiedKFold(n_splits=cv)

    fold = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        total_sample = X_train.shape[0]

        for jj,sample in enumerate(sample_size):
            #print('sample numer'+str(sample))

            if total_sample<sample:
                continue

            for ii in range(reps):
                network = keras.Sequential()
                #network.add(layers.Dense(2, activation="relu", input_shape=(2)))
                network.add(layers.Dense(dim+2, activation='relu', input_shape=(dim,)))
                network.add(layers.Dense(dim+2, activation='relu'))
                network.add(layers.Dense(units=max_class, activation = 'softmax'))

                network_base = keras.Sequential()
                network_base.add(layers.Dense(dim+2, activation='relu', input_shape=(dim,)))
                network_base.add(layers.Dense(dim+2, activation='relu'))
                network_base.add(layers.Dense(units=max_class, activation = 'softmax'))

                train_idx =  get_stratified_samples(y_train, sample)

                network_base.fit(X_train[train_idx], keras.utils.to_categorical(y_train[train_idx]), epochs=100, batch_size=32, verbose=False)
                proba_nn = network_base.predict_proba(X_test)
                predicted_label = np.argmax(proba_nn, axis = 1)
                ece_nn[jj][ii] = get_ece(proba_nn, predicted_label, y_test)
                error_nn[jj][ii] = 1 - np.mean(y_test==predicted_label)
                kappa_nn[jj][ii] = cohen_kappa_score(predicted_label, y_test)

                model_kdn = kdn(network, covariance_types = cov_type, criterion=criterion, kwargs={'n_estimators':n_estimators})
                model_kdn.fit(X_train[train_idx], y_train[train_idx])
                proba_kdn = model_kdn.predict_proba(X_test)
                predicted_label = np.argmax(proba_kdn, axis = 1)
                ece_kdn[jj][ii] = get_ece(proba_kdn, predicted_label, y_test)
                error_kdn[jj][ii] = 1 - np.mean(y_test==predicted_label)    
                kappa_kdn[jj][ii] = cohen_kappa_score(predicted_label, y_test)

            mean_nn[jj][fold] = np.mean(error_nn[jj])   
            #var_rf[jj] = np.var(error_rf[jj], ddof=1)
            mean_kdn[jj][fold] = np.mean(error_kdn[jj])   
            #var_kdf[jj] = np.var(error_kdf[jj], ddof=1)
            mean_kappa_nn[jj][fold] = np.mean(kappa_nn[jj])
            mean_kappa_kdn[jj][fold] = np.mean(kappa_kdn[jj])

            mean_ece_nn[jj][fold] = np.mean(ece_nn[jj])   
            #var_ece_rf[jj] = np.var(ece_rf[jj], ddof=1)
            mean_ece_kdn[jj][fold] = np.mean(ece_kdn[jj])   
            #var_ece_kdf[jj] = np.var(ece_kdf[jj], ddof=1)
            folds.append(fold)
            samples.append(sample)
            dims.append(dim)
        fold += 1

    df['error_nn'] = np.ravel(mean_nn)
    df['error_kdn'] = np.ravel(mean_kdn)
    df['kappa_nn'] = np.ravel(mean_kappa_nn)
    df['kappa_kdn'] = np.ravel(mean_kappa_kdn)
    df['ece_nn'] = np.ravel(mean_ece_nn)
    df['ece_kdn'] = np.ravel(mean_ece_kdn)
    df['fold'] = folds
    df['sample'] = samples
    df['dimension'] = dims

    df.to_csv(folder+'/'+'openML_cc18_task_bic_'+str(task_id)+'.csv')

#%%
folder = 'result_cov_nn'
os.mkdir(folder)
cv = 5
reps = 10
n_estimators = 500
n_cores = 1
df = pd.DataFrame() 
benchmark_suite = openml.study.get_suite('OpenML-CC18')
covarice_types = {"full", "tied", "diag", "spherical"}
criterion = 'bic'
#%%
total_cores = multiprocessing.cpu_count()
assigned_workers = total_cores//n_cores

for task_id in benchmark_suite.tasks:
    print('Doing taks '+str(task_id))
    experiment(
        task_id,
        covarice_types,
        criterion,
        folder
        )

# %%
