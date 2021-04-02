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
def experiment(task_id, n_estimators=500, cv=5, reps=10):
    df = pd.DataFrame() 
    #task_id = 14
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()

    if np.isnan(np.sum(y)):
        return

    if np.isnan(np.sum(X)):
        return
    
    sample_size = [10,100,500,1000]

    mean_rf = np.zeros((len(sample_size),cv), dtype=float)
    mean_kdf = np.zeros((len(sample_size),cv), dtype=float)
    mean_ece_rf = np.zeros((len(sample_size),cv), dtype=float)
    mean_ece_kdf = np.zeros((len(sample_size),cv), dtype=float)
    mean_kappa_rf = np.zeros((len(sample_size),cv), dtype=float)
    mean_kappa_kdf = np.zeros((len(sample_size),cv), dtype=float)
    folds = []
    samples = []

    error_rf = np.zeros((len(sample_size),reps), dtype=float)
    error_kdf = np.zeros((len(sample_size),reps), dtype=float)
    ece_rf = np.zeros((len(sample_size),reps), dtype=float)
    ece_kdf = np.zeros((len(sample_size),reps), dtype=float)
    kappa_rf = np.zeros((len(sample_size),reps), dtype=float)
    kappa_kdf = np.zeros((len(sample_size),reps), dtype=float)

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
                train_idx =  get_stratified_samples(y_train, sample)

                if len(train_idx) < 10:
                    return
                    
                model_rf = rf(n_estimators=n_estimators).fit(X_train[train_idx], y_train[train_idx])
                proba_rf = model_rf.predict_proba(X_test)
                predicted_label = np.argmax(proba_rf, axis = 1)
                ece_rf[jj][ii] = get_ece(proba_rf, predicted_label, y_test)
                error_rf[jj][ii] = 1 - np.mean(y_test==predicted_label)
                kappa_rf[jj][ii] = cohen_kappa_score(predicted_label, y_test)

                model_kdf = kdf({'n_estimators':n_estimators})
                model_kdf.fit(X_train[train_idx], y_train[train_idx])
                proba_kdf = model_kdf.predict_proba(X_test)
                predicted_label = np.argmax(proba_kdf, axis = 1)
                ece_kdf[jj][ii] = get_ece(proba_kdf, predicted_label, y_test)
                error_kdf[jj][ii] = 1 - np.mean(y_test==predicted_label)    
                kappa_kdf[jj][ii] = cohen_kappa_score(predicted_label, y_test)

            mean_rf[jj][fold] = np.mean(error_rf[jj])   
            #var_rf[jj] = np.var(error_rf[jj], ddof=1)
            mean_kdf[jj][fold] = np.mean(error_kdf[jj])   
            #var_kdf[jj] = np.var(error_kdf[jj], ddof=1)
            mean_kappa_rf[jj][fold] = np.mean(kappa_rf[jj])
            mean_kappa_kdf[jj][fold] = np.mean(kappa_kdf[jj])

            mean_ece_rf[jj][fold] = np.mean(ece_rf[jj])   
            #var_ece_rf[jj] = np.var(ece_rf[jj], ddof=1)
            mean_ece_kdf[jj][fold] = np.mean(ece_kdf[jj])   
            #var_ece_kdf[jj] = np.var(ece_kdf[jj], ddof=1)
            folds.append(fold)
            samples.append(sample)
        fold += 1

    df['error_rf'] = np.ravel(mean_rf)
    df['error_kdf'] = np.ravel(mean_kdf)
    df['kappa_rf'] = np.ravel(mean_kappa_rf)
    df['kappa_kdf'] = np.ravel(mean_kappa_kdf)
    df['ece_rf'] = np.ravel(mean_ece_rf)
    df['ece_kdf'] = np.ravel(mean_ece_kdf)
    df['fold'] = folds
    df['sample'] = samples

    df.to_csv('openML_cc18_task_'+str(task_id)+'.csv')

#%%
np.random.seed(12345)
cv = 5
reps = 10
n_estimators = 500
n_cores = 1
df = pd.DataFrame() 
benchmark_suite = openml.study.get_suite('OpenML-CC18')

#%%
total_cores = multiprocessing.cpu_count()
assigned_workers = total_cores//n_cores

Parallel(n_jobs=assigned_workers,verbose=1)(
        delayed(experiment)(
                task_id
                ) for task_id in benchmark_suite.tasks
            )

# %%
'''import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np

tasks = [3,6,12,14,16,18,22,28,32,43,219,3902,3903,3917,9952,9960,9978,10093,14965,14969,146820,167141]
sample_size = [10,100,500,1000]
reps = 5

sns.set_context('talk')

for task in tasks:
    fig, ax = plt.subplots(1,2, figsize=(16,8))
    
    error_kdf = np.zeros((4,5), dtype=float)
    error_rf = np.zeros((4,5), dtype=float)
    ece_kdf = np.zeros((4,5), dtype=float)
    ece_rf = np.zeros((4,5), dtype=float)

    df = pd.read_csv('openML_cc18_task_'+str(task)+'.csv')
    for ii in range(reps):
        error_kdf[:,ii] = df['error_kdf'][df['fold']==ii]
        error_rf[:,ii] = df['error_rf'][df['fold']==ii]

        ax[0].plot(sample_size, error_kdf[:,ii], c='r', alpha=0.5, lw=1)
        ax[0].plot(sample_size, error_rf[:,ii], c='k', alpha=0.5, lw=1)

    ax[0].plot(sample_size, np.mean(error_kdf,axis=1), label='KDF', c='r', lw=3)
    #ax.fill_between(sample_size, mean_kdf-1.96*var_kdf, mean_kdf+1.96*var_kdf, facecolor='r', alpha=0.5)
    ax[0].plot(sample_size, np.mean(error_rf,axis=1), label='RF', c='k', lw=3)
    #ax.fill_between(sample_size, mean_rf-1.96*var_kdf, mean_rf+1.96*var_kdf, facecolor='k', alpha=0.5)

    ax[0].set_xlabel('Sample size')
    ax[0].set_ylabel('Generalization Error')
    ax[0].set_xscale('log')
    ax[0].legend(frameon=False)
    ax[0].set_title('Generalization Error', fontsize=24)
    ax[0].set_yticks([0,.2,.4,.6,.8,1])
    right_side = ax[0].spines["right"]
    right_side.set_visible(False)
    top_side = ax[0].spines["top"]
    top_side.set_visible(False)

    for ii in range(reps):
        ece_kdf[:,ii] = df['ece_kdf'][df['fold']==ii]
        ece_rf[:,ii] = df['ece_rf'][df['fold']==ii]

        ax[1].plot(sample_size, ece_kdf[:,ii], c='r', alpha=0.5, lw=1)
        ax[1].plot(sample_size, ece_rf[:,ii], c='k', alpha=0.5, lw=1)

    ax[1].plot(sample_size, np.mean(ece_kdf,axis=1), label='KDF', c='r', lw=3)
    #ax.fill_between(sample_size, mean_kdf-1.96*var_kdf, mean_kdf+1.96*var_kdf, facecolor='r', alpha=0.5)
    ax[1].plot(sample_size, np.mean(ece_rf,axis=1), label='RF', c='k', lw=3)
    #ax.fill_between(sample_size, mean_rf-1.96*var_kdf, mean_rf+1.96*var_kdf, facecolor='k', alpha=0.5)

    ax[1].set_xlabel('Sample size')
    ax[1].set_ylabel('ECE')
    ax[1].set_xscale('log')
    ax[1].legend(frameon=False)
    ax[1].set_title('Expected Callibration Error',fontsize=24)
    ax[1].set_yticks([0,.2,.4,.6,.8,1])
    right_side = ax[1].spines["right"]
    right_side.set_visible(False)
    top_side = ax[1].spines["top"]
    top_side.set_visible(False)

    plt.savefig('plots/openML_cc18_'+str(task)+'.pdf')
#plt.show()'''

# %%
