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
import os

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
    
    max_class = len(np.unique(y))
    max_sample = np.floor(len(y)*(cv-1.1)/cv)
    sample_size = np.logspace(
        np.log10(max_class*2),
        np.log10(max_sample),
        num=10,
        endpoint=True,
        dtype=int
        )

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

    df.to_csv(folder+'/'+'openML_cc18_task_'+str(task_id)+'.csv')

#%%
np.random.seed(12345)
folder = 'singleton_removed'
os.mkdir(folder)
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
from scipy.interpolate import interp1d

tasks = [3,11,12,14,16,18,22,23,
        28,31,32,37,43,45,49,53,
        2074,3022,3549,3560,3902,
        3903,3913,3917,3918,9946,
        9952,9957,9960,9971,9978,
        10093,10101,125922,146817,
        146819,146820,146821,146822,
        146824,167140,167141]
reps = 5
samples = []
delta_kappas = []
delta_eces = []
kappa_over_dataset = [] 
ece_over_dataset = []

sns.set_context('talk')
fig, ax = plt.subplots(1,2, figsize=(16,8))
#minimum = 0
#maximum = 1e10

for task in tasks:
    df = pd.read_csv('openML_cc18_task_'+str(task)+'.csv')
    sample_ = list(np.unique(df['sample']))
    samples.extend(sample_)

samples = np.sort(np.unique(samples))

for task in tasks:
    df = pd.read_csv('openML_cc18_task_'+str(task)+'.csv')
    sample_ = list(np.unique(df['sample']))

    kappa_kdf = np.zeros((len(sample_),reps), dtype=float)
    kappa_rf = np.zeros((len(sample_),reps), dtype=float)
    delta_kappa = np.zeros((len(sample_),reps), dtype=float)
    delta_ece = np.zeros((len(sample_),reps), dtype=float)
    ece_kdf = np.zeros((len(sample_),reps), dtype=float)
    ece_rf = np.zeros((len(sample_),reps), dtype=float)

    for ii in range(reps):
        kappa_kdf[:,ii] = df['kappa_kdf'][df['fold']==ii]
        kappa_rf[:,ii] = df['kappa_rf'][df['fold']==ii]
        delta_kappa[:,ii] = kappa_kdf[:,ii] - kappa_rf[:,ii]

        #ax[0].plot(sample_size, delta_kappa[:,ii], c='k', alpha=0.5, lw=1)

    mean_delta_kappa = np.mean(delta_kappa,axis=1)
    interp_func_kappa = interp1d(sample_, mean_delta_kappa)
    interpolated_kappa = np.array([np.nan]*len(samples))
    interpolated_kappa_ = interp_func_kappa(samples[np.where((samples>=sample_[0]) & (samples<=sample_[-1]))[0]])
    interpolated_kappa[np.where((samples>=sample_[0]) & (samples<=sample_[-1]))[0]] = interpolated_kappa_
    kappa_over_dataset.append(interpolated_kappa)

    ax[0].plot(sample_, mean_delta_kappa, c='r', alpha=0.3)
    #ax.fill_between(sample_size, mean_kdf-1.96*var_kdf, mean_kdf+1.96*var_kdf, facecolor='r', alpha=0.5)
    #ax[0].plot(sample_size, np.mean(kappa_rf,axis=1), label='RF', c='k', lw=3)
    #ax.fill_between(sample_size, mean_rf-1.96*var_kdf, mean_rf+1.96*var_kdf, facecolor='k', alpha=0.5)

    ax[0].set_xlabel('Sample size')
    ax[0].set_ylabel('kappa_kdf - kappa_rf')
    ax[0].set_xscale('log')
    #ax[0].legend(frameon=False)
    ax[0].set_title('Delta Kappa', fontsize=24)
    #ax[0].set_yticks([0,.2,.4,.6,.8,1])
    right_side = ax[0].spines["right"]
    right_side.set_visible(False)
    top_side = ax[0].spines["top"]
    top_side.set_visible(False)

    for ii in range(reps):
        ece_kdf[:,ii] = df['ece_kdf'][df['fold']==ii]
        ece_rf[:,ii] = df['ece_rf'][df['fold']==ii]
        delta_ece[:,ii] = ece_kdf[:,ii] - ece_rf[:,ii]

        #ax[1].plot(sample_size, ece_kdf[:,ii], c='r', alpha=0.5, lw=1)
        #ax[1].plot(sample_size, delta_ece[:,ii], c='k', alpha=0.5, lw=1)

    mean_delta_ece = np.mean(delta_ece,axis=1)
    interp_func_ece = interp1d(sample_, mean_delta_ece)
    interpolated_ece = np.array([np.nan]*len(samples))
    interpolated_ece_ = interp_func_ece(samples[np.where((samples>=sample_[0]) & (samples<=sample_[-1]))[0]])
    interpolated_ece[np.where((samples>=sample_[0]) & (samples<=sample_[-1]))[0]] = interpolated_ece_
    ece_over_dataset.append(interpolated_ece)

    #interpolated_ece[np.where(samples<sample_[0] & samples>sample_[-1])] = np.nan

    ax[1].plot(sample_, mean_delta_ece, c='r', alpha=0.3)
    #ax.fill_between(sample_size, mean_kdf-1.96*var_kdf, mean_kdf+1.96*var_kdf, facecolor='r', alpha=0.5)
    #ax[1].plot(sample_size, np.mean(ece_rf,axis=1), label='RF', c='k', lw=3)
    #ax.fill_between(sample_size, mean_rf-1.96*var_kdf, mean_rf+1.96*var_kdf, facecolor='k', alpha=0.5)

    ax[1].set_xlabel('Sample size')
    ax[1].set_ylabel('ECE_kdf - ECE_rf')
    ax[1].set_xscale('log')
    #ax[1].legend(frameon=False)
    ax[1].set_title('Delta ECE',fontsize=24)
    #ax[1].set_yticks([0,.2,.4,.6,.8,1])
    right_side = ax[1].spines["right"]
    right_side.set_visible(False)
    top_side = ax[1].spines["top"]
    top_side.set_visible(False)

ax[0].hlines(0, 4, np.max(samples), colors="k", linestyles="dashed", linewidth=1.5)
ax[1].hlines(0, 4, np.max(samples), colors="k", linestyles="dashed", linewidth=1.5)
ax[0].plot(samples, np.nanmean(kappa_over_dataset, axis=0), c='r', lw=3)
ax[1].plot(samples, np.nanmean(ece_over_dataset, axis=0), c='r', lw=3)
plt.savefig('plots/openML_cc18_all.pdf')'''
#plt.show()

# %%
