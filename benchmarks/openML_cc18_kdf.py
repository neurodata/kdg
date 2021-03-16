#%%
from kdg import kdf
from kdg.utils get_ece
import openml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier as rf
#%%
np.random.seed(12345)
cv = 5
reps = 10
n_estimators = 10
df = pd.DataFrame() 
benchmark_suite = openml.study.get_suite('OpenML-CC18')

#%%
def get_ece(predicted_posterior, y):
    hists = []
    hists_hat = []
    amts = []
    num_bins = 40
    eces_across_y_vals = []
    for y_val in np.unique(y):
        for i in range(num_bins):
            prop = i*1./num_bins
            inds = np.where((predicted_posterior[:, y_val] >= prop) & (predicted_posterior[:, y_val] <= prop+1./num_bins))[0]
            amts.append(len(inds))
            if len(inds) > 0:
                hists.append(len(np.where(y[inds] == y_val)[0])*1./len(inds))
                hists_hat.append(np.mean(predicted_posterior[inds, y_val]))
            else:
                hists.append(prop)
                hists_hat.append(prop + 0.5/num_bins)
        eces_across_y_vals.append(np.dot(np.abs(np.array(hists) - np.array(hists_hat)), amts) / np.sum(amts))
    return np.mean(eces_across_y_vals)
#%%
ids = []
fold = []
error_kdf = []
error_rf = []
sample_size = []

'''i=0
for task_id in benchmark_suite.tasks:
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()
    X = np.nan_to_num(X)

    print('Doing task %d sample size %d'%(task_id,X.shape[0]))

    skf = StratifiedKFold(n_splits=cv)
    
    ii = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model_rf = rf(n_estimators=n_estimators).fit(X_train, y_train)
        predicted_label = model_rf.predict(X_test)
        error_rf.append(
            1 - np.mean(y_test==predicted_label)
        )
        print('rf %f task %d'%(error_rf[-1],task_id))
        model_kdf = kdf({'n_estimators':n_estimators})
        model_kdf.fit(X_train, y_train)
        predicted_label = model_kdf.predict(X_test)
        error_kdf.append(
            1 - np.mean(y_test==predicted_label)
        )
        print('kdf %f task %d\n\n'%(error_kdf[-1],task_id))
        ids.append(task_id)
        fold.append(ii)
        sample_size.append(X.shape[0])
        ii +=1
    i += 1

    if i==5:
        break

df['task_id'] = ids
df['data_fold'] = fold
df['error_rf'] = error_rf
df['error_kdf'] = error_kdf
df['sample_size'] = sample_size

df.to_csv('openML_cc18.csv')


#%%
import matplotlib.pyplot as plt 
import seaborn as sns
df = pd.read_csv('openML_cc18.csv')
# %%
task_ids = np.unique(
    np.array(
        df['task_id']
    )
)

diff_err = {}
for task in task_ids:
    diff_err[str(task)] = np.array(df['error_rf'][df['task_id']==task]) - np.array(df['error_kdf'][df['task_id']==task])

diff_err = pd.DataFrame.from_dict(diff_err)
diff_err = pd.melt(diff_err,var_name='task', value_name='Error_Difference')
# %%
fig, ax = plt.subplots(1,1, figsize=(8,10))
ax.tick_params(labelsize=22)
ax = sns.stripplot(
    x="task", y="Error_Difference", data=diff_err,
    ax=ax
    )
ax.set_xlabel('Task ids', fontsize=20)
ax.set_ylabel('error_rf - error_kdf', fontsize=20)
plt.savefig('openMLcc18.pdf')'''
# %%
task_id = 14
task = openml.tasks.get_task(task_id)
X, y = task.get_X_and_y()
sample_size = [10,100,500,1000]
total_sample = X.shape[0]
mean_rf = np.zeros(len(sample_size), dtype=float)
mean_kdf = np.zeros(len(sample_size), dtype=float)
var_rf = np.zeros(len(sample_size), dtype=float)
var_kdf = np.zeros(len(sample_size), dtype=float)

error_rf = np.zeros((len(sample_size),reps), dtype=float)
error_kdf = np.zeros((len(sample_size),reps), dtype=float)
proba_rf = np.zeros((len(sample_size),reps), dtype=float)
proba_kdf = np.zeros((len(sample_size),reps), dtype=float)
for jj,sample in enumerate(sample_size):

    print('sample numer'+str(sample))
    for ii in range(reps):
        train_idx =  np.random.choice(range(total_sample), sample, replace=False)
        test_idx = np.random.choice(
            np.delete(range(total_sample), train_idx),
            1000,
            replace=False
        )

        model_rf = rf(n_estimators=n_estimators).fit(X[train_idx], y[train_idx])
        predicted_label = model_rf.predict(X[test_idx])
        proba_rf[jj][ii] = model_rf.predict_proba(X[test_idx])
        error_rf[jj][ii] = 1 - np.mean(y[test_idx]==predicted_label)

        model_kdf = kdf({'n_estimators':n_estimators})
        model_kdf.fit(X[train_idx], y[train_idx])
        predicted_label = model_kdf.predict(X[test_idx])
        proba_kdf[jj][ii] = model_kdf.predict_proba(X[test_idx])
        error_kdf[jj][ii] = 1 - np.mean(y[test_idx]==predicted_label)    

    mean_rf[jj] = np.mean(error_rf[jj])   
    var_rf[jj] = np.var(error_rf[jj], ddof=1)
    mean_kdf[jj] = np.mean(error_kdf[jj])   
    var_kdf[jj] = np.var(error_kdf[jj], ddof=1)
# %%
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

for ii in range(reps):
    ax.plot(sample_size, error_kdf[:,ii], c='r', ls='-.', lw=1)
    ax.plot(sample_size, error_rf[:,ii], c='k', ls='-.', lw=1)

ax.plot(sample_size, mean_kdf, label='KDF', c='r', lw=2)
ax.fill_between(sample_size, mean_kdf-1.96*var_kdf, mean_kdf+1.96*var_kdf, facecolor='r', alpha=0.5)
ax.plot(sample_size, mean_rf, label='RF', c='k', lw=2)
ax.fill_between(sample_size, mean_rf-1.96*var_kdf, mean_rf+1.96*var_kdf, facecolor='k', alpha=0.5)

ax.set_xlabel('Sample size')
ax.set_ylabel('Generalization Error')
ax.set_xscale('log')
ax.legend(frameon=False)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('openML_cc18_2.pdf')
plt.show()
# %%
