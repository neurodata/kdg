# %% timing experiment to determine required core
from kdg import kdf
import time
import numpy as np
import openml
from sklearn.model_selection import StratifiedKFold
#%%
n_workers = range(1,97)
benchmark_suite = openml.study.get_suite('OpenML-CC18')
task = openml.tasks.get_task(3)
X, y = task.get_X_and_y()

skf = StratifiedKFold(n_splits=5)
train_index, test_index = list(skf.split(X, y))[0]
spent_time = []

model_kdf = kdf({'n_estimators':500,'max_features':0.33})
model_kdf.fit(X[train_index], y[train_index])

for workers in n_workers:
    print('using %d workers'%workers)
    start = time.time()
    proba_kdf = model_kdf.predict_proba(X[test_index], total_worker=workers)
    end = time.time()
    print(end-start)
    spent_time.append(end-start)


# %%
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(range(1,len(spent_time)+1), spent_time, c='k', lw=3)
ax.set_xticks(range(1,len(spent_time)+1,2))
ax.set_xlabel('core #', fontsize=22)
ax.set_ylabel('time (sec)', fontsize=22)
ax.set_title('timing experiment')
#plt.show()
plt.savefig('timing_plot.pdf')
# %%
