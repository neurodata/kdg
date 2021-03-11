from kdg import kdf
import openml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier as rf
#%%
cv = 5
n_estimators = 10
df = pd.DataFrame() 
benchmark_suite = openml.study.get_suite('OpenML-CC18')
ids = []
fold = []
error_kdf = []
error_rf = []

for task_id in benchmark_suite.tasks:
    print('Doing task %d'%task_id)
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()
    X = np.nan_to_num(X)

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
        print('rf %f'%error_rf[-1])
        model_kdf = kdf({'n_estimators':n_estimators})
        model_kdf.fit(X_train, y_train)
        predicted_label = model_kdf.predict(X_test)
        error_kdf.append(
            1 - np.mean(y_test==predicted_label)
        )
        print('kdf %f\n\n'%error_kdf[-1])
        ids.append(task_id)
        fold.append(ii)

df['task_id'] = ids
df['data_fold'] = fold
df['error_rf'] = error_rf
df['error_kdf'] = error_kdf

df.to_csv('openML_cc18.csv')


    