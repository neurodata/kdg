#%%
import numpy as np
from kdg import kdf
from kdg.utils import sparse_parity
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
reps = 10
n_estimators = range(1,501,10)
n_train = 10000
n_test = 1000
rf_med = []
rf_25_quantile = []
rf_75_quantile = []
rf_selected_med = []
rf_25_selected_quantile = []
rf_75_selected_quantile = []
# %%
for n_estimator in n_estimators:
    print('Doing for estimators ', n_estimator)
    err = []
    err_ = []
    for _ in range(reps):
        X, y = sparse_parity(n_train)
        X_test, y_test = sparse_parity(n_test)

        rf_model = rf(n_estimators=n_estimator).fit(X, y)
        err.append(
            1 - np.mean(
                rf_model.predict(X_test)==y_test
            )
        )
    
        rf_model = rf(n_estimators=1).fit(X[:,:3], y)
        err_.append(
            1 - np.mean(
                rf_model.predict(X_test[:,:3])==y_test
            )
        )

    rf_med.append(np.median(err))
    rf_25_quantile.append(
            np.quantile(err,[.25])[0]
        )
    rf_75_quantile.append(
            np.quantile(err,[.75])[0]
        )

    rf_selected_med.append(np.median(err_))
    rf_25_selected_quantile.append(
            np.quantile(err_,[.25])[0]
        )
    rf_75_selected_quantile.append(
            np.quantile(err_,[.75])[0]
        )
#%%
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(n_estimators, rf_med, c="k", label='RF')
ax.fill_between(n_estimators, rf_25_quantile, rf_75_quantile, facecolor='k', alpha=.3)

ax.plot(n_estimators, rf_selected_med, c="r", label='RF (feature selected)')
ax.fill_between(n_estimators, rf_25_selected_quantile, rf_75_selected_quantile, facecolor='r', alpha=.3)

ax.set_xticks(range(1,501,100))
ax.set_xlabel('Tree #')
ax.set_ylabel('Error')

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('plots/rf_increasing_tree.pdf')
# %%
