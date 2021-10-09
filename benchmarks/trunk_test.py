#%%
import numpy as np
from kdg import kdf, random_tree_ensemble
from kdg.utils import trunk_sim
import pandas as pd
# %%
reps = 10
n_estimators = 30
n_train = 100
n_test = 1000
dimensions = range(1,652,10)
#%%
err_kdf_med = []
err_kdf_25_quantile = []
err_kdf_75_quantile = []
err_rf_med = []
err_rf_25_quantile = []
err_rf_75_quantile = []
err_random_med = []
err_random_25_quantile = []
err_random_75_quantile = []
dims = []

for dim in dimensions:
    err_kdf = []
    err_rf = []
    err_random = []

    print('Doing dimension ',dim)
    for _ in range(reps):
        X, y = trunk_sim(n_train, p_star=dim, p=dim)
        X_test, y_test = trunk_sim(n_test, p_star=dim, p=dim)
        model_kdf = kdf(kwargs={'n_estimators':n_estimators})
        model_kdf.fit(X, y)
        model_random_tree = random_tree_ensemble(n_estimators=n_estimators)
        model_random_tree.fit(X, y)

        err_kdf.append(
           1 - np.mean(model_kdf.predict(X_test)==y_test)
        )
        err_rf.append(
           1 - np.mean(model_kdf.rf_model.predict(X_test)==y_test)
        )
        err_random.append(
           1 - np.mean(model_random_tree.predict(X_test)==y_test)
        )

    err_rf_med.append(np.median(err_rf))
    err_rf_25_quantile.append(
            np.quantile(err_rf,[.25])[0]
        )
    err_rf_75_quantile.append(
        np.quantile(err_rf,[.75])[0]
    )
    err_random_med.append(np.median(err_random))
    err_random_25_quantile.append(
            np.quantile(err_random,[.25])[0]
        )
    err_random_75_quantile.append(
        np.quantile(err_random,[.75])[0]
    )
    err_kdf_med.append(np.median(err_kdf))
    err_kdf_25_quantile.append(
            np.quantile(err_kdf,[.25])[0]
        )
    err_kdf_75_quantile.append(
        np.quantile(err_kdf,[.75])[0]
    )
    dims.append(dim)

df = pd.DataFrame()
df['err_rf_med'] = err_rf_med
df['err_rf_25_quantile'] = err_rf_25_quantile
df['err_rf_75_quantile'] = err_rf_75_quantile
df['err_kdf_med'] = err_kdf_med 
df['err_kdf_25_quantile'] = err_kdf_25_quantile
df['err_kdf_75_quantile'] = err_kdf_75_quantile
df['err_random_med'] = err_random_med 
df['err_random_25_quantile'] = err_random_25_quantile
df['err_random_75_quantile'] = err_random_75_quantile


df.to_csv('sim_res/trunk_res_30tree.csv')
# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('sim_res/trunk_res_30tree.csv')
dimensions = range(1,652,10)

sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.fill_between(dimensions, df['err_kdf_25_quantile'], df['err_kdf_75_quantile'], facecolor='r', alpha=.3)
ax.plot(dimensions, df['err_kdf_med'], c='r', lw=3, label='KDF')

ax.fill_between(dimensions, df['err_rf_25_quantile'], df['err_rf_75_quantile'], facecolor='k', alpha=.3)
ax.plot(dimensions, df['err_rf_med'], c='k', lw=3, label='RF')

ax.fill_between(dimensions, df['err_random_25_quantile'], df['err_random_75_quantile'], facecolor='r', alpha=.3)
ax.plot(dimensions, df['err_random_med'], c='b', lw=3, label='Random')

ax.set_ylabel('Generilation Error')
ax.set_xlabel('Dimensions')

plt.legend(frameon=False)
plt.savefig('plots/trunk_res_30tree.pdf')
# %%
