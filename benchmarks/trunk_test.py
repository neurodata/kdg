#%%
import numpy as np
from kdg import kdf
from kdg.utils import trunk_sim
import pandas as pd
# %%
reps = 100
n_train = 1000
n_test = 1000
dimensions = range(1,501,5)
#%%
err_kdf_med = []
err_kdf_25_quantile = []
err_kdf_75_quantile = []
err_rf_med = []
err_rf_25_quantile = []
err_rf_75_quantile = []
dims = []

for dim in dimensions:
    err_kdf = []
    err_rf = []

    print('Doing dimension ',dim)
    for _ in range(reps):
        X, y = trunk_sim(n_train, dim=dim)
        X_test, y_test = trunk_sim(n_test, dim=dim)
        model_kdf = kdf(covariance_types = {'diag', 'full', 'spherical'}, criterion='bic', kwargs={'n_estimators':500})
        model_kdf.fit(X, y)

        err_kdf.append(
           1 - np.mean(model_kdf.predict(X_test)==y_test)
        )
        err_rf.append(
           1 - np.mean(model_kdf.rf_model.predict(X_test)==y_test)
        )
    
    err_rf_med.append(np.median(err_rf))
    err_rf_25_quantile.append(
            np.quantile(err_rf,[.25])[0]
        )
    err_rf_75_quantile.append(
        np.quantile(err_rf,[.75])[0]
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

df.to_csv('trunk_res.csv')
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('trunk_res.csv')
err_kdf_med = df['err_kdf_med']
err_kdf_25_quantile = df['err_kdf_25_quantile']
err_kdf_75_quantile = df['err_kdf_75_quantile']
err_rf_med = df['err_rf_med']
err_rf_25_quantile = df['err_rf_25_quantile']
err_rf_75_quantile = df['err_rf_75_quantile']

sns.set_context('talk')
fig, ax = plt.subplots(1,1,figsize=(8,8))

ax.plot(range(1,501,5), err_rf_med, c="k", label='RF')
ax.fill_between(range(1,501,5), err_rf_25_quantile, err_rf_75_quantile, facecolor='k', alpha=.3)

ax.plot(range(1,501,5), err_kdf_med, c="r", label='KDF')
ax.fill_between(range(1,501,5), err_kdf_25_quantile, err_kdf_75_quantile, facecolor='r', alpha=.3)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xlabel('Dimension')
ax.set_ylabel('error')
ax.legend(frameon=False)

plt.savefig('plots/trunk.pdf')
# %%
