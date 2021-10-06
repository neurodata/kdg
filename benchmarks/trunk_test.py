#%%
import numpy as np
from kdg import kdf
from kdg.utils import trunk_sim
import pandas as pd
# %%
reps = 10
n_train = 100
n_test = 1000
dimensions = range(1,2000,100)
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
        X, y = trunk_sim(n_train, p_star=dim, p=dim)
        X_test, y_test = trunk_sim(n_test, p_star=dim, p=dim)
        model_kdf = kdf(kwargs={'n_estimators':500})
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


df.to_csv('sim_res/trunk_res2.csv')
# %%
