#%%
import numpy as np
from kdg import kdf
from kdg.utils import sparse_parity
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
#%%
p = 20
p_star = 3
sample_size = np.logspace(
        np.log10(10),
        np.log10(5000),
        num=10,
        endpoint=True,
        dtype=int
        )
n_test = 1000
reps = 100
covarice_types = {'diag', 'full', 'spherical'}
criterion = 'aic'
n_estimators = 500
df = pd.DataFrame()
reps_list = []
accuracy_kdf = []
accuracy_rf = []
sample_list = []
# %%
for sample in sample_size:
    print('Doing sample %d'%sample)
    for ii in range(reps):
        X, y = sparse_parity(
            sample,
            p_star=p_star,
            p=p
        )
        X_test, y_test = sparse_parity(
            n_test,
            p_star=p_star,
            p=p
        )

        #train kdf
        model_kdf = kdf(
            covariance_types = covarice_types,
            criterion = criterion, 
            kwargs={'n_estimators':n_estimators}
        )
        model_kdf.fit(X, y)
        accuracy_kdf.append(
            np.mean(
                model_kdf.predict(X_test) == y_test
            )
        )

        #train rf
        model_rf = rf(n_estimators=n_estimators).fit(X, y)
        accuracy_rf.append(
            np.mean(
                model_rf.predict(X_test) == y_test
            )
        )
        reps_list.append(ii)
        sample_list.append(sample)

df['accuracy kdf'] = accuracy_kdf
df['accuracy rf'] = accuracy_rf
df['reps'] = reps_list
df['sample'] = sample_list

df.to_csv('high_dim_res_aic_kdf.csv')
# %%
