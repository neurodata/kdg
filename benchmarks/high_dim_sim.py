#%%
import numpy as np
from kdg import kdf
from kdg.utils import sparse_parity
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
#%%
p = 20
n_samples = 5000
n_test = 1000
reps = 100
covarice_types = {'diag', 'full', 'spherical'}
criterion = 'bic'
n_estimators = 500
df = pd.DataFrame()
p_star_list = []
reps_list = []
accuracy_kdf = []
accuracy_rf = []
# %%
for p_star in range(20,0,-1):
    for ii in range(reps):
        X, y = sparse_parity(
            n_samples,
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
        accuracy_kdf.extend(
            np.mean(
                model_kdf.predict(X_test) == y_test
            )
        )
        #train rf
        model_rf = rf(n_estimators=n_estimators).fit(X, y)
        accuracy_rf.extend(
            np.mean(
                model_rf.predict(X_test) == y_test
            )
        )

        reps_list.extend(ii)
        p_star_list.extend(p_star)

df['accuracy kdf'] = accuracy_kdf
df['accuracy rf'] = accuracy_rf
df['reps'] = reps_list
df['p_star'] = p_star_list

df.to_csv('high_dim_res_aic_kdf.csv')