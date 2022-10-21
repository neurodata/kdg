#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kdg import kdf
from kdg.utils import generate_gaussian_parity
# %%
reps = 10
test_sample = 1000
train_samples = [10,100,1000,10000]
scale = 10
noise_pow = .01
# %%
err_kdf_med = []
err_kdf_25 = []
err_kdf_75 =[]
err_rf_med = []
err_rf_25 = []
err_rf_75 =[]

for sample in train_samples:
    err = []
    err_ = []
    for _ in range(reps):
        X, y = generate_gaussian_parity(sample)
        X_ = X[:,0]*scale + np.random.rand(sample)*noise_pow
        X = np.concatenate((X,np.reshape(X_,(-1,1))), axis=1)

        X_test, y_test = generate_gaussian_parity(test_sample)
        X_ = X_test[:,0]*scale + np.random.rand(test_sample)*noise_pow
        X_test = np.concatenate((X_test,np.reshape(X_,(-1,1))), axis=1)

        model_kdf = model_kdf = kdf(
            kwargs={'n_estimators':500}
        )
        model_kdf.fit(X, y)
        err.append(
            1 - np.mean(model_kdf.predict(X_test) == y_test)
        )
        err_.append(
            1 - np.mean(model_kdf.rf_model.predict(X_test) == y_test)
        )

    err_kdf_med.append(
        np.median(err)
    )
    err_kdf_25.append(
        np.quantile(err,[.25])[0]
    )
    err_kdf_75.append(
        np.quantile(err,[.75])[0]
    )
    err_rf_med.append(
        np.median(err_)
    )
    err_rf_25.append(
        np.quantile(err_,[.25])[0]
    )
    err_rf_75.append(
        np.quantile(err_,[.75])[0]
    )

#%%
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(train_samples, err_kdf_med, c='r', linewidth=3, label='KDF')
ax.fill_between(train_samples, err_kdf_25, err_kdf_75, facecolor='r', alpha=.3)
ax.plot(train_samples, err_rf_med, c='b', linewidth=2, label='RF')
ax.fill_between(train_samples, err_rf_25, err_rf_75, facecolor='b', alpha=.3)
ax.set_xscale("log")
ax.legend()
plt.savefig('plots/correlated_features.pdf')
# %%
