#%%
import numpy as np
from kdg.utils import multiclass_guassian
from kdg import kdf
# %%
def inject_label_noise(y, per):
    total_samples = len(y)
    k = len(np.unique(y))
    samples_to_take = int(np.floor(total_samples*per))
    indices = np.random.choice(
                    list(range(total_samples)),
                    samples_to_take,
                    replace=False,
                )
    y[indices] = np.random.choice(
                    list(range(k)),
                    samples_to_take
                )

    return y
# %%
mc_reps = 100
n_estimators = 500
n = 141643
n_test = 10000
per = np.arange(0, .6, step = .1)
err_rf_med = []
err_rf_25_quantile = []
err_rf_75_quantile = []

err_kdf_med = []
err_kdf_25_quantile = []
err_kdf_75_quantile = []

for p in per:
    res_kdf = []
    res_rf = []
    for rep in range(mc_reps):
        X, y = multiclass_guassian(n)
        y = inject_label_noise(y, p)
        X_test, y_test = multiclass_guassian(n_test)

        model_kdf = kdf(kwargs={'n_estimators':n_estimators})
        model_kdf.fit(X, y)
        res_kdf.append(
            1 - np.mean(
            model_kdf.predict(X_test) == y_test
            )
        )
        res_rf.append(
            1 - np.mean(
            model_kdf.rf_model.predict(X_test) == y_test
            )
        )
    err_kdf_med.append(
        np.median(res_kdf)
    )
    err_kdf_25_quantile.append(
            np.quantile(res_kdf,[.25])[0]
        )
    err_kdf_75_quantile.append(
        np.quantile(res_kdf,[.75])[0]
    )

    err_rf_med.append(
        np.median(res_rf)
    )
    err_rf_25_quantile.append(
            np.quantile(res_rf,[.25])[0]
        )
    err_rf_75_quantile.append(
        np.quantile(res_rf,[.75])[0]
    )
# %% plot the result
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(per, err_rf_med, c="k", label='RF')
ax.fill_between(per, err_rf_25_quantile, err_rf_75_quantile, facecolor='k', alpha=.3)

ax.plot(per, err_kdf_med, c="r", label='KDF')
ax.fill_between(per, err_kdf_25_quantile, err_kdf_75_quantile, facecolor='r', alpha=.3)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xscale('log')
ax.set_xlabel('Sample size')
ax.set_ylabel('error')
ax.legend(frameon=False)

plt.savefig('plots/label_noise_cep.pdf')

# %%
