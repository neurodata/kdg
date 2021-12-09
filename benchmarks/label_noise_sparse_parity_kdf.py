import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from kdg.utils import gaussian_sparse_parity
from kdg import kdf
from sklearn.ensemble import RandomForestClassifier as rf


def sparse_parity_noise_trial(n_samples, noise_p=0.0, n_estimators=500, p_star=3, p=20):
    """Single label noise trial for sparse parity"""
    X, y = gaussian_sparse_parity(n_samples, p_star=p_star, p=p, cluster_std=0.5)
    X_test, y_test = gaussian_sparse_parity(1000, p_star=p_star, p=p, cluster_std=0.5)

    # Generate noise and flip labels
    n_noise = np.int32(np.round(len(X) * noise_p))
    noise_indices = random.sample(range(len(X)), n_noise)
    y[noise_indices] = 1 - y[noise_indices]

    model_kdf = kdf(kwargs={"n_estimators": n_estimators})
    model_kdf.fit(X, y)
    error_kdf = 1 - np.mean(model_kdf.predict(X_test) == y_test)

    model_rf = rf(n_estimators=n_estimators)
    model_rf.fit(X, y)
    error_rf = 1 - np.mean(model_rf.predict(X_test) == y_test)
    return error_kdf, error_rf


### Run experiment with varying proportion of label noise
df = pd.DataFrame()
reps = 10
n_estimators = 500
n_samples = 1000
noise_p = 0.30

err_kdf = []
err_rf = []
dimensions = np.arange(1, 100, 5)
dimension_list = []
reps_list = []

for dim in dimensions:
    print("Doing dimension {}".format(dim))
    for ii in range(reps):
        err_kdf_rep, err_rf_rep = sparse_parity_noise_trial(
            n_samples, noise_p=noise_p, n_estimators=n_estimators, p_star=dim, p=dim
        )
        err_kdf.append(err_kdf_rep)
        err_rf.append(err_rf_rep)
        dimension_list.append(dim)
        reps_list.append(ii)
        print("KDF error = {:.3f}, RF error = {:.3f}".format(err_kdf_rep, err_rf_rep))

# Construct DataFrame
df["reps"] = reps_list
df["dimension"] = dimension_list
df["error_kdf"] = err_kdf
df["error_rf"] = err_rf

err_kdf_med = []
err_kdf_25_quantile = []
err_kdf_75_quantile = []
err_rf_med = []
err_rf_25_quantile = []
err_rf_75_quantile = []

for dim in dimensions:
    curr_kdf = df["error_kdf"][df["dimension"] == dim]
    curr_rf = df["error_rf"][df["dimension"] == dim]

    err_kdf_med.append(np.median(curr_kdf))
    err_kdf_25_quantile.append(np.quantile(curr_kdf, [0.25])[0])
    err_kdf_75_quantile.append(np.quantile(curr_kdf, [0.75])[0])

    err_rf_med.append(np.median(curr_rf))
    err_rf_25_quantile.append(np.quantile(curr_rf, [0.25])[0])
    err_rf_75_quantile.append(np.quantile(curr_rf, [0.75])[0])

# Plotting
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(dimensions, err_kdf_med, c="r", label="KDF")
ax.fill_between(
    dimensions, err_kdf_25_quantile, err_kdf_75_quantile, facecolor="r", alpha=0.3
)
ax.plot(dimensions, err_rf_med, c="k", label="RF")
ax.fill_between(
    dimensions, err_rf_25_quantile, err_rf_75_quantile, facecolor="k", alpha=0.3
)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xlabel("Dimension")
ax.set_ylabel("Error")
plt.title("Gaussian Sparse Parity 30% Label Noise")
ax.legend(frameon=False)
plt.savefig("plots/label_noise_sparse_parity_kdf_30noise.pdf")
plt.show()
