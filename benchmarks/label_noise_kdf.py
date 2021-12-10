import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from kdg.utils import generate_gaussian_parity
from kdg import kdf
from sklearn.ensemble import RandomForestClassifier as rf


def label_noise_trial(n_samples, p=0.10, n_estimators=500):
    """Single label noise trial with proportion p of flipped labels."""
    X, y = generate_gaussian_parity(n_samples, cluster_std=0.5)
    X_test, y_test = generate_gaussian_parity(1000, cluster_std=0.5)

    # Generate noise and flip labels
    n_noise = np.int32(np.round(len(X) * p))
    noise_indices = random.sample(range(len(X)), n_noise)
    y[noise_indices] = 1 - y[noise_indices]

    model_kdf = kdf(kwargs={"n_estimators": n_estimators})
    model_kdf.fit(X, y)
    error_kdf = 1 - np.mean(model_kdf.predict(X_test) == y_test)

    model_rf = rf(n_estimators=n_estimators)
    model_rf.fit(X, y)
    error_rf = 1 - np.mean(model_rf.predict(X_test) == y_test)
    return error_kdf, error_rf


### Run the experiment with varying proportion of label noise
df = pd.DataFrame()
reps = 10
n_estimators = 500
n_samples = 5000

err_kdf = []
err_rf = []
proportions = [0.0, 0.1, 0.2, 0.3, 0.4]
proportion_list = []
reps_list = []

for p in proportions:
    print("Doing proportion {}".format(p))
    for ii in range(reps):
        err_kdf_i, err_rf_i = label_noise_trial(
            n_samples=n_samples, p=p, n_estimators=n_estimators
        )
        err_kdf.append(err_kdf_i)
        err_rf.append(err_rf_i)
        reps_list.append(ii)
        proportion_list.append(p)
        print("KDF error = {}, RF error = {}".format(err_kdf_i, err_rf_i))

# Construct DataFrame
df["reps"] = reps_list
df["proportion"] = proportion_list
df["error_kdf"] = err_kdf
df["error_rf"] = err_rf

err_kdf_med = []
err_kdf_25_quantile = []
err_kdf_75_quantile = []
err_rf_med = []
err_rf_25_quantile = []
err_rf_75_quantile = []

for p in proportions:
    curr_kdf = df["error_kdf"][df["proportion"] == p]
    curr_rf = df["error_rf"][df["proportion"] == p]

    err_kdf_med.append(np.median(curr_kdf))
    err_kdf_25_quantile.append(np.quantile(curr_kdf, [0.25])[0])
    err_kdf_75_quantile.append(np.quantile(curr_kdf, [0.75])[0])

    err_rf_med.append(np.median(curr_rf))
    err_rf_25_quantile.append(np.quantile(curr_rf, [0.25])[0])
    err_rf_75_quantile.append(np.quantile(curr_rf, [0.75])[0])

# Plotting
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(proportions, err_kdf_med, c="r", label="KDF")
ax.fill_between(
    proportions, err_kdf_25_quantile, err_kdf_75_quantile, facecolor="r", alpha=0.3
)
ax.plot(proportions, err_rf_med, c="k", label="RF")
ax.fill_between(
    proportions, err_rf_25_quantile, err_rf_75_quantile, facecolor="k", alpha=0.3
)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xlabel("Label Noise Proportion")
ax.set_ylabel("Error")
plt.title("Gaussian XOR Label Noise")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("plots/label_noise_kdf_5000.pdf")
plt.show()
