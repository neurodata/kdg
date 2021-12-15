#
# Created on Wed Dec 15 2021 10:48:38 AM
# Author: Ashwin De Silva (ldesilv2@jhu.edu)
# Objective: Label noise experiment for KDN on Gaussian Parity Dataset
#


# import external libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from tensorflow import keras

# import internal libraries
from kdg.kdn import *
from kdg.utils import generate_gaussian_parity

def label_noise_trial(n_samples, p=0.10):
    """Single label noise trial with proportion p of flipped labels."""
    X, y = generate_gaussian_parity(n_samples, cluster_std=0.25)
    X_test, y_test = generate_gaussian_parity(1000, cluster_std=0.25)
    X_val, y_val = generate_gaussian_parity(500, cluster_std=0.25)

    # NN params
    compile_kwargs = {
        "loss": "binary_crossentropy",
        "optimizer": keras.optimizers.Adam(1e-3),
    }
    callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=False)
    fit_kwargs = {
        "epochs": 300,
        "batch_size": 32,
        "verbose": False,
        "validation_data": (X_val, keras.utils.to_categorical(y_val)),
        "callbacks": [callback],
    }

    # network architecture
    def getNN():
        initializer = keras.initializers.GlorotNormal(seed=0)
        network_base = keras.Sequential()
        network_base.add(keras.layers.Dense(5, activation="relu", kernel_initializer=initializer, input_shape=(2,)))
        network_base.add(keras.layers.Dense(5, activation="relu", kernel_initializer=initializer))
        network_base.add(keras.layers.Dense(units=2, activation="softmax", kernel_initializer=initializer))
        network_base.compile(**compile_kwargs)
        return network_base

    # Generate noise and flip labels
    n_noise = np.int32(np.round(len(X) * p))
    noise_indices = random.sample(range(len(X)), n_noise)
    y[noise_indices] = 1 - y[noise_indices]

    # train Vanilla NN
    vanilla_nn = getNN()
    vanilla_nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)

    # train KDN
    model_kdn = kdn(
        network=vanilla_nn,
        k=1e-6,
        polytope_compute_method="all",
        weighting_method="lin",
        T=2,
        c=1,
        verbose=False,
    )
    model_kdn.fit(X, y)

    error_kdn = 1 - np.mean(model_kdn.predict(X_test) == y_test)
    error_nn = 1 - np.mean(np.argmax(vanilla_nn.predict(X_test), axis=1) == y_test)

    return error_kdn, error_nn

### Run the experiment with varying proportion of label noise
df = pd.DataFrame()
reps = 10
n_samples = 1000

err_kdn = []
err_nn = []
proportions =  [0.0, 0.1, 0.2, 0.3, 0.4]
proportion_list = []
reps_list = []

for p in proportions:
    print("Doing proportion {}".format(p))
    for ii in range(reps):
        err_kdn_i, err_nn_i = label_noise_trial(
            n_samples=n_samples, p=p
        )
        err_kdn.append(err_kdn_i)
        err_nn.append(err_nn_i)
        reps_list.append(ii)
        proportion_list.append(p)
        print("KDN error = {}, NN error = {}".format(err_kdn_i, err_nn_i))

# Construct DataFrame
df["reps"] = reps_list
df["proportion"] = proportion_list
df["error_kdn"] = err_kdn
df["error_nn"] = err_nn

err_kdn_med = []
err_kdn_25_quantile = []
err_kdn_75_quantile = []
err_nn_med = []
err_nn_25_quantile = []
err_nn_75_quantile = []

for p in proportions:
    curr_kdn = df["error_kdn"][df["proportion"] == p]
    curr_nn = df["error_nn"][df["proportion"] == p]

    err_kdn_med.append(np.median(curr_kdn))
    err_kdn_25_quantile.append(np.quantile(curr_kdn, [0.25])[0])
    err_kdn_75_quantile.append(np.quantile(curr_kdn, [0.75])[0])

    err_nn_med.append(np.median(curr_nn))
    err_nn_25_quantile.append(np.quantile(curr_nn, [0.25])[0])
    err_nn_75_quantile.append(np.quantile(curr_nn, [0.75])[0])

# Plotting
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(proportions, err_kdn_med, c="r", label="KDN")
ax.fill_between(
    proportions, err_kdn_25_quantile, err_kdn_75_quantile, facecolor="r", alpha=0.3
)
ax.plot(proportions, err_nn_med, c="k", label="NN")
ax.fill_between(
    proportions, err_nn_25_quantile, err_nn_75_quantile, facecolor="k", alpha=0.3
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
plt.savefig("plots/label_noise_gp_1000.pdf")
plt.show()
