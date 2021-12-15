#
# Created on Thu Dec 09 2021 6:44:55 AM
# Author: Ashwin De Silva (ldesilv2@jhu.edu)
# Objective: Gaussian Sparse Parity Experiment
#

# import standard libraries
import numpy as np
from tensorflow import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# importing internal libraries
from kdg.utils import gaussian_sparse_parity
from kdg.kdn import *

# define the experimental setup
p = 20  # total dimensions of the data vector
p_star = 3  # number of signal dimensions of the data vector

sample_size = [500, 1000, 5000, 10000]  # sample size under consideration
n_test = 1000  # test set size
reps = 5  # number of replicates

df = pd.DataFrame()
reps_list = []
accuracy_kdn = []
accuracy_kdn_ = []
accuracy_nn = []
accuracy_nn_ = []
sample_list = []

X_val, y_val = gaussian_sparse_parity(1000)

# NN params
compile_kwargs = {
    "loss": "binary_crossentropy",
    "optimizer": keras.optimizers.Adam(1e-3),
}
callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=False)
fit_kwargs = {
    "epochs": 200,
    "batch_size": 32,
    "verbose": False,
    "validation_data": (X_val, keras.utils.to_categorical(y_val)),
    "callbacks": [callback],
}

# network architecture
def getNN():
    initializer = keras.initializers.GlorotNormal(seed=0)
    network_base = keras.Sequential()
    network_base.add(keras.layers.Dense(5, activation="relu", kernel_initializer=initializer, input_shape=(20,)))
    network_base.add(keras.layers.Dense(5, activation="relu", kernel_initializer=initializer))
    network_base.add(keras.layers.Dense(units=2, activation="softmax", kernel_initializer=initializer))
    network_base.compile(**compile_kwargs)
    return network_base


# run experiment
for sample in sample_size:
    print("Doing sample %d" % sample)
    for ii in range(reps):
        X, y = gaussian_sparse_parity(sample, p_star=p_star, p=p)
        X_test, y_test = gaussian_sparse_parity(n_test, p_star=p_star, p=p)

        # train Vanilla NN
        vanilla_nn = getNN()
        vanilla_nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)

        # train KDN
        model_kdn = kdn(
            network=vanilla_nn,
            polytope_compute_method="all",
            k=1e-6,
            weighting_method="lin",
            T=2,
            c=1,
            verbose=False,
        )
        model_kdn.fit(X, y)

        accuracy_kdn.append(np.mean(model_kdn.predict(X_test) == y_test))

        accuracy_nn.append(
            np.mean(np.argmax(vanilla_nn.predict(X_test), axis=1) == y_test)
        )
        reps_list.append(ii)
        sample_list.append(sample)

df["accuracy kdn"] = accuracy_kdn
df["accuracy nn"] = accuracy_nn
df["reps"] = reps_list
df["sample"] = sample_list

# save the results (CHANGE HERE)
df.to_csv("results/gsp.csv")

# plot

# Specify which results to plot (CHANGE HERE)
filename = "results/gsp.csv"

df = pd.read_csv(filename)

sample_size = [500, 1000, 5000, 10000]

err_nn_med = []
err_nn_25_quantile = []
err_nn_75_quantile = []

err_kdn_med = []
err_kdn_25_quantile = []
err_kdn_75_quantile = []

for sample in sample_size:
    err_nn = 1 - df["accuracy nn"][df["sample"] == sample]
    err_kdn = 1 - df["accuracy kdn"][df["sample"] == sample]

    err_nn_med.append(np.median(err_nn))
    err_nn_25_quantile.append(np.quantile(err_nn, [0.25])[0])
    err_nn_75_quantile.append(np.quantile(err_nn, [0.75])[0])

    err_kdn_med.append(np.median(err_kdn))
    err_kdn_25_quantile.append(np.quantile(err_kdn, [0.25])[0])
    err_kdn_75_quantile.append(np.quantile(err_kdn, [0.75])[0])

sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

ax.plot(sample_size, err_nn_med, c="k", label="NN")
ax.fill_between(
    sample_size, err_nn_25_quantile, err_nn_75_quantile, facecolor="k", alpha=0.3
)

ax.plot(sample_size, err_kdn_med, c="r", label="KDN")
ax.fill_between(
    sample_size, err_kdn_25_quantile, err_kdn_75_quantile, facecolor="r", alpha=0.3
)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xscale("log")
ax.set_xlabel("Sample size")
ax.set_ylabel("error")
ax.legend(frameon=False)

# Specify the figure save path (CHANGE HERE)
plt.savefig("plots/gsp.pdf")
