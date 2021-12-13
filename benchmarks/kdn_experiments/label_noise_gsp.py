# %%

# import external libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from tensorflow import keras

# import internal libraries
from kdg.utils import gaussian_sparse_parity
from kdg import kdn

def sparse_parity_noise_trial(n_samples, noise_p=0.0, p_star=3, p=20):
    """Single label noise trial for sparse parity"""
    X, y = gaussian_sparse_parity(n_samples, p_star=p_star, p=p, cluster_std=0.5)
    X_test, y_test = gaussian_sparse_parity(1000, p_star=p_star, p=p, cluster_std=0.5)
    X_val, y_val = gaussian_sparse_parity(500, p_star=p_star, p=p, cluster_std=0.5)

    # NN params
    compile_kwargs = {
        "loss": "binary_crossentropy",
        "optimizer": keras.optimizers.Adam(3e-4),
    }
    callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=False)
    fit_kwargs = {
        "epochs": 200,
        "batch_size": 64,
        "verbose": False,
        "validation_data": (X_val, keras.utils.to_categorical(y_val)),
        "callbacks": [callback],
    }

    # network architecture
    def getNN():
        network_base = keras.Sequential()
        network_base.add(keras.layers.Dense(5, activation="relu", input_shape=(X.shape[-1],)))
        network_base.add(keras.layers.Dense(5, activation="relu"))
        network_base.add(keras.layers.Dense(units=2, activation="softmax"))
        network_base.compile(**compile_kwargs)
        return network_base

    # Generate noise and flip labels
    n_noise = np.int32(np.round(len(X) * noise_p))
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

# %%

### Run experiment with varying proportion of label noise
df = pd.DataFrame()
reps = 10
n_samples = 10000
noise_p = 0.30

err_kdn = []
err_nn = []
dimensions = np.arange(1, 100, 5)
dimension_list = []
reps_list = []

for dim in dimensions:
    print("Doing dimension {}".format(dim))
    for ii in range(reps):
        err_kdn_rep, err_nn_rep = sparse_parity_noise_trial(
            n_samples, noise_p=noise_p, p_star=dim, p=dim
        )
        err_kdn.append(err_kdn_rep)
        err_nn.append(err_nn_rep)
        dimension_list.append(dim)
        reps_list.append(ii)
        print("KDN error = {:.3f}, NN error = {:.3f}".format(err_kdn_rep, err_nn_rep))

# Construct DataFrame
df["reps"] = reps_list
df["dimension"] = dimension_list
df["error_kdn"] = err_kdn
df["error_nn"] = err_nn

err_kdn_med = []
err_kdn_25_quantile = []
err_kdn_75_quantile = []
err_nn_med = []
err_nn_25_quantile = []
err_nn_75_quantile = []

for dim in dimensions:
    curr_kdn = df["error_kdn"][df["dimension"] == dim]
    curr_nn = df["error_nn"][df["dimension"] == dim]

    err_kdn_med.append(np.median(curr_kdn))
    err_kdn_25_quantile.append(np.quantile(curr_kdn, [0.25])[0])
    err_kdn_75_quantile.append(np.quantile(curr_kdn, [0.75])[0])

    err_nn_med.append(np.median(curr_nn))
    err_nn_25_quantile.append(np.quantile(curr_nn, [0.25])[0])
    err_nn_75_quantile.append(np.quantile(curr_nn, [0.75])[0])

# Plotting
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(dimensions, err_kdn_med, c="r", label="KDN")
ax.fill_between(
    dimensions, err_kdn_25_quantile, err_kdn_75_quantile, facecolor="r", alpha=0.3
)
ax.plot(dimensions, err_nn_med, c="k", label="NN")
ax.fill_between(
    dimensions, err_nn_25_quantile, err_nn_75_quantile, facecolor="k", alpha=0.3
)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xlabel("Dimension")
ax.set_ylabel("Error")
plt.title("Gaussian Sparse Parity 30% Label Noise")
ax.legend(frameon=False)
plt.savefig("plots/label_noise_sparse_parity_30noise.pdf")
plt.show()

# %%
