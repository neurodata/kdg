#%%
# import modules
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from kdg.kdn import *
from kdg.utils import gaussian_sparse_parity, trunk_sim
import pandas as pd
import itertools
import seaborn as sns

sns.set_context("talk")
#%%
# define the experimental setup
p = 20  # total dimensions of the data vector
p_star = 3  # number of signal dimensions of the data vector
"""sample_size = np.logspace(
        np.log10(10),
        np.log10(5000),
        num=10,
        endpoint=True,
        dtype=int
        )"""
sample_size = 10000  # sample size under consideration
n_test = 1000  # test set size
reps = 10  # number of replicates

df = pd.DataFrame()
reps_list = []
accuracy_kdn = []
accuracy_kdn_ = []
accuracy_nn = []
accuracy_nn_ = []
sample_list = []

# NN params
compile_kwargs = {
    "loss": "binary_crossentropy",
    "optimizer": keras.optimizers.Adam(3e-4),
}
fit_kwargs = {"epochs": 150, "batch_size": 32, "verbose": False}

#%%
# network architecture (don't change the network)
def getNN():
    network_base = keras.Sequential()
    network_base.add(layers.Dense(5, activation="relu", input_shape=(20,)))
    network_base.add(layers.Dense(5, activation="relu"))
    network_base.add(layers.Dense(units=2, activation="softmax"))
    network_base.compile(**compile_kwargs)
    return network_base


# %%
X, y = gaussian_sparse_parity(sample_size, p_star=p_star, p=p)
X_test, y_test = gaussian_sparse_parity(n_test, p_star=p_star, p=p)
#%%
# train Vanilla NN
vanilla_nn = getNN()
vanilla_nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)

# train KDN
model_kdn = kdn(
    network=vanilla_nn,
    polytope_compute_method="all",
    weighting_method="FM",
    verbose=False,
)
model_kdn.fit(X, y)

accuracy_kdn.append(np.mean(model_kdn.predict(X_test) == y_test))

accuracy_nn.append(np.mean(np.argmax(vanilla_nn.predict(X_test), axis=1) == y_test))

print("NN Accuracy:", accuracy_nn)
print("KDN Accuracy:", accuracy_kdn)


# %%

X_0 = X[y == 0]  # data that belong to class 0

# pick the point closest to the class mean
idx = np.linalg.norm(X_0 - np.mean(X_0, axis=0), ord=2, axis=1).argsort()[0]
X_ref = X_0[idx]
x_ref = X_ref[:3]

# define the distances
d = np.arange(0.01, 0.5, 0.01)

# get the activation pattern of the reference point
X_ref_polytope_id = model_kdn._get_polytope_memberships(X_ref.reshape(1, len(X_ref)))[
    0
][0]
a_ref = np.binary_repr(X_ref_polytope_id, width=model_kdn.num_fc_neurons)[::-1]
a_ref = np.array(list(a_ref)).astype("int")

# create synthetic data
rep = 100
distance_replicates = {}
for i in range(len(d)):
    tmp_array = []
    for j in range(rep):
        # sample points which has the same distance from the reference point
        u = np.random.normal(0, 1, 3)
        u_norm = np.linalg.norm(u)
        x_bar = u / u_norm * d[i] + x_ref
        X_bar = X_ref
        X_bar[:3] = x_bar
        X_bar_polytope_id = model_kdn._get_polytope_memberships(
            X_bar.reshape(1, len(X_bar))
        )[0][0]
        a_bar = np.binary_repr(X_bar_polytope_id, width=model_kdn.num_fc_neurons)[::-1]
        a_bar = np.array(list(a_bar)).astype("int")
        tmp_array.append(a_bar)
    distance_replicates[d[i]] = tmp_array

# compute weights
mean_weights = []
quantile_25_weights = []
quantile_75_weights = []
for i in range(len(d)):
    weights_per_distance = []
    for j in range(rep):
        a_bar = distance_replicates[d[i]][j]

        # compute weights
        match_status = a_bar == a_ref
        match_status = match_status.astype("int")

        weights_per_distance.append(weight)

    weights_per_distance = np.array(weights_per_distance)
    mean_weights.append(np.mean(weights_per_distance))
    quantile_25_weights.append(np.quantile(weights_per_distance, [0.25])[0])
    quantile_75_weights.append(np.quantile(weights_per_distance, [0.75])[0])

# plot distance vs. weights
mean_weights = np.array(mean_weights)
quantile_25_weights = np.array(quantile_25_weights)
quantile_75_weights = np.array(quantile_75_weights)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

ax.plot(d, mean_weights, c="r", label="AP")
ax.fill_between(d, quantile_25_weights, quantile_75_weights, facecolor="r", alpha=0.3)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xlabel("Distance")
ax.set_ylabel("Weight")
ax.set_ylim(0, 1)
ax.legend(frameon=True)

# %%
