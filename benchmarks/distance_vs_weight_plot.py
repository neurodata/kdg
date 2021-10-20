#%%
# import modules
import numpy as np
from tensorflow import keras
from keras import layers
from kdg.kdn import *
from kdg.utils import gaussian_sparse_parity, trunk_sim
import pandas as pd
#%%
# define the experimental setup
p = 20 # total dimensions of the data vector
p_star = 3 # number of signal dimensions of the data vector
'''sample_size = np.logspace(
        np.log10(10),
        np.log10(5000),
        num=10,
        endpoint=True,
        dtype=int
        )'''
sample_size = 5000 # sample size under consideration
n_test = 1000 # test set size
reps = 10 # number of replicates

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
    "optimizer": keras.optimizers.Adam(3e-4)
    }
fit_kwargs = {
    "epochs": 150,
    "batch_size": 32,
    "verbose": False
    }

#%%
# network architecture
def getNN():
    network_base = keras.Sequential()
    network_base.add(layers.Dense(10, activation='relu', input_shape=(20,)))
    network_base.add(layers.Dense(5, activation='relu'))
    network_base.add(layers.Dense(5, activation='relu'))
    network_base.add(layers.Dense(units=2, activation = 'softmax'))
    network_base.compile(**compile_kwargs)
    return network_base

# %%
X, y = gaussian_sparse_parity(
    sample_size,
    p_star=p_star,
    p=p
)
X_test, y_test = gaussian_sparse_parity(
    n_test,
    p_star=p_star,
    p=p
)
#%%
# train Vanilla NN
vanilla_nn = getNN()
vanilla_nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)

# train KDN
model_kdn = kdn(network=vanilla_nn)
model_kdn.fit(X, y)

accuracy_kdn.append(
    np.mean(
        model_kdn.predict(X_test) == y_test
    )
)

accuracy_nn.append(
    np.mean(
        np.argmax(vanilla_nn.predict(X_test), axis=1) == y_test
    )
)

print("NN Accuracy:", accuracy_nn)
print("KDN Accuracy:", accuracy_kdn)


# %%

# pick a random point from testing data 
np.random.seed(0)
idx = np.random.randint(0, sample_size)
X_star = X[idx]

# sort the points in testing data according to the distance from X_star (consider only the signal dimension)
distance_vector_ = np.linalg.norm(X[:, :3] - X_star[:3], ord=2, axis=1)
X_ = X[distance_vector_.argsort()]
distance_vector_sorted = distance_vector_[distance_vector_.argsort()]

# get the polytope memberships
polytope_memberships = model_kdn._get_polytope_memberships(X_)[0]

# get the weights
current_polytope_activation = np.binary_repr(polytope_memberships[0], width=model_kdn.num_fc_neurons)[::-1] 
a_current = np.array(list(current_polytope_activation)).astype('int')
TM_weights = []
FM_weights = []
LL_weights = []
for member in polytope_memberships:
    member_activation = np.binary_repr(member, width=model_kdn.num_fc_neurons)[::-1]
    a_member = np.array(list(member_activation)).astype('int')
    
    match_status = a_member == a_current
    match_status = match_status.astype('int')

    # weight based on the total number of matches (uncomment)
    TM_weight = np.sum(match_status)/model_kdn.num_fc_neurons

    # weight based on the first mistmatch (uncomment)
    if len(np.where(match_status==0)[0]) == 0:
        FM_weight = 1.0
    else:
        first_mismatch_idx = np.where(match_status==0)[0][0]
        FM_weight = first_mismatch_idx / model_kdn.num_fc_neurons

    # layer-by-layer weights
    total_layers = len(model_kdn.network.layers)
    LL_weight = 0
    start = 0
    for layer_id in range(total_layers):
        num_neurons = model_kdn.network.layers[layer_id].output_shape[-1]
        end = start + num_neurons
        LL_weight += np.sum(match_status[start:end])/num_neurons
        start = end
    LL_weight /= total_layers

    TM_weights.append(TM_weight)
    FM_weights.append(FM_weight)
    LL_weights.append(LL_weight)
TM_weights_sorted = np.array(TM_weights)
FM_weights_sorted = np.array(FM_weights)
LL_weights_sorted = np.array(LL_weights)

# %%
# plot distance vs. weights
import seaborn as sns

sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.scatter(distance_vector_sorted, TM_weights, c="r", label='TM')
# ax.scatter(distance_vector_sorted, FM_weights, c="b", label='FM')
# ax.scatter(distance_vector_sorted, LL_weights, c="g", label='LL')

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xlabel('Distance')
ax.set_ylabel('Weight')
ax.legend(frameon=True)

# median plots

TM_unique_weights = np.sort(np.unique(TM_weights))
TM_distance_medians = []
for i in range(len(TM_unique_weights)):
    TM_distance_medians.append(np.median(distance_vector_sorted[TM_weights==TM_unique_weights[i]]))
ax.plot(np.array(TM_distance_medians), TM_unique_weights, c="k", label='TM')

# FM_unique_weights = np.sort(np.unique(FM_weights))
# FM_distance_medians = []
# for i in range(len(FM_unique_weights)):
#     FM_distance_medians.append(np.median(distance_vector_sorted[FM_weights==FM_unique_weights[i]]))
# ax.plot(np.array(FM_distance_medians), FM_unique_weights, c="k", label='TM')

# LL_unique_weights = np.sort(np.unique(LL_weights))
# LL_distance_medians = []
# for i in range(len(LL_unique_weights)):
#     LL_distance_medians.append(np.median(distance_vector_sorted[LL_weights==LL_unique_weights[i]]))
# ax.plot(np.array(LL_distance_medians), LL_unique_weights, c="k", label='TM')


   # %%
