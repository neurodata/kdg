#%%
# import modules
from matplotlib.pyplot import xlabel, ylabel
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from kdg.kdn import *
from kdg.utils import gaussian_sparse_parity, trunk_sim
import pandas as pd
import itertools
import seaborn as sns
sns.set_context('talk')
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
sample_size = 10000 # sample size under consideration
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
# network architecture (don't change the network)
def getNN():
    network_base = keras.Sequential()
    network_base.add(layers.Dense(25, activation='relu', input_shape=(20,)))
    network_base.add(layers.Dense(10, activation='relu'))
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
model_kdn = kdn(network=vanilla_nn,
                polytope_compute_method='all',
                weighting_method='FM',
                verbose=False)
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

X_0 = X_test[y_test==0] # data that belong to class 0

# pick the point closest to the class mean
idx = np.linalg.norm(X_0 - np.mean(X_0, axis=0), ord=2, axis=1).argsort()[0]
X_ref = X_0[idx]
x_ref = X_ref[:3]

# get the activation pattern of the reference point
X_ref_polytope_id = model_kdn._get_polytope_memberships(X_ref.reshape(1, len(X_ref)))[0][0]
a_ref = model_kdn._get_activation_pattern(X_ref_polytope_id)

rep = 1000
activation_similarity = []
for j in range(rep):
    # sample points which has the same signal but different noise
    X_bar = X_ref.copy()
    X_bar[:3] = x_ref
    X_bar[3:] = np.random.uniform(low=-1.0, high=1.0, size=(p-p_star,))
    X_bar_polytope_id = model_kdn._get_polytope_memberships(X_bar.reshape(1, len(X_bar)))[0][0]
    a_bar = model_kdn._get_activation_pattern(X_bar_polytope_id)

    match_status = a_ref == a_bar
    match_status = match_status.astype('int')

    # compute layer-wise activation similarity
    activation_similarity.append([
                            sum(match_status[:25])/25,
                            sum(match_status[25:35])/10,
                            sum(match_status[35:40])/5,
                            sum(match_status[40:])/2
                            ])

activation_similarity = np.array(activation_similarity)
# %%
# plot

activation_similarity_df = pd.DataFrame(activation_similarity, columns=['L1', 'L2', 'L3', 'L4'])
ax = sns.barplot(data=activation_similarity, ci="sd", capsize=.2)
ax.set(xlabel="Layer ID", ylabel="Activation Similarity")

# %%
