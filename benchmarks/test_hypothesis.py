#%%
import numpy as np
import pandas as pd
import pickle
from kdg.utils import get_ece, trunk_sim, generate_gaussian_parity, plot_2dsim
from kdg import kdn, kdf, kde
from tensorflow import keras
from tensorflow.keras import activations
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
#%%
X, y = generate_gaussian_parity(10000, cluster_std=.3)
plot_2dsim(X,y)

#%%
compile_kwargs = {
        "loss": "binary_crossentropy",
        "optimizer": keras.optimizers.Adam(3e-4),
    }
callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10, verbose=True)
fit_kwargs = {
        "epochs": 200,
        "batch_size": 64,
        "verbose": False,
        "callbacks": [callback],
    }

#%%
def getNN(input_size, num_classes):
    network_base = keras.Sequential()
    initializer = keras.initializers.random_normal(seed=0)
    network_base.add(keras.layers.Dense(1000, kernel_initializer=initializer, input_shape=(input_size,)))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(1000, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(1000, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(1000, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(units=num_classes, activation="softmax", kernel_initializer=initializer))
    network_base.compile(**compile_kwargs)
    return network_base

#%%
def experiment(cov, reps):
    ece_dn = []
    ece_kdn = []
    for _ in range(reps):
        X, y = generate_gaussian_parity(1000, cluster_std=cov)
        X_test, y_test = generate_gaussian_parity(1000, cluster_std=cov)

        nn = getNN(input_size=2, num_classes=2)
        history = nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)
        model_kdn = kdn(network=nn)
        model_kdn.fit(X, y, k=10)
        model_kdn.global_bias = -1e20

        proba_kdn = model_kdn.predict_proba(X_test, distance='Geodesic')
        proba_dn = model_kdn.network.predict(X_test)
        
        ece_kdn.append(get_ece(proba_kdn, y_test))
        ece_dn.append(get_ece(proba_dn, y_test))
    
    
    return np.median(ece_kdn), np.median(ece_dn)
# %%
reps = 10
cov = [.1,.2,.3,.4,.5]

res = Parallel(n_jobs=-1,verbose=1)(
                delayed(experiment)(
                        cov_, reps
                        ) for cov_ in cov
                    )

kdn = []
dn = []

for ii in range(5):
    kdn.append(res[ii][0])
    dn.append(res[ii][1])
# %%
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(13,10))

ax.plot(cov, kdn, c='r', linewidth=3, label='KDN')
ax.plot(cov, dn, c='k', linewidth=3, label='DN')

ax.set_xlabel('cov', fontsize=35)
ax.set_ylabel('ECE', fontsize=35)

ax.tick_params(axis='both', which='major', labelsize=40)
ax.legend(fontsize=30, frameon=False)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
# %%
def generate_gaussian_parity(
    n_samples,
    sample_prob,
    centers=None,
    class_label=None,
    cluster_std=0.25,
    bounding_box=(-1.0, 1.0),
    angle_params=None,
    random_state=None,
):

    if random_state != None:
        np.random.seed(random_state)

    if centers == None:
        centers = np.array([(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)])

    if class_label == None:
        class_label = [0, 1, 1, 0]

    probability = np.array([sample_prob/2, (1-sample_prob)/2, (1-sample_prob)/2, sample_prob/2])
    # get the number of samples in each blob with equal probability
    samples_per_blob = np.random.multinomial(
        n_samples, probability
    )

    X = np.zeros((1,2), dtype=float)
    y = np.zeros((1), dtype=float)
    ii = 0
    for center, sample in zip(centers, samples_per_blob):
        X_, _ = make_blobs(
            n_samples=sample*10,
            n_features=2,
            centers=[center],
            cluster_std=cluster_std
        )
        col1 = (X_[:,0] > bounding_box[0]) & (X_[:,0] < bounding_box[1])
        col2 = (X_[:,1] > bounding_box[0]) & (X_[:,1] < bounding_box[1])
        X_ = X_[col1 & col2]
        X = np.concatenate((X,X_[:sample,:]), axis=0)
        y_ = np.array([class_label[ii]]*sample)
        y = np.concatenate((y, y_), axis=0)
        ii += 1

    X, y = X[1:], y[1:]


    return X, y.astype(int)
# %%
X, y = generate_gaussian_parity(10000, cluster_std=.3, sample_prob=.75)
plot_2dsim(X,y)
# %%
def experiment(p, reps):
    ece_dn = []
    ece_kdn = []
    for _ in range(reps):
        X, y = generate_gaussian_parity(100, sample_prob=p)
        X_test, y_test = generate_gaussian_parity(1000, sample_prob=p)

        nn = getNN(input_size=2, num_classes=2)
        history = nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)
        model_kdn = kdn(network=nn)
        model_kdn.fit(X, y, k=10)
        model_kdn.global_bias = -1e20

        proba_kdn = model_kdn.predict_proba(X_test, distance='Geodesic')
        proba_dn = model_kdn.network.predict(X_test)
        
        ece_kdn.append(get_ece(proba_kdn, y_test))
        ece_dn.append(get_ece(proba_dn, y_test))
    
    
    return np.median(ece_kdn), np.median(ece_dn)
# %%
reps = 10
ps = [.1,.2,.3,.4,.5]

res = Parallel(n_jobs=-1,verbose=1)(
                delayed(experiment)(
                        p, reps
                        ) for p in ps
                    )

kdn = []
dn = []

for ii in range(5):
    kdn.append(res[ii][0])
    dn.append(res[ii][1])
# %%
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(13,10))

ax.plot(cov, kdn, c='r', linewidth=3, label='KDN')
ax.plot(cov, dn, c='k', linewidth=3, label='DN')

ax.set_xlabel('P(class 0)', fontsize=35)
ax.set_ylabel('ECE', fontsize=35)

ax.tick_params(axis='both', which='major', labelsize=40)
ax.legend(fontsize=30, frameon=False)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

# %%
