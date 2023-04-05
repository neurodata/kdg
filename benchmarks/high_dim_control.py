#%%
import numpy as np
import pandas as pd
import pickle
from kdg.utils import trunk_sim
from kdg import kdn, kdf, kde
from tensorflow import keras
from tensorflow.keras import activations
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
# %%
##### main hyperparameters #####
mc_reps = 10
signal_dimension = [1,10,50,100,500,1000]
train_sample = 5000
test_sample = 1000
#%%
compile_kwargs = {
        "loss": "binary_crossentropy",
        "optimizer": keras.optimizers.Adam(3e-4),
    }
callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10, verbose=True)
fit_kwargs = {
        "epochs": 2000,
        "batch_size": 32,
        "verbose": False,
        "callbacks": [callback],
    }
# %%
def getNN(input_size, num_classes, layer_size=1000):
    network_base = keras.Sequential()
    initializer = keras.initializers.random_normal(seed=0)
    network_base.add(keras.layers.Dense(layer_size, kernel_initializer=initializer, input_shape=(input_size,)))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(layer_size, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(layer_size, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(layer_size, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(units=num_classes, activation="softmax", kernel_initializer=initializer))
    network_base.compile(**compile_kwargs)
    return network_base
# %%
def experiment(train_sample, test_sample, dim):
    X, y = trunk_sim(train_sample, p_star=dim)
    X_test, y_test = trunk_sim(test_sample, p_star=dim)

    model_kde = kde()
    model_kde.fit(X, y)

    model_kdf = kdf(kwargs={'n_estimators':500})
    model_kdf.fit(X, y)

    nn = getNN(
            input_size=X.shape[1],
            num_classes=2, 
            layer_size=100
        )
    history = nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)
    model_kdn = kdn(network=nn)
    model_kdn.fit(X, y)
    
    kde_err = 1 - \
        np.mean(model_kde.predict(X_test)==y_test)
    kdf_err = 1 - \
        np.mean(model_kdf.predict(X_test)==y_test)
    kdn_err = 1 - \
        np.mean(model_kdn.predict(X_test)==y_test)
    
    return kdn_err, kdf_err, kde_err

# %%
err_kdn = []
err_kdf = []
err_kde = []
dimension = []
df = pd.DataFrame()

for dim in signal_dimension:
    print('doing dim ', dim)
    res = Parallel(n_jobs=-1,verbose=1)(
                delayed(experiment)(
                        train_sample,
                        test_sample,
                        dim
                        ) for _ in range(mc_reps)
                    )
    
    for ii in range(mc_reps):
        err_kdn.append(
            res[ii][0]
        )
        err_kdf.append(
            res[ii][1]
        )
        err_kde.append(
            res[ii][2]
        )
        dimension.append(
            dim
        )

df['err_kdn'] = err_kdn
df['err_kdf'] = err_kdf
df['err_kde'] = err_kde
df['dimension'] = dimension

with open('controlled_dimensionality.pickle', 'wb') as f:
    pickle.dump(df, f)

    
# %%
with open('controlled_dimensionality.pickle', 'rb') as f:
    df = pickle.load(f)
# %%
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(13,10))

err_kdf_med = []
err_kdn_med = []
err_kde_med = []
err_kdf_25 = []
err_kdf_75 = []
err_kdn_25 = []
err_kdn_75 = []
err_kde_25 = []
err_kde_75 = []

for dim in signal_dimension:
    kdf_err = 1-np.array(df['err_kdf'][df['dimension']==dim])
    kde_err = 1-np.array(df['err_kde'][df['dimension']==dim])

    kdn_err = 1-np.array(df['err_kdn'][df['dimension']==dim])

    err_kdf_med.append(
        np.median(kdf_err)
    )
    err_kdn_med.append(
        np.median(kdn_err)
    )
    err_kde_med.append(
        np.median(kde_err)
    )

    qunatiles = np.quantile(kdf_err,[.25,.75],axis=0)
    err_kdf_25.append(
        qunatiles[0]
    )
    err_kdf_75.append(
        qunatiles[1]
    )

    qunatiles = np.quantile(kdn_err,[.25,.75],axis=0)
    err_kdn_25.append(
        qunatiles[0]
    )
    err_kdn_75.append(
        qunatiles[1]
    )

    qunatiles = np.quantile(kde_err,[.25,.75],axis=0)
    err_kde_25.append(
        qunatiles[0]
    )
    err_kde_75.append(
        qunatiles[1]
    )

    
dimension = np.unique(df['dimension'])
ax.plot(dimension, err_kdf_med, c='r', linewidth=3, label='KGF')
ax.fill_between(dimension, err_kdf_25, err_kdf_75, facecolor='r', alpha=.3)
ax.plot(dimension, err_kdn_med, c='b', linewidth=3, label='KGN')
ax.fill_between(dimension, err_kdn_25, err_kdn_75, facecolor='b', alpha=.3)
ax.plot(dimension, err_kde_med, c='k', linewidth=3, label='KDE')
ax.fill_between(dimension, err_kde_25, err_kde_75, facecolor='k', alpha=.3)
#ax.set_xscale('log')
ax.set_xlabel('Dimension', fontsize=35)
ax.set_ylabel('Accuracy', fontsize=35)
ax.set_yticks([0.7,0.8,0.9,1])
#ax.set_yticks([0, 1])
ax.tick_params(axis='both', which='major', labelsize=40)
ax.legend(fontsize=30, frameon=False)
ax.set_title("Trunk Simulation", fontsize=40)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig("high_dim_exp/trunk.pdf")
# %%
