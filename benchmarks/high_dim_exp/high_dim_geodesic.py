#%%
import numpy as np
import pandas as pd
import pickle
from kdg.utils import trunk_sim
from kdg import kdn, kdf
from tensorflow import keras
from tensorflow.keras import activations
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
# %%
##### main hyperparameters #####
mc_reps = 10
signal_dimension = [1, 10, 100, 200, 400, 600, 800, 1000]
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
def getNN(input_size, num_classes, layer_size):
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

    min_val = np.min(X,axis=0)
    max_val = np.max(X, axis=0)
    
    X = (X-min_val)/(max_val-min_val+1e-12)
    X_test = (X_test-min_val)/(max_val-min_val+1e-12)

    model_kdf = kdf(kwargs={'n_estimators':100})
    model_kdf.fit(X, y)

    nn = getNN(
            input_size=X.shape[1],
            num_classes=2, 
            layer_size=100
        )
    history = nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)
    model_kdn = kdn(network=nn)
    model_kdn.fit(X, y)
    
    kdf_err = 1 - \
        np.mean(model_kdf.predict(X_test)==y_test)
    kdf_err_geodesic = 1 - \
        np.mean(model_kdf.predict(X_test, distance='Geodesic')==y_test)
    kdn_err = 1 - \
        np.mean(model_kdn.predict(X_test)==y_test)
    kdn_err_geodesic = 1 - \
        np.mean(model_kdn.predict(X_test, distance='Geodesic')==y_test)
    rf_err = 1 - \
        np.mean(model_kdf.rf_model.predict(X_test)==y_test)
    dn_err = 1 - \
        np.mean(np.argmax(model_kdn.network.predict(X_test),axis=1)==y_test)
    
    return kdn_err, kdn_err_geodesic, kdf_err, kdf_err_geodesic, rf_err, dn_err

# %%
err_kdn = []
err_kdf = []
err_dn = []
err_rf = []
err_kdf_geodesic = []
err_kdn_geodesic = []
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
        err_kdn_geodesic.append(
            res[ii][1]
        )
        err_kdf.append(
            res[ii][2]
        )
        err_kdf_geodesic.append(
            res[ii][3]
        )
        err_rf.append(
            res[ii][4]
        )
        err_dn.append(
            res[ii][5]
        )
        dimension.append(
            dim
        )

df['err_kdn'] = err_kdn
df['err_kdf'] = err_kdf
df['err_dn'] = err_dn
df['err_rf'] = err_rf
df['err_kdf_geodesic'] = err_kdf_geodesic
df['err_kdn_geodesic'] = err_kdn_geodesic
df['dimension'] = dimension

with open('controlled_dimensionality_geodesic.pickle', 'wb') as f:
    pickle.dump(df, f)

    
# %%
with open('controlled_dimensionality_geodesic.pickle', 'rb') as f:
    df = pickle.load(f)
# %%
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(14,12))

err_rf_med = []
err_dn_med = []
err_kdf_med = []
err_kdn_med = []
err_kdf_geodesic_med = []
err_kdn_geodesic_med = []
err_rf_25 = []
err_rf_75 = []
err_dn_25 = []
err_dn_75 = []
err_kdf_25 = []
err_kdf_75 = []
err_kdn_25 = []
err_kdn_75 = []
err_kdn_geodesic_25 = []
err_kdn_geodesic_75 = []
err_kdf_geodesic_25 = []
err_kdf_geodesic_75 = []

for dim in signal_dimension:
    rf_err = 1-np.array(df['err_rf'][df['dimension']==dim])
    dn_err = 1-np.array(df['err_dn'][df['dimension']==dim])
    kdf_err = 1-np.array(df['err_kdf'][df['dimension']==dim])
    kdf_geodesic_err = 1-np.array(df['err_kdf_geodesic'][df['dimension']==dim])
    kdn_err = 1-np.array(df['err_kdn'][df['dimension']==dim])
    kdn_geodesic_err = 1-np.array(df['err_kdn_geodesic'][df['dimension']==dim])

    err_rf_med.append(
        np.median(rf_err)
    )
    err_dn_med.append(
        np.median(dn_err)
    )
    err_kdf_med.append(
        np.median(kdf_err)
    )
    err_kdn_med.append(
        np.median(kdn_err)
    )
    err_kdf_geodesic_med.append(
        np.median(kdf_geodesic_err)
    )
    err_kdn_geodesic_med.append(
        np.median(kdn_geodesic_err)
    )

    qunatiles = np.quantile(rf_err,[.25,.75],axis=0)
    err_rf_25.append(
        qunatiles[0]
    )
    err_rf_75.append(
        qunatiles[1]
    )

    qunatiles = np.quantile(dn_err,[.25,.75],axis=0)
    err_dn_25.append(
        qunatiles[0]
    )
    err_dn_75.append(
        qunatiles[1]
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

    qunatiles = np.quantile(kdf_geodesic_err,[.25,.75],axis=0)
    err_kdf_geodesic_25.append(
        qunatiles[0]
    )
    err_kdf_geodesic_75.append(
        qunatiles[1]
    )

    qunatiles = np.quantile(kdn_geodesic_err,[.25,.75],axis=0)
    err_kdn_geodesic_25.append(
        qunatiles[0]
    )
    err_kdn_geodesic_75.append(
        qunatiles[1]
    )
    
dimension = np.unique(df['dimension'])

ax.plot(dimension, err_rf_med, c='g', linewidth=3, label='RF')
ax.fill_between(dimension, err_rf_25, err_rf_75, facecolor='g', alpha=.3)
ax.plot(dimension, err_dn_med, c='#3f97b7', linewidth=3, label='DN')
ax.fill_between(dimension, err_dn_25, err_dn_75, facecolor='#3f97b7', alpha=.3)
ax.plot(dimension, err_kdf_med, c='k', linewidth=3, label='KGF-Euclidean')
ax.fill_between(dimension, err_kdf_25, err_kdf_75, facecolor='k', alpha=.3)
ax.plot(dimension, err_kdn_med, c='purple', linewidth=3, label='KGN-Euclidean')
ax.fill_between(dimension, err_kdn_25, err_kdn_75, facecolor='purple', alpha=.3)
ax.plot(dimension, err_kdf_geodesic_med, c='r', linewidth=3, label='KGF-Geodesic')
ax.fill_between(dimension, err_kdf_geodesic_25, err_kdf_geodesic_75, facecolor='r', alpha=.3)
ax.plot(dimension, err_kdn_geodesic_med, c='b', linewidth=3, label='KGN-Geodesic')
ax.fill_between(dimension, err_kdn_geodesic_25, err_kdn_geodesic_75, facecolor='b', alpha=.3)
#ax.set_xscale('log')

ax.set_xlabel('Dimension', fontsize=45)
ax.set_ylabel('Accuracy', fontsize=45)
ax.set_xticks([1, 200,600,1000])
ax.set_yticks([0.5, .8, 1])
ax.legend(
    fontsize=30,
    frameon=False,
    #bbox_to_anchor=(0.53,- 0.06),
    bbox_transform=plt.gcf().transFigure,
    loc="lower left",
)
ax.tick_params(axis='both', which='major', labelsize=40)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('trunk_geodesic.pdf')
# %%
