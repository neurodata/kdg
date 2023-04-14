#%%
import numpy as np
import pandas as pd
import pickle
from kdg.utils import trunk_sim
from kdg import kdn, kdf, kde, extra_kdf
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

    model_kde = kde()
    model_kde.fit(X, y)

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
    
    model_extra_kdf = extra_kdf(kwargs={'n_estimators':100, 'max_features':1})
    model_extra_kdf.fit(X, y)

    kde_err = 1 - \
        np.mean(model_kde.predict(X_test)==y_test)
    kdf_err = 1 - \
        np.mean(model_kdf.predict(X_test)==y_test)
    kdn_err = 1 - \
        np.mean(model_kdn.predict(X_test)==y_test)
    rf_err = 1 - \
        np.mean(model_kdf.rf_model.predict(X_test)==y_test)
    dn_err = 1 - \
        np.mean(np.argmax(model_kdn.network.predict(X_test),axis=1)==y_test)
    extra_kdf_err = 1 - \
        np.mean(model_extra_kdf.predict(X_test)==y_test)
    
    return kdn_err, kdf_err, kde_err, extra_kdf_err, rf_err, dn_err



def experiment_noise(train_sample, test_sample, dim):
    X, y = trunk_sim(train_sample, p_star=1, p=dim)
    X_test, y_test = trunk_sim(test_sample, p_star=1, p=dim)

    min_val = np.min(X,axis=0)
    max_val = np.max(X, axis=0)
    
    X = (X-min_val)/(max_val-min_val+1e-12)
    X_test = (X_test-min_val)/(max_val-min_val+1e-12)
    
    model_kde = kde()
    model_kde.fit(X, y)

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
    
    model_extra_kdf = extra_kdf(kwargs={'n_estimators':100, 'max_features':1})
    model_extra_kdf.fit(X, y)

    kde_err = 1 - \
        np.mean(model_kde.predict(X_test)==y_test)
    kdf_err = 1 - \
        np.mean(model_kdf.predict(X_test)==y_test)
    kdn_err = 1 - \
        np.mean(model_kdn.predict(X_test)==y_test)
    rf_err = 1 - \
        np.mean(model_kdf.rf_model.predict(X_test)==y_test)
    dn_err = 1 - \
        np.mean(np.argmax(model_kdn.network.predict(X_test),axis=1)==y_test)
    extra_kdf_err = 1 - \
        np.mean(model_extra_kdf.predict(X_test)==y_test)
    
    return kdn_err, kdf_err, kde_err, extra_kdf_err, rf_err, dn_err

# %%
err_kdn = []
err_kdf = []
err_dn = []
err_rf = []
err_kde = []
err_extra_kdf = []
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
        err_extra_kdf.append(
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
df['err_kde'] = err_kde
df['err_extra_kdf'] = err_extra_kdf
df['dimension'] = dimension

with open('controlled_dimensionality.pickle', 'wb') as f:
    pickle.dump(df, f)

    
# %%
with open('controlled_dimensionality.pickle', 'rb') as f:
    df = pickle.load(f)
# %%
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(14,12))

err_kdf_med = []
err_kdn_med = []
err_kde_med = []
err_extra_kdf_med = []
err_kdf_25 = []
err_kdf_75 = []
err_kdn_25 = []
err_kdn_75 = []
err_kde_25 = []
err_kde_75 = []
err_extra_kdf_25 = []
err_extra_kdf_75 = []

for dim in signal_dimension:
    kdf_err = 1-np.array(df['err_kdf'][df['dimension']==dim])
    kde_err = 1-np.array(df['err_kde'][df['dimension']==dim])
    kdn_err = 1-np.array(df['err_kdn'][df['dimension']==dim])
    extra_kdf_err = 1-np.array(df['err_extra_kdf'][df['dimension']==dim])

    err_kdf_med.append(
        np.median(kdf_err)
    )
    err_kdn_med.append(
        np.median(kdn_err)
    )
    err_kde_med.append(
        np.median(kde_err)
    )
    err_extra_kdf_med.append(
        np.median(extra_kdf_err)
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

    qunatiles = np.quantile(extra_kdf_err,[.25,.75],axis=0)
    err_extra_kdf_25.append(
        qunatiles[0]
    )
    err_extra_kdf_75.append(
        qunatiles[1]
    )
    
dimension = np.unique(df['dimension'])
ax.plot(dimension, err_kdf_med, c='r', linewidth=3, label='KGF')
ax.fill_between(dimension, err_kdf_25, err_kdf_75, facecolor='r', alpha=.3)
ax.plot(dimension, err_kdn_med, c='b', linewidth=3, label='KGN')
ax.fill_between(dimension, err_kdn_25, err_kdn_75, facecolor='b', alpha=.3)
ax.plot(dimension, err_kde_med, c='k', linewidth=3, label='KDE')
ax.fill_between(dimension, err_kde_25, err_kde_75, facecolor='k', alpha=.3)
ax.plot(dimension, err_extra_kdf_med, c='purple', linewidth=3, label='Extra KGF')
ax.fill_between(dimension, err_extra_kdf_25, err_extra_kdf_75, facecolor='purple', alpha=.3)
#ax.set_xscale('log')

ax.set_xlabel('Dimension', fontsize=45)
ax.set_ylabel('Accuracy', fontsize=45)
ax.set_xticks([1, 200,600,1000])
ax.set_yticks([0.7, .8, .9, 1])
ax.legend(
    fontsize=30,
    frameon=False,
    #bbox_to_anchor=(0.53,- 0.06),
    bbox_transform=plt.gcf().transFigure,
    loc="upper right",
)
ax.tick_params(axis='both', which='major', labelsize=40)

plt.savefig('trunk.pdf')
# %%
