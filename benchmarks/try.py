#%%
from numpy import dtype
from kdg import kdf, kdn, kdcnn
from kdg.utils import get_ece, plot_reliability, get_ace
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import activations
from tensorflow.keras import backend as bknd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import numpy as np
import openml
from sklearn.metrics import cohen_kappa_score
from kdg.utils import get_ece
from numpy import min_scalar_type
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.stats import multivariate_normal
import warnings
from sklearn.covariance import MinCovDet, fast_mcd, GraphicalLassoCV, LedoitWolf, EmpiricalCovariance, OAS, EllipticEnvelope, log_likelihood
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

#%%
compile_kwargs = {
        "loss": "binary_crossentropy",
        "optimizer": keras.optimizers.Adam(3e-4),
    }
callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10, verbose=True)
fit_kwargs = {
        "epochs": 200,
        "batch_size": 32,
        "verbose": True,
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

#%%
dataset_id = 37#44#11#22#11#40979#1067#1468#44#40979#1468#11#44#1050#
dataset = openml.datasets.get_dataset(dataset_id)
X, y, is_categorical, _ = dataset.get_data(
            dataset_format="array", target=dataset.default_target_attribute
        )

print(X.shape)
min_val = np.min(X,axis=0)
max_val = np.max(X, axis=0)
X = (X-min_val)/(max_val-min_val+1e-12)
_, y = np.unique(y, return_inverse=True)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200, train_size=500, random_state=10, stratify=y)

nn = getNN(input_size=X_train.shape[1], num_classes=np.max(y_train)+1, layer_size=1000)
history = nn.fit(X_train, keras.utils.to_categorical(y_train), **fit_kwargs)
#%%
model_kdn = kdn(network=nn)
model_kdn.fit(X_train, y_train, k=400)#int(np.ceil(.01*13000)))
model_kdn.global_bias=-1e100

#%%
proba = model_kdn.predict_proba(X_test,distance='Geodesic')
print(np.mean(y_test==np.argmax(proba,axis=1)))

get_ece(proba,y_test, n_bins=15)
#%%
proba_dn = model_kdn.network.predict(X_test)
print(np.mean(y_test==np.argmax(proba_dn,axis=1)))
get_ece(proba_dn,y_test, n_bins=15)
#%%
polytope_ids = model_kdn._get_polytope_ids(
                    np.array(X_train)
                )
w = 1- model_kdn._compute_geodesic(polytope_ids, polytope_ids)
#%%
k = 100#int(np.ceil(np.sqrt(X_train.shape[0])/2))#int(0.0006474*X_train.shape[0]*X.shape[1]**(1/1.323))
for id in range(100):
    #id = 15
    idx = np.argsort(w[id])[::-1]

    lbl = np.zeros(26,dtype=float)
    for ii in idx[:k]:
        lbl[int(y_train[ii])] += 1#w[id,ii]
        #print(w[id,ii], int(y_train[ii]))

    print(np.max(lbl)/np.sum(lbl))
print(X_train.shape)


# %%
