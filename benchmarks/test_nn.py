#%%
from kdg import kdn
from kdg.utils import get_ece
import openml
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import cohen_kappa_score
from kdg.utils import get_ece
from tensorflow import keras
from keras import layers
import os
from os import listdir, getcwd 
#%%
# NN params
compile_kwargs = {
    "loss": "binary_crossentropy",
    "optimizer": keras.optimizers.Adam(1e-3),
}
callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10, verbose=True)
fit_kwargs = {
    "epochs": 200,
    "batch_size": 32,
    "verbose": True,
    "callbacks": [callback],
}

# network architecture
def getNN(compile_kwargs, feature_size, total_class):
    network_base = keras.Sequential()
    network_base.add(layers.Dense(50, activation="relu", input_shape=(feature_size,)))
    network_base.add(layers.Dense(50, activation="relu"))
    network_base.add(layers.Dense(50, activation="relu"))
    network_base.add(layers.Dense(50, activation="relu"))
    network_base.add(layers.Dense(units=total_class, activation="softmax"))
    network_base.compile(**compile_kwargs)
    return network_base

def count_nn_param(model): # Compute number of params in a model (the actual number of floats)
    return sum([np.prod(keras.backend.get_value(w).shape) for w in model.trainable_weights])

def count_kdn_param(kdn_model):
    total_param = 0

    for label in kdn_model.labels:
        total_param += len(kdn_model.polytope_cardinality[label])
        total_param += len(kdn_model.polytope_cardinality[label])\
            *(kdn_model.feature_dim + 1)

    return total_param

#%%
dataset_id = 14#44#1497#1067#1468#44#40979#1468#11#44#1050#
dataset = openml.datasets.get_dataset(dataset_id)
X, y, is_categorical, _ = dataset.get_data(
            dataset_format="array", target=dataset.default_target_attribute
        )
#%%
unique_classes, counts = np.unique(y, return_counts=True)

test_sample = 100
total_sample = X.shape[0]
indx = list(
    range(
        total_sample
        )
)

train_samples = np.logspace(
np.log10(2),
np.log10(total_sample-test_sample),
num=10,
endpoint=True,
dtype=int
)

train_sample = train_samples[-1]
np.random.shuffle(indx)
indx_to_take_train = indx[:train_sample]
indx_to_take_test = indx[-test_sample:]       
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
vanilla_nn = getNN(compile_kwargs, X.shape[-1], len(np.unique(y)))
vanilla_nn.fit(
X[indx_to_take_train], 
keras.utils.to_categorical(y[indx_to_take_train], num_classes=len(np.unique(y))), 
**fit_kwargs
)

model_kdn = kdn(
    network=vanilla_nn,
    k=1e300,
    verbose=False,
)
model_kdn.fit(X[indx_to_take_train], y[indx_to_take_train])

#%%
print(np.mean(model_kdn.predict(X[indx_to_take_test])==y[indx_to_take_test]))
print(np.mean(np.argmax(vanilla_nn.predict(X[indx_to_take_test]), axis=1)==y[indx_to_take_test]))
print(np.mean(model_kdn.predict(X[indx_to_take_train])==y[indx_to_take_train]))

# %%
