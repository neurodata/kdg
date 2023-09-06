#%%
from kdg import kdn
from kdg.utils import get_ece, sample_unifrom_circle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import numpy as np
import openml
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import activations
from tensorflow.keras import backend as bknd
from sklearn.metrics import cohen_kappa_score
from kdg.utils import get_ece
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
# %%
root_dir = "openml_kdn_res_ood"

try:
    os.mkdir(root_dir)
except:
    print("directory already exists!!!")

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
def experiment(dataset_id, layer_size = 1000, reps=10, random_state=42):
    filename = 'Dataset_' + str(dataset_id) + '.csv'
    if os.path.exists(os.path.join(root_dir, filename)):
        return
        
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )

    if np.mean(is_categorical) >0:
        return

    if np.isnan(np.sum(y)):
        return

    if np.isnan(np.sum(X)):
        return
    
    X /= np.max(
        np.linalg.norm(X, 2, axis=1)
    )
    _, y = np.unique(y, return_inverse=True)
    
        
    total_sample = X.shape[0]
    test_sample = 1000 if total_sample//3>1000 else total_sample//3
    train_sample = 1000 if total_sample-test_sample>1000 else total_sample-test_sample

    r = []    
    conf_dn = []
    conf_kdn = []
    conf_kdn_geod = []
    distances = np.arange(1, 5.5, .5)

    for rep in range(reps):
        X_train, X_test, y_train, y_test = train_test_split(
                     X, y, test_size=test_sample, train_size=train_sample, random_state=random_state+rep)
        nn = getNN(input_size=X_train.shape[1], num_classes=np.max(y_train)+1, layer_size=layer_size)
        history = nn.fit(X_train, keras.utils.to_categorical(y_train), **fit_kwargs)
        model_kdn = kdn(network=nn)
        model_kdn.fit(X_train, y_train, mul=10)
        model_kdn.global_bias = -100

        proba_kdn = model_kdn.predict_proba(X_test)
        proba_kdn_geod = model_kdn.predict_proba(X_test, distance='Geodesic')
        proba_dn = model_kdn.network.predict(X_test)

        conf_dn.append(
                np.nanmean(
                    np.max(proba_dn, axis=1)
                )
            )
        conf_kdn.append(
            np.nanmean(
                    np.max(proba_kdn, axis=1)
                )
        )
        conf_kdn_geod.append(
            np.nanmean(
                    np.max(proba_kdn_geod, axis=1)
                )
        )
        r.append(
            0
        )
        for distance in distances:
            X_ood = sample_unifrom_circle(1000, r=distance, p=X_train.shape[1])
            proba_kdn = model_kdn.predict_proba(X_ood)
            proba_kdn_geod = model_kdn.predict_proba(X_ood, distance='Geodesic')
            proba_dn = model_kdn.network.predict(X_ood)
            

            conf_dn.append(
                np.nanmean(
                    np.max(proba_dn, axis=1)
                )
            )
            conf_kdn.append(
                np.nanmean(
                        np.max(proba_kdn, axis=1)
                    )
            )
            conf_kdn_geod.append(
                np.nanmean(
                        np.max(proba_kdn_geod, axis=1)
                    )
            )
            r.append(
                distance
            )
            

    df = pd.DataFrame() 
    df['conf_kdn'] = conf_kdn
    df['conf_kdn_geod'] = conf_kdn_geod
    df['conf_dn'] = conf_dn
    df['distance'] = r

    df.to_csv(os.path.join(root_dir, filename))

# %%
benchmark_suite = openml.study.get_suite('OpenML-CC18')
id_done = [6,11,12,14,16,18,22,28,32,37,44,54,182,300,458, 554,1049,1050,1063,1067,1068, 1462, 1464, 1468, 1475, 1478, 1485, 1487, 1489, 1494, 1497, 1501, 1510, 4134, 4538, 40499, 40979, 40982, 40983, 40984, 40994, 40996, 23517, 40923, 41027]
'''Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment)(
                dataset_id,
                ) for dataset_id in openml.study.get_suite("OpenML-CC18").data
            )'''

for dataset_id in openml.study.get_suite("OpenML-CC18").data:
    '''if dataset_id in id_done:
        continue'''
    
    print("Doing ", dataset_id)
    experiment(dataset_id) 
