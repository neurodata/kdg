#%%
from kdg import kdn
from kdg.utils import get_ece
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
root_dir = "openml_kdn_res"

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


def experiment(dataset_id, layer_size = 1000, reps=10, random_state=42):
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
    
    min_val = np.min(X,axis=0)
    max_val = np.max(X, axis=0)
    X = (X-min_val)/(max_val-min_val)
    _, y = np.unique(y, return_inverse=True)

    '''for ii in range(X.shape[1]):
        unique_val = np.unique(X[:,ii])
        if len(unique_val) < 10:
            return'''
        
    total_sample = X.shape[0]
    test_sample = total_sample//3
    train_samples = np.logspace(
            np.log10(10),
            np.log10(total_sample-test_sample),
            num=5,
            endpoint=True,
            dtype=int
        )
    err = []
    err_dn = []
    ece = []
    ece_dn = []
    mc_rep = []
    samples = []

    for train_sample in train_samples:
        for rep in range(reps):
            X_train, X_test, y_train, y_test = train_test_split(
                     X, y, test_size=test_sample, train_size=train_sample, random_state=random_state+rep)
            
            nn = getNN(input_size=X_train.shape[1], num_classes=np.max(y_train)+1, layer_size=layer_size)
            history = nn.fit(X_train, keras.utils.to_categorical(y_train), **fit_kwargs)
            model_kdn = kdn(network=nn)
            model_kdn.fit(X_train, y_train, mul=10)
            proba_kdn = model_kdn.predict_proba(X_test)
            proba_dn = model_kdn.network.predict(X_test)
            predicted_label_kdn = np.argmax(proba_kdn, axis = 1)
            predicted_label_dn = np.argmax(proba_dn, axis = 1)

            err.append(
                1 - np.mean(
                        predicted_label_kdn==y_test
                    )
            )
            err_dn.append(
                1 - np.mean(
                    predicted_label_dn==y_test
                )
            )
            ece.append(
                get_ece(proba_kdn, predicted_label_kdn, y_test)
            )
            ece_dn.append(
                get_ece(proba_dn, predicted_label_dn, y_test)
            )
            samples.append(
                train_sample
            )
            mc_rep.append(rep)

    df = pd.DataFrame() 
    df['err_kdn'] = err
    df['err_dn'] = err_dn
    df['ece_kdn'] = ece
    df['ece_dn'] = ece_dn
    df['rep'] = mc_rep
    df['samples'] = samples

    filename = 'Dataset_' + str(dataset_id) + '.csv'
    df.to_csv(os.path.join(root_dir, filename))

# %%
benchmark_suite = openml.study.get_suite('OpenML-CC18')

'''for dataset_id in openml.study.get_suite("OpenML-CC18").data:
    print("Doing data ", dataset_id)
    experiment(dataset_id) '''
Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment)(
                dataset_id,
                ) for dataset_id in openml.study.get_suite("OpenML-CC18").data
            )