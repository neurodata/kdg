#%%
from kdg import kdn
from kdg.utils import get_ece, get_ace
import seaborn as sns
import pandas as pd
import os
import numpy as np
import openml
from tensorflow import keras
from sklearn.calibration import CalibratedClassifierCV as calcv
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import activations
from tensorflow.keras import backend as bknd
from sklearn.metrics import cohen_kappa_score
from kdg.utils import get_ece
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

from numpy.random import seed
from tqdm import tqdm
import pandas as pd
import timeit
import tensorflow as tf
#%%
compile_kwargs = {
    "loss": "binary_crossentropy",
    "optimizer": tf.keras.optimizers.legacy.Adam(3e-4),
}
callback = keras.callbacks.EarlyStopping(
    monitor="loss", patience=10, verbose=True)
fit_kwargs = {
    "epochs": 50,
    "batch_size": 32,
    "verbose": False,
    "callbacks": [callback],
}

#%%
def getNN(input_size, num_classes, layer_size=1000):
    network_base = keras.Sequential()
    initializer = keras.initializers.random_normal(seed=0)
    network_base.add(keras.layers.Dense(
        layer_size, kernel_initializer=initializer, input_shape=(input_size,)))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(
        layer_size, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(
        layer_size, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(units=num_classes,
                     activation="softmax", kernel_initializer=initializer))
    network_base.compile(**compile_kwargs)
    return network_base

#%%
nodes = [10,100,1000,5000,10000]
dataset_id = 554
reps = 10

#%%
dataset = openml.datasets.get_dataset(dataset_id)
X, y, is_categorical, _ = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)
polytope = []
training_time = []
testing_time = []
repetition = [] 

for node in tqdm(nodes):
    for rep in range(reps):
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=10000, random_state=0+rep, stratify=y)

        #X, y, test_size=test_sample, train_size=train_sample, random_state=random_state+rep, stratify=y)
        uncalibrated_nn = getNN(input_size=X.shape[1], num_classes=np.max(
                    y)+1, layer_size=node)
        history = uncalibrated_nn.fit(
            X_train, keras.utils.to_categorical(y_train), **fit_kwargs
        )

        print('fitting kdn')
        model_kdn = kdn(network=uncalibrated_nn)
        start = timeit.timeit()
        model_kdn.fit(X_train, y_train, k=1)
        end_time = timeit.timeit()


        start_inference = timeit.timeit()
        model_kdn.predict_proba(X_train[:1000], distance='Geodesic')
        end_time_inference = timeit.timeit()

        polytope.append(len(model_kdn.polytope_means))
        training_time.append(end_time-start)
        testing_time.append(end_time_inference-start_inference)
        repetition.append(reps)

df = {}
df['polytope number'] = polytope
df['training time'] = training_time
df['testing time'] = testing_time
df['reps'] = repetition

df = pd.DataFrame.from_dict(df)
df.to_csv('time_complexity.csv', index=False)

# %%
