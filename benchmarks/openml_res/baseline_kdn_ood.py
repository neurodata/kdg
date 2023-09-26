#%%
from kdg.utils import get_ece, get_ace
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
from kdg.utils import get_ece, sample_unifrom_circle
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from scikeras.wrappers import KerasClassifier
from sklearn.calibration import CalibratedClassifierCV as calcv
# %%
root_dir = "openml_kdn_res_baseline_ood"

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
        "epochs": 200,
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

#%%
class nnwrapper(KerasClassifier):
  
  def predict_proba(self, X):
      return self.model.predict(X)
# %%
def experiment(dataset_id, n_estimators=500, reps=10, random_state=42):
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
    
    print(dataset_id)
    
    X /= np.max(
        np.linalg.norm(X, 2, axis=1)
    )
    _, y = np.unique(y, return_inverse=True)
    
    '''for ii in range(X.shape[1]):
        unique_val = np.unique(X[:,ii])
        if len(unique_val) < 10:
            return'''
        
    total_sample = X.shape[0]
    test_sample = total_sample//3
    train_sample = total_sample-test_sample
    
    r = []    
    conf_nn_isotonic = []
    conf_nn_sigmoid = []
    distances = np.arange(1, 5.5, .5)

    for rep in range(reps):
        X_train, X_test, y_train, y_test = train_test_split(
                     X, y, test_size=test_sample, train_size=train_sample, random_state=random_state+rep)
        
    for rep in range(reps):
        X_train, X_test, y_train, y_test = train_test_split(
                 X, y, test_size=test_sample, train_size=train_sample, random_state=random_state+rep, stratify=y)
        X_train, X_cal, y_train, y_cal = train_test_split(
                X_train, y_train, train_size=0.67, random_state=random_state+rep, stratify=y_train)
            
        uncalibrated_nn = KerasClassifier(build_fn=getNN(input_size=X_train.shape[1], num_classes=np.max(y_train)+1),epochs=5, batch_size=32)  
        history = uncalibrated_nn.fit(X_train, keras.utils.to_categorical(y_train), **fit_kwargs)

        calibrated_nn_isotonic = calcv(uncalibrated_nn, method = 'isotonic', ensemble=False, cv='prefit')
        calibrated_nn_isotonic.fit(X_cal, y_cal)

        calibrated_nn_sigmoid = calcv(uncalibrated_nn, method = 'sigmoid', ensemble=False, cv='prefit')
        calibrated_nn_sigmoid.fit(X_cal, y_cal)

        y_proba_isotonic = calibrated_nn_isotonic.predict_proba(X_test)
        y_proba_sigmoid = calibrated_nn_sigmoid.predict_proba(X_test)
        conf_nn_isotonic.append(
            np.nanmean(
                    np.max(y_proba_isotonic, axis=1)
                )
        )
        conf_nn_sigmoid.append(
            np.nanmean(
                    np.max(y_proba_sigmoid, axis=1)
                )
        )
        r.append(
            0
        )

        for distance in distances:
            X_ood = sample_unifrom_circle(1000, r=distance, p=X_train.shape[1]).astype('float32')
            y_proba_isotonic = calibrated_nn_isotonic.predict_proba(X_ood)
            y_proba_sigmoid = calibrated_nn_sigmoid.predict_proba(X_ood)    

            conf_nn_isotonic.append(
            np.nanmean(
                    np.max(y_proba_isotonic, axis=1)
                )
            )
            conf_nn_sigmoid.append(
                np.nanmean(
                        np.max(y_proba_sigmoid, axis=1)
                    )
            )
            r.append(
                distance
            )




    df = pd.DataFrame() 
    df['conf_nn_isotonic'] = conf_nn_isotonic
    df['conf_nn_sigmoid'] = conf_nn_sigmoid
    df['distance'] = r

    filename = 'Dataset_' + str(dataset_id) + '.csv'
    df.to_csv(os.path.join(root_dir, filename))  

#%%
benchmark_suite = openml.study.get_suite('OpenML-CC18')
#data_id_not_done = [28, 554, 1485, 40996, 41027, 23517, 40923, 40927]

Parallel(n_jobs=-2,verbose=1)(
        delayed(experiment)(
                dataset_id,
                ) for dataset_id in benchmark_suite.data
            ) 

'''for dataset_id in benchmark_suite.data:
    experiment(dataset_id) '''            