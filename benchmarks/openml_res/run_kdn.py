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
from scikeras.wrappers import KerasClassifier

from numpy.random import seed
#%%
class nnwrapper(KerasClassifier):
  
  def predict_proba(self, X):
      return self.model.predict(X)
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


def experiment(dataset_id, layer_size = 1000, reps=5, random_state=42):

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
    
    min_val = np.min(X,axis=0)
    max_val = np.max(X, axis=0)
    X = (X-min_val)/(max_val-min_val+1e-12)
    _, y = np.unique(y, return_inverse=True)

    '''for ii in range(X.shape[1]):
        unique_val = np.unique(X[:,ii])
        if len(unique_val) < 10:
            return'''
        
    total_sample = X.shape[0]
    test_sample = total_sample//3 if total_sample//3<1000 else 1000
    train_samples = np.logspace(
            np.log10(100),
            np.log10(total_sample-test_sample),
            num=4,
            endpoint=True,
            dtype=int
        )
    err = []
    err_geod = []
    err_dn = []
    ece = []
    ece_geod = []
    ece_dn = []
    err_isotonic = []
    ece_isotonic = []
    err_sigmoid = []
    ece_sigmoid = []
    mc_rep = []
    samples = []

    for train_sample in train_samples:
        for rep in range(reps):
            X_train, X_test, y_train, y_test = train_test_split(
                     X, y, test_size=test_sample, train_size=train_sample, random_state=random_state+rep, stratify=y)
            
            #print(X_train.shape, X_test.shape)
            nn = getNN(input_size=X_train.shape[1], num_classes=np.max(y_train)+1, layer_size=layer_size)
            seed(random_state+rep)
            history = nn.fit(X_train, keras.utils.to_categorical(y_train), **fit_kwargs)
            model_kdn = kdn(network=nn)
            model_kdn.fit(X_train, y_train, k=int(np.ceil(train_sample*95/100)))
            model_kdn.global_bias = -1e100
            proba_kdn = model_kdn.predict_proba(X_test)
            proba_kdn_geod = model_kdn.predict_proba(X_test, distance='Geodesic')
            proba_dn = model_kdn.network.predict(X_test)
            predicted_label_kdn = np.argmax(proba_kdn, axis = 1)
            predicted_label_kdn_geod = np.argmax(proba_kdn_geod, axis = 1)
            predicted_label_dn = np.argmax(proba_dn, axis = 1)

            err.append(
                1 - np.mean(
                        predicted_label_kdn==y_test
                    )
            )
            err_geod.append(
                1 - np.mean(
                        predicted_label_kdn_geod==y_test
                    )
            )
            err_dn.append(
                1 - np.mean(
                    predicted_label_dn==y_test
                )
            )
            ece.append(
                get_ece(proba_kdn, y_test, n_bins=15)
            )
            ece_geod.append(
                get_ece(proba_kdn_geod, y_test, n_bins=15)
            )
            ece_dn.append(
                get_ece(proba_dn, y_test, n_bins=15)
            )
            samples.append(
                train_sample
            )
            mc_rep.append(rep)

            ### train baseline ###
            if train_sample >= 600:
                X_train, X_cal, y_train, y_cal = train_test_split(
                X_train, y_train, train_size=0.9, random_state=random_state+rep, stratify=y_train)
            else:
                X_train, X_cal, y_train, y_cal = train_test_split(
                X_train, y_train, train_size=0.7, random_state=random_state+rep)

            #print(X_train.shape, X_cal.shape)
            uncalibrated_nn = KerasClassifier(build_fn=getNN(input_size=X_train.shape[1], num_classes=np.max(y_train)+1, layer_size=layer_size))
            seed(random_state+rep)
            history = uncalibrated_nn.fit(X_train, keras.utils.to_categorical(y_train), **fit_kwargs)
            
            calibrated_nn_isotonic = calcv(uncalibrated_nn, method = 'isotonic', ensemble=False, cv='prefit')
            calibrated_nn_isotonic.fit(X_cal, y_cal)

            calibrated_nn_sigmoid = calcv(uncalibrated_nn, method = 'sigmoid', ensemble=False, cv='prefit')
            calibrated_nn_sigmoid.fit(X_cal, y_cal)

            y_proba_isotonic = calibrated_nn_isotonic.predict_proba(X_test)
            y_hat_isotonic = np.argmax(y_proba_isotonic, axis=1)

            y_proba_sigmoid = calibrated_nn_sigmoid.predict_proba(X_test)
            y_hat_sigmoid = np.argmax(y_proba_sigmoid, axis=1)


            err_isotonic.append(
                1 - np.mean(
                        y_hat_isotonic==y_test
                    )
            )
            ece_isotonic.append(
                get_ece(y_proba_isotonic, y_test)
            )
            err_sigmoid.append(
                1 - np.mean(
                        y_hat_sigmoid==y_test
                    )
            )
            ece_sigmoid.append(
                get_ece(y_proba_sigmoid, y_test)
            )           
    df = pd.DataFrame() 
    df['err_kdn'] = err
    df['err_kdn_geod'] = err_geod
    df['err_dn'] = err_dn
    df['ece_kdn'] = ece
    df['ece_kdn_geod'] = ece_geod
    df['ece_dn'] = ece_dn
    df['err_isotonic'] = err_isotonic
    df['ece_isotonic'] = ece_isotonic
    df['err_sigmoid'] = err_sigmoid
    df['ece_sigmoid'] = ece_sigmoid
    df['rep'] = mc_rep
    df['samples'] = samples

    df.to_csv(os.path.join(root_dir, filename))

# %%
benchmark_suite = openml.study.get_suite('OpenML-CC18')
#data_id_not_done = [554, 40996, 40923, 40927, 41027]
#23527
'''for dataset_id in benchmark_suite.data:
    print("Doing data ", dataset_id)
    experiment(dataset_id)
    '''
Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment)(
                dataset_id,
                ) for dataset_id in benchmark_suite.data
            )

#%%
'''
benchmark_suite = openml.study.get_suite('OpenML-CC18')
root_dir = "/Users/jayantadey/kdg/benchmarks/openml_res/openml_kdn_res"
for dataset_id in benchmark_suite.data:
    filename = 'Dataset_' + str(dataset_id) + '.csv'
    if os.path.exists(os.path.join(root_dir, filename)):
        #print('exists')
        continue
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )
    if np.mean(is_categorical) >0:
        continue

    if np.isnan(np.sum(y)):
        continue

    if np.isnan(np.sum(X)):
        continue
    
    print(dataset_id)
'''