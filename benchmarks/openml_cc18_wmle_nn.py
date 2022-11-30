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
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import activations
import os
from os import listdir, getcwd 
# %%
def getNN(compile_kwargs, input_size, num_classes):
    network_base = keras.Sequential()
    #initializer = keras.initializers.GlorotNormal(seed=0)
    network_base.add(keras.layers.Dense(500, kernel_initializer=initializer, input_shape=(input_size,)))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(500, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(500, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(500, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(units=num_classes, activation="softmax", kernel_initializer=initializer))
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

def experiment_random_sample(dataset_id, folder, reps=10):
    #print(dataset_id)
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
    
    total_sample = X.shape[0]

    test_sample = total_sample//3

    max_sample = total_sample - test_sample
    train_samples = np.logspace(
        np.log10(2),
        np.log10(max_sample),
        num=10,
        endpoint=True,
        dtype=int
        )
    
    err = []
    err_nn = []
    ece = []
    ece_nn = []
    kappa = []
    kappa_nn = []
    mc_rep = []
    samples = []
    param_kdn = []
    param_nn = []
    indices = list(range(total_sample))

    # NN params
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
    for train_sample in train_samples:     
        for rep in range(reps):
            print("sample", train_sample, " reps ", rep, dataset_id)
            np.random.shuffle(indices)
            indx_to_take_train = indices[:train_sample]
            indx_to_take_test = indices[-test_sample:]

            unique_y, y_converted = np.unique(y[indx_to_take_train], return_inverse=True)
            total_class = len(unique_y)
            
            vanilla_nn = getNN(compile_kwargs, X.shape[-1], total_class)
            vanilla_nn.fit(
                X[indx_to_take_train], 
                keras.utils.to_categorical(y_converted, num_classes=total_class), 
                **fit_kwargs
                )
            print(vanilla_nn)


            model_kdn = kdn(
                        network=vanilla_nn,
                        verbose=False,
                    )
            model_kdn.fit(X[indx_to_take_train], y_converted)
            #print(model_kdn.polytope_means)
            proba_kdn = model_kdn.predict_proba(X[indx_to_take_test])
            proba_nn = vanilla_nn.predict(X[indx_to_take_test])
            predicted_label_kdn = np.argmax(proba_kdn, axis = 1)
            predicted_label_nn = np.argmax(proba_nn, axis = 1)

            for iii, _ in enumerate(predicted_label_kdn):
                predicted_label_kdn[iii] = unique_y[predicted_label_kdn[iii]]
                predicted_label_nn[iii] = unique_y[predicted_label_nn[iii]]

            err.append(
                1 - np.mean(
                        predicted_label_kdn==y[indx_to_take_test]
                    )
            )
            err_nn.append(
                1 - np.mean(
                    predicted_label_nn==y[indx_to_take_test]
                )
            )
            kappa.append(
                cohen_kappa_score(predicted_label_kdn, y[indx_to_take_test])
            )
            kappa_nn.append(
                cohen_kappa_score(predicted_label_nn, y[indx_to_take_test])
            )
            ece.append(
                get_ece(proba_kdn, predicted_label_kdn, y[indx_to_take_test])
            )
            ece_nn.append(
                get_ece(proba_nn, predicted_label_nn, y[indx_to_take_test])
            )
            samples.append(
                train_sample
            )
            param_kdn.append(
                count_kdn_param(model_kdn)
            )
            param_nn.append(
                count_nn_param(vanilla_nn)
            )
            mc_rep.append(rep)

    df = pd.DataFrame() 
    df['err_kdn'] = err
    df['err_nn'] = err_nn
    df['kappa_kdn'] = kappa
    df['kappa_nn'] = kappa_nn
    df['ece_kdn'] = ece
    df['ece_nn'] = ece_nn
    df['rep'] = mc_rep
    df['samples'] = samples
    df['kdn_param'] = param_kdn
    df['nn_param'] = param_nn

    df.to_csv(folder+'/'+'openML_cc18_'+str(dataset_id)+'_nn.csv')


#%%
folder = 'openml_res/openml_nn'
#os.mkdir(folder)
#os.mkdir(folder_rf)
benchmark_suite = openml.study.get_suite('OpenML-CC18')
#current_dir = getcwd()
#files = listdir(current_dir+'/'+folder)

for dataset_id in openml.study.get_suite("OpenML-CC18").data:
    print('doing ', dataset_id)
    experiment_random_sample(
                dataset_id,
                folder
                )