# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle
from keras.models import Model
from kdg import kdcnn, kdf, kdn
import pickle
from tensorflow.keras.datasets import cifar10
import timeit
from joblib import dump, load
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as bknd
#%%
seeds = [100]
# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

for channel in range(3):
    x_train_mean = np.mean(x_train[:,:,:,channel])
    x_train_std = np.std(x_train[:,:,:,channel])
    x_train[:,:,:,channel] -= x_train_mean
    x_train[:,:,:,channel] /= x_train_std
    x_test[:,:,:,channel] -= x_train_mean
    x_test[:,:,:,channel] /= x_train_std

#%%
x_train, x_cal, y_train, y_cal = train_test_split(
                x_train, y_train, train_size=0.9, random_state=seeds[0], stratify=y_train)
#%%
for seed in seeds:
    print("Doing seed ", seed)

    nn_file = 'resnet20_models/cifar100_model_new_'+str(seed)
    network = keras.models.load_model(nn_file)
    
    model_kdn = kdcnn(
        network=network,
        output_layer='flatten'
    )
    model_kdn.fit(x_train, y_train, k=1.2, batch=10)
    
    #dump(model_kdn, 'resnet_kdn_50000_cifar100_'+str(seed)+'.joblib')
# %%
def _get_polytope_ids(model, X):
       total_samples = X.shape[0]
       array_shape = [-1]
       array_shape.extend(
                list(model.network.get_layer(
                        model.output_layer
                    ).input.shape[1:]
                )
       )
       X = X.reshape(array_shape)
       outputs = []
       
       ii = 0
       while model.output_layer not in model.network.layers[ii].name:
           ii += 1
       
       inp = model.network.layers[ii].input

       for layer in model.network.layers[ii:]:
           if 'activation' in layer.name:
               print('got one')
               outputs.append(layer.output)

       # add the final layer
       outputs.append(layer.output)

       functor = bknd.function(inp, outputs)
       layer_outs = functor(X)
       print(len(layer_outs))
       activation = []
       for layer_out in layer_outs[:-1]:
           print('Dhuksi')
           activation.append(
               (layer_out>0).astype('bool').reshape(total_samples, -1)
           )
        # add the last layer
       activation.append(
               (layer_outs[-1]>1/len(model.labels)).astype('bool').reshape(total_samples, -1)
           )
       polytope_ids = np.concatenate(activation, axis=1)
      
       return polytope_ids
# %%
