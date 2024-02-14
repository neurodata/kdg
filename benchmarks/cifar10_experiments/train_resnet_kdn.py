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

    nn_file = 'resnet20_models/cifar10_'+str(seed)
    network = keras.models.load_model(nn_file)
    #network = keras.models.load_model('resnet20_models/cifar10_pretrained',custom_objects={'Custom':'contrastLoss'},compile=False)

    model_kdn = kdcnn(
        network=network,
        output_layer='flatten'
    )
    model_kdn.fit(x_train, y_train, k=1.2, batch=10)
    
    dump(model_kdn, 'resnet_kdn_cifar10_'+str(seed)+'.joblib')
# %%
