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
#%%
seeds = [0,100,200,400]
# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

#%%
for seed in seeds:
    print("Doing seed ", seed)

    nn_file = 'resnet20_models/cifar_model_50000_'+str(seed)
    network = keras.models.load_model(nn_file)
    
    model_kdn = kdn(
        network=network
    )
    model_kdn.fit(x_train, y_train, batch=10)

    with open('kdn_models/resnet_kdn_50000_'+str(seed)+'.pickle', 'wb') as f:
        pickle.dump(model_kdn, f)