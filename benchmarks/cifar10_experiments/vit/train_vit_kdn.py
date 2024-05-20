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
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout, Activation, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.datasets import cifar10
from vit_keras import vit, utils
import tensorflow_addons as tfa

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import gc
import numpy as np
#%%
input_shape = (32, 32, 3) #Cifar10 image size
image_size = 256 #size after resizing image
num_classes = 10

seeds = [3]
# Load the CIFAR10 data.
(train_data, train_label), (test_data, test_label) = cifar10.load_data()
train_data = (train_data/255.).astype("float16")
test_data = (test_data/255.).astype("float16")


#%%
for seed in seeds:
    print("Doing seed ", seed)
    x_train, x_cal, y_train, y_cal = train_test_split(
                train_data, train_label, random_state=seed, shuffle=True)
    nn_file = '/Users/jayantadey/kdg/benchmarks/cifar10_experiments/vit_model_'+str(seed)+'.keras'
    network = keras.models.load_model(nn_file)
    #network = keras.models.load_model('resnet20_models/cifar10_pretrained',custom_objects={'Custom':'contrastLoss'},compile=False)

    model_kdn = kdcnn(
        network=network,
        output_layer='flatten'
    )
    model_kdn.fit(x_train, y_train, X_val=x_cal, y_val=y_cal, batch=10)
    model_kdn.global_bias = -2e6
    dump(model_kdn, 'resnet_kdn_cifar10_vit_'+str(seed)+'.joblib')
# %%
