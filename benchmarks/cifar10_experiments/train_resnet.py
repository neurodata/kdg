#%%
from __future__ import print_function
from tensorflow import keras
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.applications.resnet50 import ResNet50 
from tensorflow.keras.applications.resnet50 import preprocess_input
#from keras import ops
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import pickle
#%%
weights = []
num_classes = 10
learning_rate = 0.001
batch_size = 32
projection_units = 512
num_epochs = 5
seed = 0
#%% load pretrained model weights
print('loading weights')
with open('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/pretrained_weight_contrast_finetune.pickle', 'rb') as f:
    weights = pickle.load(f)

#%%
num_classes = 10
input_shape = (32, 32, 3)
#%%
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-4
    if epoch > 18:
        lr *= 0.5e-3
    elif epoch > 16:
        lr *= 1e-3
    elif epoch > 8:
        lr *= 1e-2
    elif epoch > 4:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
#%%
model = keras.Sequential()
base_model = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )

model.add(base_model)
model.add(GlobalAveragePooling2D(name='avg_pool'))
model.add(Flatten())
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(projection_units))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(
            Dense(
                    num_classes,
                    activation='softmax'
                )
        )

model.build()
#%%
for layer_id, weight in enumerate(weights):
    print(model.layers[layer_id].name)
    model.layers[layer_id].set_weights(weight)
    model.layers[layer_id].trainable = False

#model.layers[4].set_weights(np.array([weights[-1][0], weights[-1][1]]))
#model.layers[4].trainable = False
#%%
model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=lr_schedule(0)),
            metrics=['accuracy'])

#print(model.summary())

filepath = 'cifar10.h5'
checkpoint = ModelCheckpoint(filepath=filepath,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                            cooldown=0,
                            patience=5,
                            min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]
#%%
# Load data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#(x_cifar100, y_cifar100), (_,_) = cifar100.load_data()
#x_svhn = loadmat('/Users/jayantadey/svhn/train_32x32.mat')['X']
#y_svhn = loadmat('/Users/jayantadey/svhn/train_32x32.mat')['y'] + 109

#x_svhn = loadmat('/cis/home/jdey4/train_32x32.mat')['X']
#y_svhn = loadmat('/cis/home/jdey4/train_32x32.mat')['y'] + 109

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
'''x_svhn = x_svhn.astype('float32')
x_tmp = np.zeros((x_svhn.shape[3],32,32,3), dtype=float)

for ii in range(x_svhn.shape[3]):
    x_tmp[ii,:,:,:] = x_svhn[:,:,:,ii]

x_svhn = x_tmp
del x_tmp'''
# Input image dimensions.
#input_shape = x_train.shape[1:]

# Normalize data.
'''x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_cifar100 = x_cifar100.astype('float32') / 255
x_svhn = x_svhn.astype('float32') / 255


for channel in range(3):
    x_train_mean = np.mean(x_train[:,:,:,channel])
    x_train_std = np.std(x_train[:,:,:,channel])

    x_train[:,:,:,channel] -= x_train_mean
    x_train[:,:,:,channel] /= x_train_std

    x_test[:,:,:,channel] -= x_train_mean
    x_test[:,:,:,channel] /= x_train_std

    x_cifar100[:,:,:,channel] -= x_train_mean
    x_cifar100[:,:,:,channel] /= x_train_std

    x_svhn[:,:,:,channel] -= x_train_mean
    x_svhn[:,:,:,channel] /= x_train_std'''

x_train, x_cal, y_train, y_cal = train_test_split(
                x_train, y_train, train_size=0.9, random_state=seed, stratify=y_train)

y_train_one_hot = keras.utils.to_categorical(y_train, num_classes)
y_cal_one_hot = keras.utils.to_categorical(y_cal, num_classes)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)
#%%
model.fit(x_train, y_train_one_hot,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(x_test, y_test_one_hot),
        shuffle=True,
        callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test_one_hot, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.save('resnet20_models/cifar_finetune10_'+str(seed))
# %%
