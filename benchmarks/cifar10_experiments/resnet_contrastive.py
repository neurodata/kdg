#%%
from __future__ import print_function
from tensorflow import keras
import tensorflow as tf 
import tensorflow_addons as tfa 
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D, UpSampling2D, Resizing
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
projection_unit = 100
batch_size = 32
num_epochs = 100
temperature = 0.05
#%%
class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)
    
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
    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
#%%
model = keras.Sequential()
base_model = ResNet50(
    weights=None, 
    include_top=False,
    input_shape=(32,32,3)
    )

'''for layer in base_model.layers[:-8]:
    layer.trainable = False'''

#model.add(UpSampling2D((7,7), input_shape=(32,32,3)))
model.add(base_model)
model.add(GlobalAveragePooling2D(name='avg_pool'))
model.add(Flatten())
model.add(Dense(projection_unit))

#model.layers[1].trainable = False
#%%
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                            cooldown=0,
                            patience=5,
                            min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

model.compile(
    optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
    loss=SupervisedContrastiveLoss(temperature),
)

#%%
# Load data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#y_train = tf.one_hot(y_train.reshape(-1), depth=projection_unit)
#x_ood = np.load('/Users/jayantadey/kdg/benchmarks/300K_random_images.npy').astype('float32')[:10000]
(x_cifar100, y_cifar100), (_,_) = cifar100.load_data()
y_cifar100 += 10
x_svhn = loadmat('/Users/jayantadey/svhn/train_32x32.mat')['X']
y_svhn = loadmat('/Users/jayantadey/svhn/train_32x32.mat')['y'] + 109

#x_svhn = loadmat('/cis/home/jdey4/train_32x32.mat')['X']
#y_svhn = loadmat('/cis/home/jdey4/train_32x32.mat')['y'] + 109

x_svhn = x_svhn.astype('float32')
x_tmp = np.zeros((x_svhn.shape[3],32,32,3), dtype=float)

for ii in range(x_svhn.shape[3]):
    x_tmp[ii,:,:,:] = x_svhn[:,:,:,ii]

x_svhn = x_tmp
del x_tmp

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_cifar100 = x_cifar100.astype('float32') / 255
x_svhn = x_svhn.astype('float32') / 255
x_noise = np.random.random_integers(0,high=255,size=(10000,32,32,3)).astype('float')/255.0
y_noise = 120*np.ones((10000,1), dtype='float32')

#x_ood = preprocess_input(x_ood)
#y_ood = np.array(range(len(x_ood))).reshape(-1,1)+11
#x_train = preprocess_input(x_train)
#x_test = preprocess_input(x_test)
#x_noise = preprocess_input(x_noise)
#x_cifar100 = preprocess_input(x_cifar100)
#x_svhn = preprocess_input(x_svhn)
#for channel in range(3):
#    x_train_mean = np.mean(x_train[:,:,:,channel])
#    x_train_std = np.std(x_train[:,:,:,channel])

 #   x_train[:,:,:,channel] -= x_train_mean
#    x_train[:,:,:,channel] /= x_train_std

#    x_test[:,:,:,channel] -= x_train_mean
#    x_test[:,:,:,channel] /= x_train_std

#    x_cifar100[:,:,:,channel] -= x_train_mean
 #   x_cifar100[:,:,:,channel] /= x_train_std

 #   x_svhn[:,:,:,channel] -= x_train_mean
  #  x_svhn[:,:,:,channel] /= x_train_std

   # x_noise[:,:,:,channel] -= x_train_mean
    #x_noise[:,:,:,channel] /= x_train_std

    #x_ood[:,:,:,channel] -= x_train_mean
    #x_ood[:,:,:,channel] /= x_train_std

#x_train = np.concatenate((x_train, x_noise, x_ood))
#y_train = np.concatenate((y_train, y_noise, y_ood))

#x_train = np.concatenate((x_train, x_cifar100, x_svhn, x_noise))
#y_train = np.concatenate((y_train, y_cifar100, y_svhn, y_noise))

#x_train = np.concatenate((x_train, x_noise))
#y_train = np.concatenate((y_train, y_noise))


#%%
history = model.fit(x=x_train[:10], y=y_train[:10], batch_size=batch_size, epochs=num_epochs, shuffle=True, callbacks=callbacks)
#%%
for layer_id, layer in enumerate(model.layers):
    pretrained_weights = model.layers[layer_id].get_weights()
    weights.append(
        pretrained_weights
    )

with open('pretrained_weight_contrast_finetune.pickle', 'wb') as f:
    pickle.dump(weights, f)

model.save('resnet20_models/cifar10_pretrained_contrast')
# %%
