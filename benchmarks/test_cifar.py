#%%
from __future__ import print_function
import enum
#from cv2 import threshold
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10, mnist
import numpy as np
from kdg import kdn
import os
from kdg.utils import get_ece

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#%%
# Training parameters
batch_size = 120  # orig paper trained all networks with batch_size=128
epochs = 150
data_augmentation = True
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Load the MNIST data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

x_train_ = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test_ = x_test.reshape((x_test.shape[0], 32, 32, 3))
# Convert class vectors to binary class matrices.
y_train_ = keras.utils.to_categorical(y_train, num_classes)
y_test_ = keras.utils.to_categorical(y_test, num_classes)

input_shape = x_train_.shape[1:]
#%%
'''def define_model():
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3,3), kernel_initializer='he_uniform')(inputs)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(100, kernel_initializer='he_uniform')(x)
    x = Activation('relu')(x)
    x = Dense(10, kernel_initializer='he_uniform')(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model'''

def define_model():
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (5,5), kernel_initializer='he_uniform')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (5,5), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(96, (5,5), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (5,5), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(160, (5,5), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(10240, kernel_initializer='he_uniform')(x)
    x = Activation('relu')(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# %%
'''initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.98
    )

model = define_model()
opt = Adam(learning_rate=lr_schedule, beta_1=.98)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=20,
        # randomly shift images horizontally
        width_shift_range=0.2,
        # randomly shift images vertically
        height_shift_range=0.2,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0,
        )
datagen.fit(x_train_)
model.fit_generator(datagen.flow(x_train_, y_train_, batch_size=batch_size),
                        validation_data=(x_test_, y_test_),
                        epochs=epochs, verbose=1, workers=4,
                        use_multiprocessing=True)
#history = model.fit(x_train_, y_train_, epochs=epochs, batch_size=batch_size, validation_data=(x_test_, y_test_), verbose=1)
# %%
scores = model.evaluate(x_test_, y_test_, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
model.save('mnist_test')'''
# %%
network = keras.models.load_model('cifar_model')
#print(np.mean(np.argmax(network.predict(x_test), axis=1)==y_test.reshape(-1)))
model_kdn = kdn(
    network=network,
    k=1e300,
    threshold=1e-40,
    verbose=False,
)
model_kdn.fit(x_train_[:50000], y_train[:50000])

# %%
print(np.mean(model_kdn.predict(x_test)==y_test))
# %%
from matplotlib.pyplot import imshow
#import cv2 

digit= 0
polytope_id = 0
location = model_kdn.polytope_means[digit][polytope_id]
cov = model_kdn.polytope_cov[digit][polytope_id].toarray()
rng = np.random.default_rng()
pic = rng.multivariate_normal(
    mean = location,
    cov = cov,
    size=(1)
)
pic = pic.reshape(32, 32, 3)+x_train_mean
#pic = cv2.fastNlMeansDenoisingColored(pic)
'''for ii in range(28):
    for jj in range(28):
        if cov[ii,jj] < 1/255:
            print(cov[ii,jj])
            pic[ii,jj] = 0'''

imshow(pic)

# %%
from numpy.random import multivariate_normal as pdf
from matplotlib.pyplot import imshow

digit= 0
polytope_id = 15
location = model_kdn.polytope_means[digit][polytope_id]
cov = model_kdn.polytope_cov[digit][polytope_id]
pic = np.zeros(32*32*3, dtype=float)

for ii, mn in enumerate(location):
    if cov[ii] < 1e-300/255:
        pic[ii] = mn
    else:
        pic[ii] = np.random.normal(location[ii], [cov[ii]], 1)
#cov_mtrx = np.eye(len(location))*cov
#pic = pdf(location, cov_mtrx, 1).reshape(28,28) + x_train_mean
imshow(pic.reshape(32,32,3)+x_train_mean, cmap='gray')
# %%
