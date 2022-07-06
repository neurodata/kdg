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
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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

x_train_ = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test_ = x_test.reshape((x_test.shape[0], 28, 28, 1))
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
network = keras.models.load_model('mnist_test')
#print(np.mean(np.argmax(network.predict(x_test), axis=1)==y_test.reshape(-1)))
model_kdn = kdn(
    network=network,
    k=1e300,
    verbose=False,
)
model_kdn.fit(x_train_, y_train)

# %%
print(np.mean(model_kdn.predict(x_test)==y_test))
# %%
def _compute_log_likelihood_1d(X, location, variance): 
    if variance < 1e-2:
        return 0
    else:                 
        return -(X-location)**2/(2*variance) - .5*np.log(2*np.pi*variance)

def _compute_log_likelihood(model, X, label, polytope_idx):
    polytope_mean = model.polytope_means[label][polytope_idx]
    polytope_cov = model.polytope_cov[label][polytope_idx]
    likelihood = np.zeros(X.shape[0], dtype = float)

    for ii in range(model.feature_dim):
        likelihood += _compute_log_likelihood_1d(X[:,ii], polytope_mean[ii], polytope_cov[ii])

    likelihood += np.log(model.polytope_cardinality[label][polytope_idx]) -\
        np.log(model.total_samples_this_label[label])

    return likelihood
        
def predict_proba(model, X, return_likelihood=False):
    r"""
    Calculate posteriors using the kernel density forest.
    Parameters
    ----------
    X : ndarray
        Input data matrix.
    """
    
    #X = check_array(X)
    X = X.reshape(
        X.shape[0],
        -1
    )
    log_likelihoods = np.zeros(
        (np.size(X,0), len(model.labels)),
        dtype=float
    )
    
    for ii,label in enumerate(model.labels):
        total_polytope_this_label = len(model.polytope_means[label])
        tmp_ = np.zeros((X.shape[0],total_polytope_this_label), dtype=float)

        for polytope_idx,_ in enumerate(model.polytope_means[label]):
            tmp_[:,polytope_idx] = _compute_log_likelihood(model, X, label, polytope_idx) 
        
        max_pow = np.max(
                np.concatenate(
                    (
                        tmp_,
                        model.global_bias*np.ones((X.shape[0],1), dtype=float)
                    ),
                    axis=1
                ),
                axis=1
            )
        pow_exp = np.nan_to_num(
            max_pow.reshape(-1,1)@np.ones((1,total_polytope_this_label), dtype=float)
        )
        tmp_ -= pow_exp
        likelihoods = np.sum(np.exp(tmp_), axis=1) +\
                np.exp(model.global_bias - pow_exp[:,0]) 
            
        likelihoods *= model.prior[label] 
        log_likelihoods[:,ii] = np.log(likelihoods) + pow_exp[:,0]

    max_pow = np.nan_to_num(
        np.max(log_likelihoods, axis=1).reshape(-1,1)@np.ones((1,len(model.labels)))
    )
    log_likelihoods -= max_pow
    likelihoods = np.exp(log_likelihoods)

    total_likelihoods = np.sum(likelihoods, axis=1)

    proba = (likelihoods.T/total_likelihoods).T
    
    if return_likelihood:
        return proba, likelihoods
    else:
        return proba 

def predict(model, X):
    r"""
    Perform inference using the kernel density forest.
    Parameters
    ----------
    X : ndarray
        Input data matrix.
    """
    return np.argmax(predict_proba(model, X), axis = 1)
# %%
from numpy.random import multivariate_normal as pdf
from matplotlib.pyplot import imshow

digit= 4
polytope_id = 0
location = model_kdn.polytope_means[digit][polytope_id]
cov = model_kdn.polytope_cov[digit][polytope_id]
pic = np.zeros(28*28, dtype=float)

for ii, mn in enumerate(location):
    if cov[ii] < 1e-3/255:
        pic[ii] = mn
    else:
        pic[ii] = np.random.normal(location[ii], [cov[ii]], 1)
#cov_mtrx = np.eye(len(location))*cov
#pic = pdf(location, cov_mtrx, 1).reshape(28,28) + x_train_mean
imshow(pic.reshape(28,28)+x_train_mean, cmap='gray')
#%%
imshow(location.reshape(28,28)+x_train_mean, cmap='gray')
# %%
from matplotlib.pyplot import imshow
import cv2 

digit= 2
polytope_id = 1
location = model_kdn.polytope_means[digit][polytope_id]
cov = model_kdn.polytope_cov[digit][polytope_id]
rng = np.random.default_rng()
pic = rng.multivariate_normal(
    mean = location,
    cov = cov,
    size=(1)
)
pic = pic.reshape(28,28)+x_train_mean
#pic = cv2.fastNlMeansDenoisingColored(pic)
'''for ii in range(28):
    for jj in range(28):
        if cov[ii,jj] < 1/255:
            print(cov[ii,jj])
            pic[ii,jj] = 0'''

imshow(pic, cmap='gray')

# %%
