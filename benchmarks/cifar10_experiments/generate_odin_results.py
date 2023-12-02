# %%
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Model
from kdg import kdf, kdn
from tensorflow.keras.datasets import cifar10, cifar100
import timeit
from scipy.io import loadmat
import random
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import tensorflow as tf
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
tf.compat.v1.enable_eager_execution()
from kdg.utils import generate_gaussian_parity, generate_ood_samples, generate_spirals, generate_ellipse
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from scipy.io import loadmat
import random
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tqdm import tqdm
import torch
#%%
def predict_proba(model, x, T):
    logits = model(x)/T
    proba = tf.nn.softmax(logits,axis=1).numpy()

    return proba

def fpr_at_95_tpr(conf_in, conf_out):
    TPR = 95
    PERC = np.percentile(conf_in, 100-TPR)
    #FP = np.sum(conf_out >=  PERC)
    FPR = np.sum(conf_out >=  PERC)/len(conf_out)
    return FPR, PERC

def cross_ent(logits, y):
    logits = tf.nn.softmax(logits, axis=1)
    losses = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    return losses(logits, y)


def gen_adv(x, eps, T):
    x = tf.Variable(x)
    with tf.GradientTape() as tape:
        logits = model(x)/T
        label = logits.numpy().argmax(1)
        label = tf.one_hot(label, depth=num_classes)
        loss = cross_ent(logits, label)

        grad = tape.gradient(loss, x).numpy()
    grad = (grad>0.0)*1.0
    grad = (grad-0.5)*2
    grad[0][0] = (grad[0][0] )/(63.0/255.0)
    grad[0][1] = (grad[0][1] )/(62.1/255.0)
    grad[0][2] = (grad[0][2])/(66.7/255.0)
    
    grad = tf.Variable(grad, dtype=float)

    x_tilde = x - eps*grad

    return x_tilde
#%%
batchsize = 128 # orig paper trained all networks with batch_size=128
num_classes = 100
seeds = [100, 200, 300, 400]
'''T = 5.0 #cifar10 params
eps = 0.028'''
T = 5.0
eps = 0.026
#%%

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3

# Model version

# Computed depth from supplied model parameter n
depth = n * 6 + 2

# Model name, depth and version
model_type = 'ResNet%d' % (depth)

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_ntormalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

#%%
# Load the CIFAR10 and CIFAR100 data.
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
(_, _), (x_cifar10, y_cifar10) = cifar10.load_data()
x_noise = np.random.random_integers(0,high=255,size=(1000,32,32,3)).astype('float32')/255.0
x_svhn = loadmat('/Users/jayantadey/DF-CNN/data_five/SVHN/train_32x32.mat')['X']
y_svhn = loadmat('/Users/jayantadey/DF-CNN/data_five/SVHN/train_32x32.mat')['y']
test_ids =  random.sample(range(0, x_svhn.shape[3]), 2000)
x_svhn = x_svhn[:,:,:,test_ids].astype('float32')
x_tmp = np.zeros((2000,32,32,3), dtype=float)

for ii in range(2000):
    x_tmp[ii,:,:,:] = x_svhn[:,:,:,ii]

x_svhn = x_tmp
del x_tmp
# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_cifar10 = x_cifar10.astype('float32') / 255
x_svhn = x_svhn.astype('float32') / 255


for channel in range(3):
    x_train_mean = np.mean(x_train[:,:,:,channel])
    x_train_std = np.std(x_train[:,:,:,channel])

    x_train[:,:,:,channel] -= x_train_mean
    x_train[:,:,:,channel] /= x_train_std

    x_test[:,:,:,channel] -= x_train_mean
    x_test[:,:,:,channel] /= x_train_std

    x_cifar10[:,:,:,channel] -= x_train_mean
    x_cifar10[:,:,:,channel] /= x_train_std

    x_svhn[:,:,:,channel] -= x_train_mean #+ 1
    x_svhn[:,:,:,channel] /= x_train_std

    x_noise[:,:,:,channel] -= x_train_mean
    x_noise[:,:,:,channel] /= x_train_std
#%%
x_train = gen_adv(x_train, eps, T)
x_test = gen_adv(x_test, eps, T)
x_cifar10 = gen_adv(x_cifar10, eps, T)
x_svhn = gen_adv(x_svhn, eps, T)
x_noise = gen_adv(x_noise, eps, T)
#%% Load model file
input_shape = x_train.shape

for seed in seeds: 
    print('doing seed ',seed)
    model = resnet_v1(input_shape=input_shape[1:], depth=depth, num_classes=num_classes)

    #load pretrained model
    #pretrained_model = keras.models.load_model('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/resnet20_models/cifar100_model_new_'+str(seed))
    pretrained_model = keras.models.load_model('resnet20_models/cifar_model_new_'+str(seed))
    for layer_id, layer in enumerate(model.layers):
        pretrained_weights = pretrained_model.layers[layer_id].get_weights()
        layer.set_weights(pretrained_weights)
        layer.trainable = False
    
    proba_in = predict_proba(model, x_test, T) 
    proba_cifar10 = predict_proba(model, x_cifar10, T)
    proba_svhn = predict_proba(model, x_svhn, T)
    proba_noise = predict_proba(model, x_noise, T)

    summary = (proba_in, proba_cifar10, proba_svhn, proba_noise)
    file_to_save = 'resnet20_cifar100_ODIN_'+str(seed)+'.pickle'

    with open(file_to_save, 'wb') as f:
        pickle.dump(summary, f)

# %%
