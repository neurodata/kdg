#%%
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
from tensorflow.keras.datasets import cifar100
from tqdm import tqdm

#%% load OOD data
ood_set = np.load('/Users/jayantadey/kdg/benchmarks/300K_random_images.npy')

#%%
def cross_ent(logits, y):
    logits = tf.nn.softmax(logits, axis=1)
    losses = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    return losses(logits, y)


def max_conf(logits):
    losses = -tf.reduce_mean(tf.reduce_mean(logits, 1) - tf.reduce_logsumexp(logits,axis=1))
    return losses


#%%
batchsize = 128 # orig paper trained all networks with batch_size=128
epochs = 30
data_augmentation = False
num_classes = 100
seed = 400

#%%
np.random.seed(seed)

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
# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train = y_train.ravel()
y_test = y_test.ravel()
# Input image dimensions.
input_shape = x_train.shape

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
ood_set = ood_set.astype('float32')/255

for channel in range(3):
    x_train_mean = np.mean(x_train[:,:,:,channel])
    x_train_std = np.std(x_train[:,:,:,channel])
    x_train[:,:,:,channel] -= x_train_mean
    x_train[:,:,:,channel] /= x_train_std
    x_test[:,:,:,channel] -= x_train_mean
    x_test[:,:,:,channel] /= x_train_std

    ood_set[:,:,:,channel] -= x_train_mean
    ood_set[:,:,:,channel] /= x_train_std

#%%
model = resnet_v1(input_shape=input_shape[1:], depth=depth, num_classes=num_classes)

#load pretrained model
pretrained_model = keras.models.load_model('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/resnet20_models/cifar100_model_new_'+str(seed))
#pretrained_model = keras.models.load_model('resnet20_models/cifar_model_new_'+str(seed))
for layer_id, layer in enumerate(model.layers[:-1]):
    pretrained_weights = pretrained_model.layers[layer_id].get_weights()
    layer.set_weights(pretrained_weights)
    layer.trainable = False

model.summary()
print(model_type)

test_logits = pretrained_model(x_test)
test_err = np.mean(test_logits.numpy().argmax(1) != y_test)
print('Pretrained error ',test_err)
#%%
iteration = input_shape[0]//batchsize
#optimizer = tf.optimizers.Adam(3e-3) 
lr = 3e-3
ood_batch_size = (ood_set.shape[0]//iteration)
#y_train_one_hot = tf.one_hot(y_train, depth=num_classes)
#model.fit(x_train, y_train_one_hot,
#                    batch_size=40,
#                    epochs=epochs,
 #                   shuffle=True)
for i in range(1,epochs+1):
    if i > .8*epochs:
        lr *= 0.5e-3
    elif i > .6*epochs:
        lr *= 1e-3
    elif i > .4*epochs:
        lr *= 1e-2
    elif i > .2*epochs:
        lr *= 1e-1

    optimizer = tf.optimizers.Adam(lr) 
    perm = np.arange(batchsize*iteration)
    perm_ood = np.arange(ood_batch_size*iteration)
    
    np.random.shuffle(perm)
    np.random.shuffle(perm_ood)
    perm = perm.reshape(-1,batchsize)
    perm_ood = perm_ood.reshape(-1,ood_batch_size)

    for j in tqdm(range(iteration)):
        x_train_ = x_train[perm[j]]
        y_train_ = tf.one_hot(y_train[perm[j]], depth=num_classes)
        X_ood = ood_set[perm_ood[j]]
        
        with tf.GradientTape() as tape:
            logits = model(x_train_, training=True)
            logits_ood = model(X_ood)
            loss_main = cross_ent(logits, y_train_)
            loss_ood = max_conf(logits_ood)
            loss = loss_main + 0.5*loss_ood
            grads = tape.gradient(loss, model.trainable_weights)
            #print(loss)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_err = np.mean(logits.numpy().argmax(1) != y_train_.numpy().argmax(1))
    test_logits = model(x_test)
    test_err = np.mean(test_logits.numpy().argmax(1) != y_test)

    print("Epoch {:03d}: loss_main={:.3f} loss_ood={:.3f} train err={:.2%} test err={:.2%}".format(i, loss_main, loss_ood, train_err, test_err))

model.save('resnet20_models/cifar100_OE_'+str(seed))
# %%
