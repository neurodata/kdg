#%%
import tensorflow as tf
#tf.compat.v1.enable_eager_execution()

import cv2
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
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
from tensorflow.keras.datasets import cifar100, cifar10
from tqdm import tqdm
from vit_keras import vit, utils
from sklearn.model_selection import train_test_split
#%%
def cross_ent(logits, y):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    return tf.reduce_mean(losses)


def max_conf(logits):
    y = tf.argmax(logits, 1)
    y = tf.one_hot(y, num_classes)
    losses = -tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    return tf.reduce_mean(losses)

def gen_adv(x):
    eps = 0.025
    n_iters = 40
    step_size = 0.0075

    unif = tf.random.uniform(minval=-eps, maxval=eps, shape=tf.shape(x))
    x_adv = x + unif #tf.clip_by_value(x + unif, 0., 1.)
    
    for i in range(n_iters):
        x_adv = tf.Variable(x_adv)
        with tf.GradientTape() as tape:
            loss = max_conf(model(x_adv))
            grad = tape.gradient(loss, x_adv)
            g = tf.sign(grad)

        # import pdb;pdb.set_trace()
        x_adv_start = x_adv + step_size*g
        #x_adv = tf.clip_by_value(x_adv, 0., 1.)
        delta = x_adv - x_adv_start
        delta = tf.clip_by_value(delta, -eps, eps)
        x_adv = x_adv_start + delta

    return x_adv

def build_model():
        inputs = Input(shape=input_shape)
        x = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (image_size, image_size)))(inputs) #Resize image to  size 224x224
        base_model = vit.vit_b16(image_size=image_size, activation="sigmoid", pretrained=True,
                                include_top=False, pretrained_top=False)
        
        #base_model.trainable = False #Set false for transfer learning
        x = base_model(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        outputs = Dense(num_classes, activation="softmax")(x)

        model_final = Model(inputs=inputs, outputs=outputs)
        return model_final

#%%
batchsize = 32  # orig paper trained all networks with batch_size=128
epochs = 10
num_classes = 10
seed = 0

#%%
np.random.seed(seed)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = .001
    if epoch > 8:
        lr *= 0.5e-3
    elif epoch > 6:
        lr *= 1e-3
    elif epoch > 4:
        lr *= 1e-2
    elif epoch > 2:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

#%%
image_size = 256 #size after resizing image
input_shape = (32, 32, 3) #Cifar10 image size
num_classes = 10

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Normalize data.
x_train = x_train.astype('float32')/255.0 
x_test = x_test.astype('float32') /255.0

#x_train = np.array([cv2.resize(img, (image_size, image_size)) for img in tqdm(x_train)])
#x_test = np.array([cv2.resize(img, (image_size, image_size)) for img in tqdm(x_test)])

y_train = y_train.ravel()
y_test = y_test.ravel()
# Input image dimensions.
#%%

x_train, x_cal, y_train, y_cal = train_test_split(
            x_train, y_train, random_state=seed, shuffle=True)

#x_train = tf.convert_to_tensor(x_train)
#%%
model = build_model()

#nn_file = '/Users/jayantadey/kdg/benchmarks/cifar10_experiments/vit_model_'+str(seed)+'.keras'
#model = keras.models.load_model(nn_file)

model.summary()

#%%
iteration = len(y_train)//batchsize
steps_per_epoch = 200
optimizer = tf.optimizers.Adam(3e-4) 

for i in range(1,epochs+1):
    perm = np.arange(len(y_train))
    np.random.shuffle(perm)
    perm = perm[:(len(y_train)//batchsize)*batchsize].reshape(-1,batchsize)

    for j in tqdm(range(steps_per_epoch)):
        x_train_ = x_train[perm[j]]
        y_train_ = tf.one_hot(y_train[perm[j]], depth=num_classes)
        X_noise = tf.random.uniform([2*x_train_.shape[0], x_train_.shape[1], x_train_.shape[2], x_train_.shape[3]],minval=0,maxval=1)
        X_noise = gen_adv(X_noise)
        with tf.GradientTape() as tape:
            logits = model(x_train_)
            logits_noise = model(X_noise)
            loss_main = cross_ent(logits, y_train_)
            loss_acet = max_conf(logits_noise)
            loss = loss_main + loss_acet
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_err = np.mean(logits.numpy().argmax(1) != y_train_.numpy().argmax(1))
    print("Epoch {:03d}: loss_main={:.3f} loss_acet={:.3f} err={:.2%}".format(i, loss_main, loss_acet, train_err))

model.save('ACET_vit_'+str(seed)+'.keras')
# %%
