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
from tensorflow.keras.datasets import cifar100, cifar10
from tqdm import tqdm

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

#%%
batchsize = 40  # orig paper trained all networks with batch_size=128
epochs = 10
data_augmentation = False
num_classes = 10
seed = 200

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
# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.ravel()
y_test = y_test.ravel()
# Input image dimensions.
input_shape = x_train.shape

# Normalize data.
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 

#%%
nn_file = 'resnet20_models/cifar_finetune10_'+str(seed)
model = keras.models.load_model(nn_file)

model.summary()

#%%
iteration = input_shape[0]//batchsize
optimizer = tf.optimizers.Adam(3e-4) 

for i in range(1,epochs+1):
    perm = np.arange(input_shape[0])
    np.random.shuffle(perm)
    perm = perm.reshape(-1,batchsize)

    for j in tqdm(range(iteration)):
        x_train_ = x_train[perm[j]]
        y_train_ = tf.one_hot(y_train[perm[j]], depth=num_classes)
        X_noise = tf.random.uniform([2*x_train_.shape[0], x_train_.shape[1], x_train_.shape[2], x_train_.shape[3]],minval=0,maxval=255)
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

model.save('resnet20_models/cifar10_ACET_'+str(seed))
# %%
