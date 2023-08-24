#%%
from numpy import dtype
from kdg import kdf, kdn, kdcnn
from kdg.utils import get_ece, get_ace, plot_reliability, sample_unifrom_circle
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from tensorflow.keras import activations
from tensorflow.keras import backend as bknd
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import numpy as np
import random
import pickle 
from joblib import Parallel, delayed
# %%
def getLeNet(input_shape, num_classes):
    model = keras.Sequential()
    inputs = Input(shape=input_shape)
    x = Conv2D(6, (3,3), kernel_initializer='he_uniform')(inputs)
    x = Activation('relu')(x)
    x = AveragePooling2D()(x)
    x = Conv2D(16, (3,3), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((2,2))(x)

    x = Flatten()(x)
    x = Dense(120, kernel_initializer='he_uniform')(x)
    x = Activation('relu')(x)
    x = Dense(84, kernel_initializer='he_uniform')(x)
    x = Activation('relu')(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
    
# %%
### Hyperparameters ###
subtract_pixel_mean = False
normalize = True
classes_to_consider = [[0,1], [2,3],
                       [4,5], [6,7],
                       [8,9]]
seeds = [0,100,200,300,400]

#%%
def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.compat.v1.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(initial)
    return W
#%%
def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.compat.v1.constant(0.1, shape=shape)
    b = tf.Variable(initial)
    return b
#%%
def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv
#%%
def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    return h_max

def avg_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    return h_max

#%%
class LeNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()

        self.W_conv1 = weight_variable([3, 3, 3, 6])
        self.b_conv1 = bias_variable([6])
        self.W_conv2 = weight_variable([3, 3, 6, 16])
        self.b_conv2 = bias_variable([16])
        self.W_fc1 = weight_variable([1024, 120])
        self.b_fc1 = bias_variable([120])
        self.W_fc2 = weight_variable([120, 84])
        self.b_fc2 = bias_variable([84])
        self.W_fc3 = weight_variable([84, num_classes])
        self.b_fc3 = bias_variable([num_classes])

        self.vars = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, 
                     self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.W_fc3, self.b_fc3]


    def call(self, x, training = True):
        h_conv1 = tf.nn.relu(conv2d(x, self.W_conv1) + self.b_conv1)
        h_pool1 = avg_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = avg_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
        h_fc3 = tf.nn.softmax(tf.matmul(h_fc2, self.W_fc3) + self.b_fc3)

        return h_fc3

def cross_ent(logits, y):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    return tf.reduce_mean(losses)


def max_conf(logits):
    y = tf.argmax(logits, 1)
    y = tf.one_hot(y, dim)
    losses = -tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    return tf.reduce_mean(losses)

def gen_adv(x):
    eps = 0.025
    n_iters = 20
    step_size = 0.02

    unif = tf.random.uniform(minval=-eps, maxval=eps, shape=tf.shape(x))
    x_adv = x + unif #tf.clip_by_value(x + unif, 0., 1.)
    
    for i in range(n_iters):
        x_adv = tf.Variable(x_adv)
        with tf.GradientTape() as tape:
            loss = max_conf(cnn(x_adv))
            grad = tape.gradient(loss, x_adv)
            g = tf.sign(grad)

        # import pdb;pdb.set_trace()
        x_adv_start = x_adv + step_size*g
        #x_adv = tf.clip_by_value(x_adv, 0., 1.)
        delta = x_adv - x_adv_start
        delta = tf.clip_by_value(delta, -eps, eps)
        x_adv = x_adv_start + delta

    return x_adv

# %%
### preprocess the data ###
def get_data(classes):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    input_shape = x_train.shape[1:]

    train_idx = np.where(y_train==classes[0])[0]
    test_idx = np.where(y_test==classes[0])[0]

    for ii in classes[1:]:
        train_idx = np.concatenate((
                        train_idx,
                        np.where(y_train==ii)[0]
                    ))
        test_idx = np.concatenate((
                        test_idx,
                        np.where(y_test==ii)[0]
                    ))

    x_train, y_train = x_train[train_idx], y_train[train_idx]
    x_test, y_test = x_test[test_idx], y_test[test_idx]
    
    _, y_train = np.unique(y_train, return_inverse=True)
    _, y_test = np.unique(y_test, return_inverse=True)
    
    if normalize:
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
    else:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
    
    x_train_mean = np.mean(x_train, axis=0)
    if subtract_pixel_mean:
        x_train -= x_train_mean
        x_test -= x_train_mean
    return (x_train, y_train), (x_test, y_test), x_train_mean
# %%
def experiment(task, sample_size, seed=0):
    random.seed(seed)
    classes = classes_to_consider[task]
    (x_train, y_train), (x_test, y_test), _ = get_data(classes)
    total_sample = x_train.shape[0]
    idx_to_train = random.sample(
            range(total_sample), 
            sample_size
        )
    input_shape = x_train[0].shape

    nn = getLeNet(
            input_shape, 
            num_classes=len(np.unique(y_train))
        )
    nn.compile(**compile_kwargs)
    history = nn.fit(
        x_train[idx_to_train],
        keras.utils.to_categorical(
            y_train[idx_to_train]
            ), 
        **fit_kwargs
        )
    
    predicted_label_dn = np.argmax(
            nn.predict(x_test), 
            axis=1
        )
    print('Trained model with classes ', classes, ' seed ', seed)
    print('Accuracy:', np.mean(predicted_label_dn==y_test.reshape(-1)))

    # train KGN model
    model_kdn = kdn(network=nn)
    model_kdn.fit(x_train[idx_to_train], y_train[idx_to_train], batch=5)
    model_kdn.global_bias = -1e9 - np.log10(sample_size)

    proba_kdn_geodesic = model_kdn.predict_proba(x_test, distance='Geodesic')
    proba_kdn_euclidean = model_kdn.predict_proba(x_test)
    proba_dn = model_kdn.network.predict(x_test)

    predicted_label_kdn_geodesic = np.argmax(proba_kdn_geodesic, axis = 1)
    predicted_label_kdn_euclidean = np.argmax(proba_kdn_euclidean, axis = 1)
    predicted_label_dn = np.argmax(proba_dn, axis = 1)

    acc_kdn_geodesic = np.mean(predicted_label_kdn_geodesic == y_test)
    acc_kdn_euclidean = np.mean(predicted_label_kdn_euclidean == y_test)
    acc_dn = np.mean(predicted_label_dn == y_test)

    print('Accuracy KDN-geodesic:', acc_kdn_geodesic)
    print('Accuracy KDN-euclidean:', acc_kdn_euclidean)


    ECE_kdn_geodesic = get_ece(proba_kdn_geodesic, y_test)
    ECE_kdn_euclidean = get_ece(proba_kdn_euclidean, y_test)
    ECE_dn = get_ece(proba_dn, y_test)

    print('ECE KDN-geodesic:', ECE_kdn_geodesic)
    print('ECE KDN-euclidean:', ECE_kdn_euclidean)
    print('ECE DN:', ECE_dn)

    return 1-acc_kdn_geodesic, ECE_kdn_geodesic,\
           1-acc_kdn_euclidean, ECE_kdn_euclidean,\
           1-acc_dn, ECE_dn


def experiment_out(task, seed, n_test=1000):
    random.seed(seed)
    classes = classes_to_consider[task]
    (x_train, y_train), (_, _), _ = get_data(classes)

    x_train = x_train.reshape(-1,32*32*3)
    normalize = np.max(np.linalg.norm(
                                    x_train,
                                    2,
                                    axis=1
                                )
                        )
    x_train = x_train/normalize
    x_train = x_train.reshape(-1,32,32,3)
    input_shape = x_train[0].shape

    nn = getLeNet(
            input_shape, 
            num_classes=len(np.unique(y_train))
        )
    nn.compile(**compile_kwargs)
    history = nn.fit(
        x_train,
        keras.utils.to_categorical(
            y_train
            ), 
        **fit_kwargs
        )
    
    print('Trained model with classes ', classes, ' seed ', seed)

    # train KGN model
    model_kdn = kdn(network=nn)
    model_kdn.fit(x_train, y_train, batch=5)
    model_kdn.global_bias = -5e6 - np.log10(x_train.shape[0])

    mmcOut_dn = []
    mmcOut_kdn_geod = []
    mmcOut_kdn_euc = []
    for r in np.arange(0,20.5,1):
        X_ood = sample_unifrom_circle(n=n_test, r=r, p=32*32*3)
        X_ood = X_ood.reshape(-1,32,32,3)
        mmcOut_dn.append(
            np.mean(np.max(model_kdn.network.predict(X_ood), axis=1))
        )
        mmcOut_kdn_geod.append(
            np.mean(np.max(model_kdn.predict_proba(X_ood, distance='Geodesic'), axis=1))
            )
        mmcOut_kdn_euc.append(
            np.mean(np.max(model_kdn.predict_proba(X_ood), axis=1))
        )

    return mmcOut_kdn_geod, mmcOut_kdn_euc, mmcOut_dn
# %%
train_samples = [10, 100, 1000, 10000]

for task in range(5):
    df = pd.DataFrame()
    err_kdn_geod_med = []
    err_kdn_euc_med = []
    err_dn_med = []
    err_kdn_geod_25 = []
    err_kdn_geod_75 = []
    err_kdn_euc_25 = []
    err_kdn_euc_75 = []
    err_dn_25 = []
    err_dn_75 = []

    ece_kdn_geod_med = []
    ece_kdn_euc_med = []
    ece_dn_med = []
    ece_kdn_geod_25 = []
    ece_kdn_geod_75 = []
    ece_kdn_euc_25 = []
    ece_kdn_euc_75 = []
    ece_dn_25 = []
    ece_dn_75 = []

    samples = []
    for sample in train_samples:
        samples.append(sample)

        res = Parallel(n_jobs=-1)(
                delayed(experiment)(
                        task,
                        sample,
                        seed=seed
                        ) for seed in seeds
                    )
        err_kdn_geod = []
        err_kdn_euc = []
        err_dn = []
        ece_kdn_geod = []
        ece_kdn_euc = []
        ece_dn = []
        for ii, _ in enumerate(seeds):
            err_kdn_geod.append(
                res[ii][0]
            )
            ece_kdn_geod.append(
                res[ii][1]
            )

            err_kdn_euc.append(
                res[ii][2]
            )
            ece_kdn_euc.append(
                res[ii][3]
            )

            err_dn.append(
                res[ii][4]
            )
            ece_dn.append(
                res[ii][5]
            )
        ##############################
        err_kdn_geod_med.append(
            np.median(
                err_kdn_geod
            )
        )
        err_kdn_euc_med.append(
            np.median(
                err_kdn_euc
            )
        )
        err_dn_med.append(
            np.median(
                err_dn
            )
        )
        
        err_kdn_geod_25.append(
            np.quantile(err_kdn_geod, [0.25])[0]
        )
        err_kdn_geod_75.append(
            np.quantile(err_kdn_geod, [0.75])[0]
        )
        err_kdn_euc_25.append(
            np.quantile(err_kdn_euc, [0.25])[0]
        )
        err_kdn_euc_75.append(
            np.quantile(err_kdn_euc, [0.75])[0]
        )
        err_dn_25.append(
            np.quantile(err_dn, [0.25])[0]
        )
        err_dn_75.append(
            np.quantile(err_dn, [0.75])[0]
        )
        
        #####################################
        ece_kdn_geod_med.append(
            np.median(
                ece_kdn_geod
            )
        )
        ece_kdn_euc_med.append(
            np.median(
                ece_kdn_euc
            )
        )
        ece_dn_med.append(
            np.median(
                ece_dn
            )
        )
        
        ece_kdn_geod_25.append(
            np.quantile(ece_kdn_geod, [0.25])[0]
        )
        ece_kdn_geod_75.append(
            np.quantile(ece_kdn_geod, [0.75])[0]
        )
        ece_kdn_euc_25.append(
            np.quantile(ece_kdn_euc, [0.25])[0]
        )
        ece_kdn_euc_75.append(
            np.quantile(ece_kdn_euc, [0.75])[0]
        )
        ece_dn_25.append(
            np.quantile(ece_dn, [0.25])[0]
        )
        ece_dn_75.append(
            np.quantile(ece_dn, [0.75])[0]
        )
    df['err_kdn_geod_med'] = err_kdn_geod_med
    df['err_kdn_euc_med'] = err_kdn_euc_med
    df['err_dn_med'] = err_dn_med
    df['err_kdn_geod_25'] = err_kdn_geod_25
    df['err_kdn_euc_25'] = err_kdn_euc_25
    df['err_dn_25'] = err_dn_25
    df['err_kdn_geod_75'] = err_kdn_geod_75
    df['err_kdn_euc_75'] = err_kdn_euc_75
    df['err_dn_75'] = err_dn_75

    df['ece_kdn_geod_med'] = ece_kdn_geod_med
    df['ece_kdn_euc_med'] = ece_kdn_euc_med
    df['ece_dn_med'] = ece_dn_med
    df['ece_kdn_geod_25'] = ece_kdn_geod_25
    df['ece_kdn_euc_25'] = ece_kdn_euc_25
    df['ece_dn_25'] = ece_dn_25
    df['ece_kdn_geod_75'] = ece_kdn_geod_75
    df['ece_kdn_euc_75'] = ece_kdn_euc_75
    df['ece_dn_75'] = ece_dn_75

    with open('results/Task'+str(task+1)+'.pickle', 'wb') as f:
        pickle.dump(df, f)


#%%
'''for task in range(5):
    df = pd.DataFrame()
    mmcOut_kdn_geod = []
    mmcOut_kdn_euc = []
    mmcOut_dn = []

    res = Parallel(n_jobs=-1)(
                delayed(experiment_out)(
                        task,
                        seed=seed
                        ) for seed in seeds
                )
    print(res)
    for ii, _ in enumerate(seeds):
        mmcOut_kdn_geod.append(
            res[ii][0]
        )
        mmcOut_kdn_euc.append(
            res[ii][1]
        )
        mmcOut_dn.append(
            res[ii][2]
        )
    
    mmcOut_kdn_geod_med =\
        np.median(mmcOut_kdn_geod,axis=0)
    mmcOut_kdn_euc_med =\
        np.median(mmcOut_kdn_euc,axis=0)
    mmcOut_dn_med =\
        np.median(mmcOut_dn,axis=0)

    mmcOut_kdn_geod_25 =\
        np.quantile(mmcOut_kdn_geod, [0.25], axis=0)[0]
    mmcOut_kdn_euc_25 =\
        np.quantile(mmcOut_kdn_euc, [0.25], axis=0)[0]
    mmcOut_dn_25 =\
        np.quantile(mmcOut_dn, [0.25], axis=0)[0]

    mmcOut_kdn_geod_75 =\
        np.quantile(mmcOut_kdn_geod, [0.75], axis=0)[0]
    mmcOut_kdn_euc_75 =\
        np.quantile(mmcOut_kdn_euc, [0.75], axis=0)[0]
    mmcOut_dn_75 =\
        np.quantile(mmcOut_dn, [0.75], axis=0)[0]

    df['mmcOut_kdn_geod_med'] = mmcOut_kdn_geod_med
    df['mmcOut_kdn_euc_med'] = mmcOut_kdn_euc_med
    df['mmcOut_dn_med'] = mmcOut_dn_med
    df['mmcOut_kdn_geod_25'] = mmcOut_kdn_geod_25
    df['mmcOut_kdn_euc_25'] = mmcOut_kdn_euc_25
    df['mmcOut_dn_25'] = mmcOut_dn_25
    df['mmcOut_kdn_geod_75'] = mmcOut_kdn_geod_75
    df['mmcOut_kdn_euc_75'] = mmcOut_kdn_euc_75
    df['mmcOut_dn_75'] = mmcOut_dn_75

    with open('results/Task'+str(task+1)+'_OOD.pickle', 'wb') as f:
        pickle.dump(df, f)'''
# %%
