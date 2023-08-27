#%%
import tensorflow as tf
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
tf.compat.v1.enable_eager_execution()
from kdg.utils import generate_gaussian_parity, generate_ood_samples, generate_spirals, generate_ellipse, get_ece, sample_unifrom_circle
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from scipy.io import loadmat
import random
from joblib import Parallel, delayed
import pandas as pd
# %%
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
    y = tf.one_hot(y, 2)
    losses = -tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    return tf.reduce_mean(losses)

def gen_adv(x, cnn):
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
### Hyperparameters ###
subtract_pixel_mean = False
normalize = True
classes_to_consider = [[0,1], [2,3],
                       [4,5], [6,7],
                       [8,9]]
seeds = [0,100,200,300,400]

compile_kwargs = {
        "loss": "binary_crossentropy",
        "optimizer": keras.optimizers.Adam(3e-4),
    }
callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10, verbose=True)
fit_kwargs = {
        "epochs": 100,
        "batch_size": 32,
        "verbose": False,
        "callbacks": [callback],
    }
#%%
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
def experiment(task, sample_size, batchsize=20, epochs=20, seed=0):

    if sample_size<batchsize:
        batchsize = sample_size

    random.seed(seed)
    classes = classes_to_consider[task]
    (x_train, y_train), (x_test, y_test), _ = get_data(classes)
    total_sample = x_train.shape[0]
    idx_to_train = random.sample(
            range(total_sample), 
            sample_size
        )
    x_train, y_train = x_train[idx_to_train], y_train[idx_to_train]
    total_sample = x_train.shape[0]
    input_shape = x_train[0].shape
    iteration = total_sample//batchsize

    cnn = LeNet(num_classes=2)
    optimizer = tf.optimizers.Adam(3e-4) 

    for i in range(1, epochs+1):
        perm = np.arange(total_sample)
        np.random.shuffle(perm)
        perm = perm.reshape(-1,batchsize)

        for j in range(iteration):
            x_train_ = x_train[perm[j]]
            y_train_ = tf.one_hot(y_train[perm[j]], depth=2)
            X_noise = tf.random.uniform([2*x_train_.shape[0], x_train_.shape[1], x_train_.shape[2], x_train_.shape[3]],minval=-1,maxval=1)
            
            X_noise = gen_adv(X_noise, cnn)
            with tf.GradientTape() as tape:
                logits = cnn(x_train_)
                logits_noise = cnn(X_noise)
                loss_main = cross_ent(logits, y_train_)
                loss_acet = max_conf(logits_noise)
                loss = loss_main + loss_acet

            grads = tape.gradient(loss, cnn.vars)
            optimizer.apply_gradients(zip(grads, cnn.vars))

        train_err = np.mean(logits.numpy().argmax(1) != y_train_.numpy().argmax(1))
        print("Epoch {:03d}: loss_main={:.3f} loss_acet={:.3f} err={:.2%}".format(i, loss_main, loss_acet, train_err))

    predicted_proba = cnn.predict(x_test)
    predicted_label = np.argmax(
            predicted_proba, 
            axis=1
        )
    print('Trained model with classes ', classes, ' seed ', seed)
    print('Accuracy:', np.mean(predicted_label==y_test.reshape(-1)))

    err = 1 - np.mean(predicted_label==y_test.reshape(-1))
    ECE = get_ece(predicted_proba, y_test.reshape(-1))
    

    print('ECE:', ECE)

    return err, ECE


def experiment_out(task, seed, batchsize=20, epochs=20, n_test=1000):
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
    total_sample = x_train.shape[0]
    iteration = total_sample//batchsize

    cnn = LeNet(num_classes=2)
    optimizer = tf.optimizers.Adam(3e-4)
    
    for i in range(1, epochs+1):
        perm = np.arange(total_sample)
        np.random.shuffle(perm)
        perm = perm.reshape(-1,batchsize)

        for j in range(iteration):
            x_train_ = x_train[perm[j]]
            y_train_ = tf.one_hot(y_train[perm[j]], depth=2)
            X_noise = tf.random.uniform([2*x_train_.shape[0], x_train_.shape[1], x_train_.shape[2], x_train_.shape[3]],minval=-1,maxval=1)
            
            X_noise = gen_adv(X_noise, cnn)
            with tf.GradientTape() as tape:
                logits = cnn(x_train_)
                logits_noise = cnn(X_noise)
                loss_main = cross_ent(logits, y_train_)
                loss_acet = max_conf(logits_noise)
                loss = loss_main + loss_acet

            grads = tape.gradient(loss, cnn.vars)
            optimizer.apply_gradients(zip(grads, cnn.vars))

        train_err = np.mean(logits.numpy().argmax(1) != y_train_.numpy().argmax(1))
        print("Epoch {:03d}: loss_main={:.3f} loss_acet={:.3f} err={:.2%}".format(i, loss_main, loss_acet, train_err))


    print('Trained model with classes ', classes, ' seed ', seed)

    mmcOut = []
    for r in np.arange(0,20.5,1):
        X_ood = sample_unifrom_circle(n=n_test, r=r, p=32*32*3)
        X_ood = X_ood.reshape(-1,32,32,3)
        mmcOut.append(
            np.mean(np.max(cnn.predict(X_ood), axis=1))
        )

    return mmcOut
# %%
'''train_samples = [10, 100, 1000, 10000]

for task in range(5):
    df = pd.DataFrame()
    err_med = []
    err_25 = []
    err_75 = []

    ece_med = []
    ece_25 = []
    ece_75 = []

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
        err = []
        ece = []
        for ii, _ in enumerate(seeds):
            err.append(
                res[ii][0]
            )
            ece.append(
                res[ii][1]
            )

        ##############################
        err_med.append(
            np.median(
                err
            )
        )   
        err_25.append(
            np.quantile(err, [0.25])[0]
        )
        err_75.append(
            np.quantile(err, [0.75])[0]
        )
        
        ece_med.append(
            np.median(
                ece
            )
        )
        ece_25.append(
            np.quantile(ece, [0.25])[0]
        )
        ece_75.append(
            np.quantile(ece, [0.75])[0]
        )
        
        
    df['err_med'] = err_med
    df['err_25'] = err_25
    df['err_75'] = err_75
   
    df['ece_med'] = ece_med
    df['ece_25'] = ece_25
    df['ece_75'] = ece_75

    with open('results/ACET_Task'+str(task+1)+'.pickle', 'wb') as f:
        pickle.dump(df, f)'''


#%%
for task in range(5):
    df = pd.DataFrame()

    res = Parallel(n_jobs=-1)(
                delayed(experiment_out)(
                        task,
                        seed=seed
                        ) for seed in seeds
                )
    print(res)
        
    
    mmcOut_med =\
        np.median(res,axis=0)

    mmcOut_25 =\
        np.quantile(res, [0.25], axis=0)[0]
    
    mmcOut_75 =\
        np.quantile(res, [0.75], axis=0)[0]
   
    df['mmcOut_med'] = mmcOut_med
    df['mmcOut_25'] = mmcOut_25
    df['mmcOut_75'] = mmcOut_75

    with open('results/ACET_Task'+str(task+1)+'_OOD.pickle', 'wb') as f:
        pickle.dump(df, f)
# %%