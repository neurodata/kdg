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

#%%
classes_to_consider = [[0,1], [2,3],
                       [4,5], [6,7],
                       [8,9]]
seeds = [0,100,200,300,400]
done = [0,1,3]

for task, classes in enumerate(classes_to_consider):
    if task in done:
        continue

    subtract_pixel_mean = True
    normalize = True
    (x_train, y_train), (x_test, y_test), trn_mean = get_data(classes)
    input_shape = x_train.shape
    batchsize=40
    iteration = input_shape[0]//batchsize
    epochs = 10
    dim=2
    #y_train = tf.one_hot(y_train, depth=2)
    for seed in seeds:
        mmc_dn = {}
        acc_dn = 0
        
        print('Training model with classes ', classes, ' seed ', seed)
        
        np.random.seed(seed)
        cnn = LeNet(num_classes=2)
        # the default learning rate of Adam might not be the best for this dataset
        optimizer = tf.optimizers.Adam(3e-4) 

        # Training loop
        acet = True

        for i in range(1, epochs+1):
            perm = np.arange(input_shape[0])
            np.random.shuffle(perm)
            perm = perm.reshape(-1,batchsize)

            for j in range(iteration):
                x_train_ = x_train[perm[j]]
                y_train_ = tf.one_hot(y_train[perm[j]], depth=2)
                X_noise = tf.random.uniform([2*x_train_.shape[0], x_train_.shape[1], x_train_.shape[2], x_train_.shape[3]],minval=-1,maxval=1)
                if acet:
                    X_noise = gen_adv(X_noise)
                with tf.GradientTape() as tape:
                    logits = cnn(x_train_)
                    logits_noise = cnn(X_noise)
                    loss_main = cross_ent(logits, y_train_)
                    loss_acet = acet * max_conf(logits_noise)
                    loss = loss_main + loss_acet

                grads = tape.gradient(loss, cnn.vars)
                optimizer.apply_gradients(zip(grads, cnn.vars))

            train_err = np.mean(logits.numpy().argmax(1) != y_train_.numpy().argmax(1))
            print("Epoch {:03d}: loss_main={:.3f} loss_acet={:.3f} err={:.2%}".format(i, loss_main, loss_acet, train_err))
        
        for task_, classes_ in enumerate(classes_to_consider):
            subtract_pixel_mean = False
            normalize = True
            (_, _), (x_test, y_test), _ = get_data(classes)
            x_test -= trn_mean
            predicted_logits = cnn(x_test)
            mmc_dn['Task '+str(task_+1)] = np.mean(np.max(predicted_logits,axis=1))
        
            if task==task_:
                acc_dn = np.mean(predicted_logits.numpy().argmax(1) == y_test)
                print(acc_dn)
        
        (_, _), (x_test, y_test) = keras.datasets.cifar100.load_data()
        test_ids =  random.sample(range(0, x_test.shape[0]), 2000)
        x_test = x_test[test_ids].astype('float32')/255
        x_test -= trn_mean
        
        predicted_logits = cnn(x_test)
        mmc_dn['cifar100'] = np.mean(np.max(predicted_logits,axis=1))
        
        x_test = loadmat('/cis/home/jdey4/train_32x32.mat')['X']
        test_ids =  random.sample(range(0, x_test.shape[3]), 2000)
        x_test = x_test[:,:,:,test_ids].astype('float32').reshape(2000,32,32,3)/255
        x_test -= trn_mean
        
        predicted_logits = cnn(x_test)
        mmc_dn['svhn'] = np.mean(np.max(predicted_logits,axis=1))
        
        x_test = np.random.random_integers(0,high=255,size=(2000,32,32,3)).astype('float')
        x_test -= trn_mean
        predicted_logits = cnn(x_test)
        mmc_dn['noise'] = np.mean(np.max(predicted_logits,axis=1))
        
        summary = (mmc_dn, acc_dn)
        
        print(summary)
        with open('results/ACET_'+str(task)+'_'+str(seed)+'.pickle', 'wb') as f:
            pickle.dump(summary,f)