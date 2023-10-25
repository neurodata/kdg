# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle
from keras.models import Model
from kdg import kdcnn, kdf, kdn, get_ece
import pickle
from tensorflow.keras.datasets import cifar10, cifar100
import timeit
from scipy.io import loadmat
import random
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
#%%
def fpr_at_95_tpr(conf_in, conf_out):
    TPR = 95
    PERC = np.percentile(conf_in, 100-TPR)
    #FP = np.sum(conf_out >=  PERC)
    FPR = np.sum(conf_out >=  PERC)/len(conf_out)
    return FPR, PERC
#%%
# Load the CIFAR10 and CIFAR100 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(_, _), (x_cifar100, y_cifar100) = cifar100.load_data()
x_noise = np.random.random_integers(0,high=255,size=(1000,32,32,3)).astype('float')/255.0
x_svhn = loadmat('/cis/home/jdey4/train_32x32.mat')['X']
y_svhn = loadmat('/cis/home/jdey4/train_32x32.mat')['y']
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
x_cifar100 = x_cifar100.astype('float32') / 255
x_svhn = x_svhn.astype('float32') / 255


for channel in range(3):
    x_train_mean = np.mean(x_train[:,:,:,channel])
    x_train_std = np.std(x_train[:,:,:,channel])

    x_train[:,:,:,channel] -= x_train_mean
    x_train[:,:,:,channel] /= x_train_std

    x_test[:,:,:,channel] -= x_train_mean
    x_test[:,:,:,channel] /= x_train_std

    x_cifar100[:,:,:,channel] -= x_train_mean
    x_cifar100[:,:,:,channel] /= x_train_std

    x_svhn[:,:,:,channel] -= x_train_mean #+ 1
    x_svhn[:,:,:,channel] /= x_train_std

    x_noise[:,:,:,channel] -= x_train_mean
    x_noise[:,:,:,channel] /= x_train_std

#%%
#with open('resnet20_0.pickle', 'rb') as f:
#    (proba_in, proba_cifar100, proba_svhn, proba_noise, proba_in_dn, proba_cifar100_dn, proba_svhn_dn, proba_noise_dn, proba_in_acet, proba_cifar100_acet, proba_svhn_acet, proba_noise_acet) = pickle.load(f)

filename = '/cis/home/jdey4/kdg/benchmarks/cifar10_experiments/resnet_kdn_50000_new_0.joblib'
model_kdn = joblib.load(filename)

model_kdn.global_bias = -3e10

proba_in = model_kdn.predict_proba(x_test[:1000], distance='Geodesic', n_jobs=50)

label_y = np.argmax(proba_in,axis=1)
print(np.mean(label_y==y_test[:1000].ravel()))
print(get_ece(proba_in, y_test[:1000].ravel(), n_bins=15))