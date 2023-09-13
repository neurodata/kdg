# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle
from keras.models import Model
from kdg import kdcnn, kdf, kdn
import pickle
from tensorflow.keras.datasets import cifar10, cifar100
import timeit
from scipy.io import loadmat
import random
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
x_svhn = loadmat('/Users/jayantadey/svhn/train_32x32.mat')['X']
test_ids =  random.sample(range(0, x_svhn.shape[3]), 1000)
x_svhn = x_svhn[:,:,:,test_ids].astype('float32').reshape(1000,32,32,3)

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_cifar100 = x_cifar100.astype('float32') / 255
x_svhn = x_svhn.astype('float32') / 255

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean
x_cifar100 -= x_train_mean
x_svhn -= x_train_mean
#%% Load model file
seed = 400
filename = 'resnet_kdn_50000_'+str(seed)+'.pickle'

with open(filename, 'rb') as f:
    model_kdn = pickle.load(f)

model_kdn.global_bias = -8e3

proba_in = model_kdn.predict_proba(x_test, distance='Geodesic')
proba_cifar100 = model_kdn.predict_proba(x_cifar100, distance='Geodesic')
proba_svhn = model_kdn.predict_proba(x_svhn, distance='Geodesic')

proba_in_dn = model_kdn.network.predict(x_test)
proba_cifar100_dn = model_kdn.network.predict(x_cifar100)
proba_svhn_dn = model_kdn.network.predict(x_svhn)

summary = (proba_in, proba_cifar100, proba_svhn, proba_in_dn, proba_cifar100_dn, proba_svhn_dn)
file_to_save = 'results/resnet20_'+str(seed)+'.pickle'

with open(file_to_save, 'wb') as f:
    pickle.dump(summary, f)
# %%
