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
from vit_keras import vit, utils
#%%
num_classes = 10
input_shape = (32,32,3)
image_size = 256 #size after resizing image
seeds = [0, 1, 2, 3, 2022]
#%%
# Load the CIFAR10 and CIFAR100 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(_, _), (x_cifar100, y_cifar100) = cifar100.load_data()
x_noise = np.random.random_integers(0,high=255,size=(1000,32,32,3)).astype('float32')/255.0

#x_svhn = loadmat('/Users/jayantadey/DF-CNN/data_five/SVHN/test_32x32.mat')['X']
#y_svhn = loadmat('/Users/jayantadey/DF-CNN/data_five/SVHN/test_32x32.mat')['y']
x_svhn = loadmat('/cis/home/jdey4/test_32x32.mat')['X']
#y_svhn = loadmat('/cis/home/jdey4/train_32x32.mat')['y']
#test_ids =  random.sample(range(0, x_svhn.shape[3]), 2000)
x_svhn = x_svhn.astype('float32')
x_tmp = np.zeros((x_svhn.shape[3],32,32,3), dtype=float)

for ii in range(x_svhn.shape[3]):
    x_tmp[ii,:,:,:] = x_svhn[:,:,:,ii]

x_svhn = x_tmp
del x_tmp

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32')/255.0 
x_test = x_test.astype('float32')/255.0 
x_cifar100 = x_cifar100.astype('float32')/255.0 
x_svhn = x_svhn.astype('float32')/255.0 


#%% Load model file
input_shape = x_train.shape

for seed in seeds: 
    print('doing seed ',seed)

    #load pretrained model
    #pretrained_model = keras.models.load_model('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/resnet20_models/cifar100_model_new_'+str(seed))
    model = keras.models.load_model('OE_vit_'+str(seed))

    proba_in = model.predict(x_test)
    proba_cifar100 = model.predict(x_cifar100),
    proba_svhn = model.predict(x_svhn)
    proba_noise = model.predict(x_noise),

    summary = (proba_in, proba_cifar100, proba_svhn, proba_noise)
    file_to_save = 'OE_vit_'+str(seed)+'.pickle'

    with open(file_to_save, 'wb') as f:
        pickle.dump(summary, f)

# %%
