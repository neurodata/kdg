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
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.applications.resnet50 import ResNet50 
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers
#from keras import ops
from tensorflow.keras.datasets import cifar10, cifar100
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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
input_shape = (32, 32, 3)
batchsize = 128 # orig paper trained all networks with batch_size=128
num_classes = 10
seeds = [0, 100, 200, 300, 400]
T = [1.0, 1.0] #cifar10 params
eps = [0.032, 0.028]
'''T = 5.0 #cifar100 params
eps = 0.026'''


#%%
data_augmentation = keras.Sequential(
    [
        layers.Normalization()
    ]
)
#%%
model = keras.Sequential()
base_model = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )

inputs = keras.Input(shape=input_shape)
model.add(inputs)
model.add(data_augmentation)
model.add(base_model)
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(
            Dense(
                    num_classes
                )
        )

model.build()
#%%
# Load the CIFAR10 and CIFAR100 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(_, _), (x_cifar100, y_cifar100) = cifar100.load_data()
x_noise = np.random.random_integers(0,high=255,size=(1000,32,32,3)).astype('float32')/255.0
x_svhn = loadmat('/Users/jayantadey/DF-CNN/data_five/SVHN/train_32x32.mat')['X']
y_svhn = loadmat('/Users/jayantadey/DF-CNN/data_five/SVHN/train_32x32.mat')['y']
#test_ids =  random.sample(range(0, x_svhn.shape[3]), 2000)
x_svhn = x_svhn.astype('float32')
x_tmp = np.zeros((x_svhn.shape[0],32,32,3), dtype=float)

for ii in range(x_svhn.shape[0]):
    x_tmp[ii,:,:,:] = x_svhn[:,:,:,ii]

x_svhn = x_tmp
del x_tmp
# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')
x_cifar100 = x_cifar100.astype('float32') 
x_svhn = x_svhn.astype('float32')

#%%
x_train = gen_adv(x_train, eps, T)
x_test = gen_adv(x_test, eps, T)
x_cifar100 = gen_adv(x_cifar100, eps, T)
x_svhn = gen_adv(x_svhn, eps, T)
x_noise = gen_adv(x_noise, eps, T)
#%% Load model file
input_shape = x_train.shape

for seed in seeds: 
    print('doing seed ',seed)

    #load pretrained model
    #pretrained_model = keras.models.load_model('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/resnet20_models/cifar100_model_new_'+str(seed))
    pretrained_model = keras.models.load_model('resnet20_models/cifar_finetune10_'+str(seed))
    for layer_id, layer in enumerate(model.layers):
        pretrained_weights = pretrained_model.layers[layer_id].get_weights()
        layer.set_weights(pretrained_weights)
        layer.trainable = False
    
    proba_in = predict_proba(model, x_test, T) 
    proba_cifar100 = predict_proba(model, x_cifar100, T)
    proba_svhn = predict_proba(model, x_svhn, T)
    proba_noise = predict_proba(model, x_noise, T)

    summary = (proba_in, proba_cifar100, proba_svhn, proba_noise)
    file_to_save = 'resnet20_cifar10_ODIN_'+str(seed)+'.pickle'

    with open(file_to_save, 'wb') as f:
        pickle.dump(summary, f)

# %%
