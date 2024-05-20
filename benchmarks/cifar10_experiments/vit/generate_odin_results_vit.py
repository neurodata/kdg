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
from vit_keras import vit, utils
#%%
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
    outputs = Dense(num_classes)(x)

    model_final = Model(inputs=inputs, outputs=outputs)
    return model_final


def predict_proba(model, x, T):
    logits = model.predict(x)/T
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
input_shape = (32, 32, 3) #Cifar10 image size
image_size = 256 #size after resizing image
num_classes = 10
T = 1.0 #cifar100 params
eps = 0.006

#%%
# Load the CIFAR10 and CIFAR100 data.
(_, _), (x_test, y_test) = cifar10.load_data()
(_, _), (x_cifar100, y_cifar100) = cifar100.load_data()
x_noise_ = np.random.random_integers(0,high=255,size=(1000,32,32,3)).astype('float32')/255.0
x_svhn = loadmat('/Users/jayantadey/DF-CNN/data_five/SVHN/test_32x32.mat')['X']
y_svhn = loadmat('/Users/jayantadey/DF-CNN/data_five/SVHN/test_32x32.mat')['y']
#test_ids =  random.sample(range(0, x_svhn.shape[3]), 2000)
x_svhn = x_svhn.astype('float32')
x_tmp = np.zeros((x_svhn.shape[3],32,32,3), dtype=float)

for ii in range(x_svhn.shape[3]):
    x_tmp[ii,:,:,:] = x_svhn[:,:,:,ii]

x_svhn = x_tmp
del x_tmp

x_test_ = x_test.astype('float32')/255.0
x_cifar100_ = x_cifar100.astype('float32')/255.0 
x_svhn_ = x_svhn.astype('float32')/255.0

def load_adv(x, eps, T):
    total_len = x.shape[0]
    x_ = []
    batchsize=50
    total_batch=total_len//batchsize
    print(total_batch)
    for ii in tqdm(range(total_batch)):
        x_.append(
            gen_adv(x[ii*batchsize:(ii+1)*batchsize], eps, T)
        )
    
    if (ii+1)*batchsize<total_len-1:
        x_.append(
            gen_adv(x[(ii+1)*batchsize:], eps, T)
        )
    
    return np.concatenate(
        x_, axis=0
    )

#%% Load model file
seeds = [2,3,2022]

for ii, seed in enumerate(seeds): 
    print('doing seed ',seed)
    np.random.seed(seed)

    nn_file = '/Users/jayantadey/kdg/benchmarks/cifar10_experiments/vit_model_'+str(seed)+'.keras'
    model_to_copy = keras.models.load_model(nn_file)
    model = build_model()

    for layer_id, layer in enumerate(model.layers):
            pretrained_weights = model_to_copy.layers[layer_id].get_weights()
            layer.set_weights(pretrained_weights)


    x_svhn = load_adv(x_svhn_, eps, T)
    x_test = load_adv(x_test_, eps, T)
    x_cifar100 = load_adv(x_cifar100_, eps, T)
    x_noise = load_adv(x_noise_, eps, T)
    
    
    proba_in = predict_proba(model, x_test, T) 
    proba_cifar100 = predict_proba(model, x_cifar100, T)
    proba_svhn = predict_proba(model, x_svhn, T)
    proba_noise = predict_proba(model, x_noise, T)

    summary = (proba_in, proba_cifar100, proba_svhn, proba_noise)
    file_to_save = 'ODIN_vit_'+str(seed)+'.pickle'

    with open(file_to_save, 'wb') as f:
        pickle.dump(summary, f)

# %%
