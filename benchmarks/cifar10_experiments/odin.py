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

#%% load OOD data
ood_set = np.load('/Users/jayantadey/kdg/benchmarks/300K_random_images.npy')

#%%
def fpr_at_95_tpr(conf_in, conf_out):
    TPR = 95
    PERC = np.percentile(conf_in, 100-TPR)
    #FP = np.sum(conf_out >=  PERC)
    FPR = np.sum(conf_out >=  PERC)/len(conf_out)
    return FPR

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
batchsize = 128 # orig paper trained all networks with batch_size=128
num_classes = 10
seed = 400
input_shape = (32, 32, 3)
#%% load pretrained model weights
print('loading weights')
with open('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/pretrained_weight_contrast.pickle', 'rb') as f:
    weights = pickle.load(f)

#%%
data_augmentation = keras.Sequential(
    [
        layers.Normalization()
    ]
)
#%%
np.random.seed(seed)

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
pretrained_model = keras.models.load_model('resnet20_models/cifar_finetune10_'+str(seed))
for layer_id, layer in enumerate(model.layers):
    pretrained_weights = pretrained_model.layers[layer_id].get_weights()
    layer.set_weights(pretrained_weights)
    layer.trainable = False
#%%
# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
ood_set = ood_set.astype('float32')

x_train, x_cal, y_train, y_cal = train_test_split(
                x_train, y_train, train_size=0.9, random_state=seed, stratify=y_train)
y_train = y_train.ravel()
y_test = y_test.ravel()
# Input image dimensions.
input_shape = x_train.shape

#%%
model.summary()

perm = np.arange(50000)
np.random.shuffle(perm)
idx = perm[:1000]
#%%
fpr = 1
eps_to_try = np.linspace(0,.04,21)
chosen_eps = eps_to_try[0]
T_ = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
chosen_T = T_[0]

for T in T_:
    for eps in eps_to_try:
        print('Doing ', eps)
        conf_in = np.max(tf.nn.softmax(model(gen_adv(x_cal,eps,T))/T), axis=1)
        #print(conf_in)
        conf_out = np.max(tf.nn.softmax(model(ood_set[idx])/T), axis=1)
        f = fpr_at_95_tpr(conf_in, conf_out)
        print(f)
        if fpr > f:
            fpr=f
            chosen_eps = eps
            chosen_T = T

print('Chosen eps ', chosen_eps, 'Chosen T ', chosen_T)


# %%
