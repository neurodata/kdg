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
from vit_keras import vit, utils
#%% load OOD data
ood_set = np.load('/Users/jayantadey/kdg/benchmarks/300K_random_images.npy')

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
    with tf.GradientTape(persistent=True) as tape:
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
seed = 0
input_shape = (32, 32, 3) #Cifar10 image size
image_size = 256 #size after resizing image
num_classes = 10

#%%
np.random.seed(seed)

nn_file = '/Users/jayantadey/kdg/benchmarks/cifar10_experiments/vit_model_'+str(seed)+'.keras'
model_to_copy = keras.models.load_model(nn_file)
model = build_model()

for layer_id, layer in enumerate(model.layers):
        pretrained_weights = model_to_copy.layers[layer_id].get_weights()
        layer.set_weights(pretrained_weights)

#%%
# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')/255.0 
x_test = x_test.astype('float32') /255.0
ood_set = ood_set.astype('float32')/255.0

x_train, x_cal, y_train, y_cal = train_test_split(
                x_train, y_train, train_size=0.95, random_state=seed, shuffle=True)

y_train = y_train.ravel()
y_test = y_test.ravel()
# Input image dimensions.
input_shape = x_train.shape

#%%
model.summary()

perm = np.arange(50000)
np.random.shuffle(perm)
idx = perm[:2500]
#%%
fpr = 1
eps_to_try = np.linspace(0,.04,21)
chosen_eps = eps_to_try[0]
T_ = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
chosen_T = T_[0]


for T in T_:
    for eps in eps_to_try:
        print('Doing ', eps)

        conf_in = []
        conf_out = []
        for ii in tqdm(range(100)):
            conf_in.append(
                 np.max(tf.nn.softmax(model(gen_adv(x_cal[ii*25:(ii+1)*25],eps,T))/T), axis=1)
                )
        
        #print(conf_in)
        for ii in tqdm(range(100)):
            conf_out.append(
                 np.max(tf.nn.softmax(model(ood_set[idx[ii*25:(ii+1)*25]])/T), axis=1)
            )

        conf_in = np.concatenate(
             conf_in
        )
        conf_out = np.concatenate(
             conf_out
        )
        f = fpr_at_95_tpr(conf_in, conf_out)
        print(f)
        if fpr > f:
            fpr=f
            chosen_eps = eps
            chosen_T = T
        else:
             break

print('Chosen eps ', chosen_eps, 'Chosen T ', chosen_T)


# %%
