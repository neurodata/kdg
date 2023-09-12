# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle
from keras.models import Model
from kdg import kdcnn, kdf, kdn
import pickle
from tensorflow.keras.datasets import cifar10
import timeit
import seaborn as sns
import matplotlib.pyplot as plt
#%%
# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean
# %%
for ii in range(0,20):
    with open('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/temp/temp'+str(ii)+'.pickle','rb') as f:
        print(ii)
        if ii==0:
            w_ = pickle.load(f)
        else:
            w_ *= pickle.load(f)
# %%
idx_to_visualize = 0
clr = sns.color_palette("tab10", 10)

arg = np.argsort(w_[idx_to_visualize])

plt.plot(w_[arg])
# %%
a = []
for jj in range(1000):
    w = []
    scale = w_[jj]**1
    for ii in range(2):
        idx = np.where(y_train[idx_to_train]==ii)[0]
        w.append(np.sum(scale[idx]))
    print(w)
    a.append(np.max(w/np.sum(w)))
print(np.mean(a))
# %%
for ii in range(5):
    with open('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/temp/temp'+str(ii)+'.pickle', 'rb') as f:
        if ii ==0:
            w_ = pickle.load(f)
        else:
            w_ *= pickle.load(f)