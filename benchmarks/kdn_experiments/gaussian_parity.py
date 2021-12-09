#
# Created on Thu Dec 09 2021 6:04:08 AM
# Author: Ashwin De Silva (ldesilv2@jhu.edu)
# Objective: Gaussian Parity Experiment
#

# import standard libraries
import numpy as np
from tensorflow import keras
from keras import layers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import internal libraries
from kdg.kdn import *
from kdg.utils import generate_gaussian_parity

# generate training data
X, y = generate_gaussian_parity(10000)
X_val, y_val = generate_gaussian_parity(500)

# NN params
compile_kwargs = {
    "loss": "binary_crossentropy",
    "optimizer": keras.optimizers.Adam(1e-3)
    }
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=True)
fit_kwargs = {
    "epochs": 200,
    "batch_size": 32,
    "verbose": True,
    "validation_data": (X_val, keras.utils.to_categorical(y_val)),
    "callbacks": [callback]
    }

# network architecture
def getNN():
    network_base = keras.Sequential()
    network_base.add(layers.Dense(5, activation='relu', input_shape=(2,)))
    network_base.add(layers.Dense(5, activation='relu'))
    network_base.add(layers.Dense(units=2, activation = 'softmax'))
    network_base.compile(**compile_kwargs)
    return network_base

# train Vanilla NN
vanilla_nn = getNN()
history = vanilla_nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)

# plot the training loss and validation loss 
fig, ax = plt.subplots()
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_xlabel('epochs')
ax.set_ylabel('loss')
ax.legend(['train', 'val'])

# print the accuracy of Vanilla NN and KDN
X_test, y_test = generate_gaussian_parity(1000)       
accuracy_nn = np.mean(
                np.argmax(vanilla_nn.predict(X_test), axis=1) == y_test
            )
print("Vanilla NN accuracy : ", accuracy_nn)

# train KDN
model_kdn = kdn(network=vanilla_nn, 
                k = 1e-6,
                polytope_compute_method='all', 
                weighting_method='lin',
                T=2,
                c=1,
                verbose=False)
model_kdn.fit(X, y)

# print the accuracy of Vanilla NN and KDN
accuracy_kdn = np.mean(
                model_kdn.predict(X_test) == y_test
                )
print("KDN accuracy : ", accuracy_kdn)

# plot

# define the grid
p = np.arange(-4,4,step=0.005)
q = np.arange(-4,4,step=0.005)
xx, yy = np.meshgrid(p,q)
tmp = np.ones(xx.shape)
grid_samples = np.concatenate(
            (
                xx.reshape(-1,1),
                yy.reshape(-1,1)
            ),
            axis=1
        )    

# plot
proba_kdn = model_kdn.predict_proba(grid_samples)
proba_nn = model_kdn.predict_proba_nn(grid_samples)

fig, ax = plt.subplots(1, 3, figsize=(40, 40))

colors = sns.color_palette("Dark2", n_colors=2)
clr = [colors[i] for i in y]
ax[0].scatter(X[:, 0], X[:, 1], c=clr, s=20)
ax[0].set_xlim(-2, 2)
ax[0].set_ylim(-2, 2)
ax[0].set_title('Data')
ax[0].set_aspect('equal')

ax1 = ax[1].imshow(proba_nn[:,0].reshape(1600, 1600).T, extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap='bwr', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
ax[1].set_title('NN')
ax[1].set_aspect('equal')
fig.colorbar(ax1, ax=ax[1], fraction=0.046, pad=0.04)

ax2 = ax[2].imshow(proba_kdn[:,0].reshape(1600, 1600).T, extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap='bwr', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
ax[2].set_title('KDN')
ax[2].set_aspect('equal')
fig.colorbar(ax2, ax=ax[2], fraction=0.046, pad=0.04)

fig.savefig('plots/gaussian_parity.pdf')
plt.show()
