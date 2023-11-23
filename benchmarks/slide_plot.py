#%%
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from kdg import kdn
from kdg.utils import generate_gaussian_parity, hellinger, plot_2dsim, generate_ood_samples, sample_unifrom_circle, get_ece
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed
from scipy.io import savemat, loadmat
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier as rf 
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import activations
from tensorflow.keras import backend as bknd
import timeit
# %%
X, y = generate_gaussian_parity(3000, cluster_std=.3)
plot_2dsim(X, y)
plt.xlim(-2,2)
plt.ylim(-2,2)
# %%
compile_kwargs = {
        "loss": "binary_crossentropy",
        "optimizer": keras.optimizers.Adam(3e-4),
    }
callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10, verbose=True)
fit_kwargs = {
        "epochs": 200,
        "batch_size": 64,
        "verbose": False,
        "callbacks": [callback],
    }
# %%
def getNN(input_size, num_classes):
    network_base = keras.Sequential()
    initializer = keras.initializers.random_normal(seed=0)
    network_base.add(keras.layers.Dense(1000, kernel_initializer=initializer, input_shape=(input_size,)))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(1000, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(1000, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(1000, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(units=num_classes, activation="softmax", kernel_initializer=initializer))
    network_base.compile(**compile_kwargs)
    return network_base
# %%
# train Vanilla NN
nn = getNN(input_size=2, num_classes=2)
history = nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)
# %%
X_test, y_test = generate_gaussian_parity(10000, cluster_std=.3)
conf_nn = nn.predict(X_test)
# %%
plt.hist(np.amax(conf_nn, axis=1), density=True)
# %%
rf_model = rf(n_estimators=10)
rf_model.fit(X, y)
# %%
conf_rf = rf_model.predict_proba(X_test)
# %%
plt.hist(np.amax(conf_rf, axis=1), density=True)
#%%
sns.set_context('talk')
fig, ax = plt.subplots(1,3, figsize=(24,8))
plot_2dsim(X,y,ax=ax[0])
ax[0].set_xlim(-2,2)
ax[0].set_ylim(-2,2)
ax[0].set_xticks([-2,-1,0,1,2])
ax[0].set_yticks([-2,-1,0,1,2])
ax[0].tick_params(labelsize=24)
ax[0].set_title('Simulation', fontsize=32)

ax[1].hist(np.amax(conf_rf, axis=1), density=True)
ax[1].tick_params(labelsize=24)
ax[1].set_ylabel('Frequency', fontsize=24)
ax[1].set_xlabel('Confidence', fontsize=24)
ax[1].set_title('Random Forest', fontsize=32)


ax[2].hist(np.amax(conf_nn, axis=1), density=True)
ax[2].tick_params(labelsize=24)
ax[2].set_ylabel('Frequency', fontsize=24)
ax[2].set_xlabel('Confidence', fontsize=24)
ax[2].set_title('Deep-net', fontsize=32)

plt.savefig('../plots/overconfident.png')
# %%
