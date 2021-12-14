# import modules
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from kdg.kdn import *
from kdg.utils import gaussian_sparse_parity, trunk_sim
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt

# define the experimental setup
p = 20 # total dimensions of the data vector
p_star = 3 # number of signal dimensions of the data vector
'''sample_size = np.logspace(
        np.log10(10),
        np.log10(5000),
        num=10,
        endpoint=True,
        dtype=int
        )'''

sample_size = [1000, 5000, 10000] # sample size under consideration
n_test = 1000 # test set size
reps = 10 # number of replicates

df = pd.DataFrame()
reps_list = []
accuracy_kdn = []
accuracy_kdn_ = []
accuracy_nn = []
accuracy_nn_ = []
sample_list = []

# NN params
compile_kwargs = {
    "loss": "binary_crossentropy",
    "optimizer": keras.optimizers.Adam(3e-4)
    }
fit_kwargs = {
    "epochs": 150,
    "batch_size": 32,
    "verbose": False
    }

#%%

# network architecture
def getNN():
    network_base = keras.Sequential()
    network_base.add(layers.Dense(10, activation='relu', input_shape=(20,)))
    network_base.add(layers.Dense(5, activation='relu'))
    network_base.add(layers.Dense(units=2, activation = 'softmax'))
    network_base.compile(**compile_kwargs)
    return network_base

# %%
for sample in sample_size:
    print('Doing sample %d'%sample)
    for ii in range(reps):
        X, y = gaussian_sparse_parity(
            sample,
            p_star=p_star,
            p=p
        )
        X_test, y_test = gaussian_sparse_parity(
            n_test,
            p_star=p_star,
            p=p
        )

        # train Vanilla NN
        vanilla_nn = getNN()
        vanilla_nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)

        # train KDN
        model_kdn = kdn(network=vanilla_nn, weighting_method="AP")
        model_kdn.fit(X, y)

        accuracy_kdn.append(
            np.mean(
                model_kdn.predict(X_test) == y_test
            )
        )
        
        accuracy_nn.append(
            np.mean(
                np.argmax(vanilla_nn.predict(X_test), axis=1) == y_test
            )
        )
        reps_list.append(ii)
        sample_list.append(sample)
        print("NN Accuracy:", accuracy_nn)
        print("KDN Accuracy:", accuracy_kdn)
        

df['accuracy kdn'] = accuracy_kdn
df['accuracy nn'] = accuracy_nn
df['reps'] = reps_list
df['sample'] = sample_list

# save the results (CHANGE HERE)
df.to_csv('results_AP/high_dim_kdn_gaussian_weighting_allFC_AP.csv')

# %% plot the result

# Specify which results to plot (CHANGE HERE)
filename1 = 'results_AP/high_dim_kdn_gaussian_weighting_allFC_AP.csv'

df = pd.read_csv(filename1)

sample_size = [1000, 5000, 10000]

err_nn_med = []
err_nn_25_quantile = []
err_nn_75_quantile = []

err_nn_med_ = []
err_nn_25_quantile_ = []
err_nn_75_quantile_ = []

err_kdn_med = []
err_kdn_25_quantile = []
err_kdn_75_quantile = []

err_kdn_med_ = []
err_kdn_25_quantile_ = []
err_kdn_75_quantile_ = []
#clr = ["#e41a1c", "#f781bf", "#306998"]
#c = sns.color_palette(clr, n_colors=3)

for sample in sample_size:
    err_nn = 1 - df['accuracy nn'][df['sample']==sample]
    #err_rf_ = 1 - df['feature selected rf'][df['sample']==sample]
    err_kdn = 1 - df['accuracy kdn'][df['sample']==sample]
    #err_kdf_ = 1 - df['feature selected kdf'][df['sample']==sample]

    err_nn_med.append(np.median(err_nn))
    err_nn_25_quantile.append(
            np.quantile(err_nn,[.25])[0]
        )
    err_nn_75_quantile.append(
        np.quantile(err_nn,[.75])[0]
    )

    #err_rf_med_.append(np.median(err_rf_))
    #err_rf_25_quantile_.append(
    #        np.quantile(err_rf_,[.25])[0]
    #    )
    #err_rf_75_quantile_.append(
    #    np.quantile(err_rf_,[.75])[0]
    #)

    err_kdn_med.append(np.median(err_kdn))
    err_kdn_25_quantile.append(
            np.quantile(err_kdn,[.25])[0]
        )
    err_kdn_75_quantile.append(
        np.quantile(err_kdn,[.75])[0]
    )

    #err_kdf_med_.append(np.median(err_kdf_))
    #err_kdf_25_quantile_.append(
    #        np.quantile(err_kdf_,[.25])[0]
    #    )
    #err_kdf_75_quantile_.append(
    #    np.quantile(err_kdf_,[.75])[0]
    #)

sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(sample_size, err_nn_med, c="k", label='NN')
ax.fill_between(sample_size, err_nn_25_quantile, err_nn_75_quantile, facecolor='k', alpha=.3)

#ax.plot(sample_size, err_rf_med_, c="g", label='RF (feature selected)')
#ax.fill_between(sample_size, err_rf_25_quantile_, err_rf_75_quantile_, facecolor='g', alpha=.3)

ax.plot(sample_size, err_kdn_med, c="r", label='KDN')
ax.fill_between(sample_size, err_kdn_25_quantile, err_kdn_75_quantile, facecolor='r', alpha=.3)

#ax.plot(sample_size, err_kdf_med_, c="b", label='KDF (feteaure selected)')
#ax.fill_between(sample_size, err_kdf_25_quantile_, err_kdf_75_quantile_, facecolor='b', alpha=.3)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xscale('log')
ax.set_xlabel('Sample size')
ax.set_ylabel('error')
ax.legend(frameon=False)

# Specify the save path (CHANGE HERE)
plt.savefig('plots/high_dim_kdn_gaussian_weighting_allFC_AP.pdf')
