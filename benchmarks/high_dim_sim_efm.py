# import modules
import numpy as np
from tensorflow import keras
from keras import layers
import pandas as pd
import os

from kdg.kdn import *
from kdg.utils import gaussian_sparse_parity, trunk_sim

import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import drive 
drive.mount('/content/drive')
os.chdir("/content/drive/My Drive/NDD/Weighted_KDN")
#os.mkdir("results_EFM")
#os.mkdir("plots")
!ls

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

sample_size = [1000, 2000, 5000, 10000, 20000] # sample size under consideration
network_size = [5, 10, 15, 20] #test under a variety of network sizes

n_test = 1000 # test set size
reps = 10 # number of replicates

df = pd.DataFrame()
reps_list = []
accuracy_kdn = []
accuracy_nn = []
accuracy_efm = []
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
def getNN(dense_size=5):
    network_base = keras.Sequential()
    network_base.add(layers.Dense(dense_size, activation='relu', input_shape=(20,)))
    network_base.add(layers.Dense(dense_size, activation='relu'))
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

        # train KDN - unweighted
        unweighted_kdn = kdn(network=vanilla_nn, weighting_method=None,verbose=False)
        unweighted_kdn.fit(X, y)

        accuracy_kdn.append(
            np.mean(
                unweighted_kdn.predict(X_test) == y_test
            )
        )

        # train KDN - pseudo-ensembled FM
        efm_kdn = kdn(network=vanilla_nn, weighting_method="EFM", verbose=False)
        efm_kdn.fit(X, y)

        accuracy_efm.append(
            np.mean(
                efm_kdn.predict(X_test) == y_test
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
    print("KDN-Unweighted Accuracy:", accuracy_kdn)
    print("KDN-EFM Accuracy:", accuracy_efm)
    

df['accuracy unweighted'] = accuracy_kdn
df['accuracy efm'] = accuracy_efm
df['accuracy nn'] = accuracy_nn
df['reps'] = reps_list
df['sample'] = sample_list

# save the results (CHANGE HERE)
df.to_csv('results_EFM/high_dim_kdn_gaussian_weighting_allFC_EFM_samplesize.csv')

# %%

sample = 2000

df2 = pd.DataFrame()
reps_list = []
accuracy_kdn = []
accuracy_nn = []
accuracy_efm = []
sample_list = []

for N in network_size:
    print(f'Using network with {N} nodes')
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
        vanilla_nn = getNN(N)
        vanilla_nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)

        # train KDN - unweighted
        unweighted_kdn = kdn(network=vanilla_nn, weighting_method=None,verbose=False)
        unweighted_kdn.fit(X, y)

        accuracy_kdn.append(
            np.mean(
                unweighted_kdn.predict(X_test) == y_test
            )
        )

        # train KDN - pseudo-ensembled FM
        efm_kdn = kdn(network=vanilla_nn, weighting_method="EFM", verbose=False)
        efm_kdn.fit(X, y)

        accuracy_efm.append(
            np.mean(
                efm_kdn.predict(X_test) == y_test
            )
        )
        
        accuracy_nn.append(
            np.mean(
                np.argmax(vanilla_nn.predict(X_test), axis=1) == y_test
            )
        )
        reps_list.append(ii)
        sample_list.append(N)
    print("NN Accuracy:", accuracy_nn)
    print("KDN-Unweighted Accuracy:", accuracy_kdn)
    print("KDN-EFM Accuracy:", accuracy_efm)
    
df2['accuracy unweighted'] = accuracy_kdn
df2['accuracy efm'] = accuracy_efm
df2['accuracy nn'] = accuracy_nn
df2['reps'] = reps_list
df2['sample'] = sample_list

# save the results (CHANGE HERE)
df2.to_csv('results_EFM/high_dim_kdn_gaussian_weighting_allFC_EFM_networksize.csv')

# %% plot the result
import pandas as pd
# Specify which results to plot (CHANGE HERE)
filename1 = 'results_EFM/high_dim_kdn_gaussian_weighting_allFC_EFM_networksize.csv'

df = pd.read_csv(filename1)

err_nn_med = []
err_nn_25_quantile = []
err_nn_75_quantile = []

err_kdn_med = []
err_kdn_25_quantile = []
err_kdn_75_quantile = []

err_efm_med = []
err_efm_25_quantile = []
err_efm_75_quantile = []

#clr = ["#e41a1c", "#f781bf", "#306998"]
#c = sns.color_palette(clr, n_colors=3)

for sample in network_size:
    print(sample)
    err_nn = 1 - df['accuracy nn'][df['sample']==sample]
    #err_rf_ = 1 - df['feature selected rf'][df['sample']==sample]
    err_kdn = 1 - df['accuracy unweighted'][df['sample']==sample]
    #err_kdf_ = 1 - df['feature selected kdf'][df['sample']==sample]
    err_efm = 1 - df['accuracy efm'][df['sample']==sample]

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

    err_efm_med.append(np.median(err_efm))
    err_efm_25_quantile.append(
            np.quantile(err_efm,[.25])[0]
        )
    err_efm_75_quantile.append(
        np.quantile(err_efm,[.75])[0]
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

ax.plot(sample_size, err_kdn_med, c="b", label='KDN-Unweighted')
ax.fill_between(sample_size, err_kdn_25_quantile, err_kdn_75_quantile, facecolor='b', alpha=.3)

ax.plot(sample_size, err_efm_med, c="r", label='KDN-EFM')
ax.fill_between(sample_size, err_efm_25_quantile, err_efm_75_quantile, facecolor='r', alpha=.3)


right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xscale('log')
ax.set_xlabel('Sample size')
ax.set_ylabel('error')
ax.legend(frameon=False)

# Specify the save path (CHANGE HERE)
plt.savefig('plots/high_dim_efm_gaussian_weighting_allFC_EFM_networksize.pdf')