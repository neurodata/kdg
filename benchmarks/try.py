#%%
from numpy import dtype
from kdg import kdf, kdn, kdcnn
from kdg.utils import get_ece, plot_reliability, get_ace
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import activations
from tensorflow.keras import backend as bknd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import numpy as np
import openml
from sklearn.metrics import cohen_kappa_score
from kdg.utils import get_ece
from numpy import min_scalar_type
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.stats import multivariate_normal
import warnings
from sklearn.covariance import MinCovDet, fast_mcd, GraphicalLassoCV, LedoitWolf, EmpiricalCovariance, OAS, EllipticEnvelope, log_likelihood
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rf 
from tensorflow.keras.datasets import cifar10, cifar100
import joblib
#%%
compile_kwargs = {
        "loss": "binary_crossentropy",
        "optimizer": keras.optimizers.Adam(3e-4),
    }
callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10, verbose=True)
fit_kwargs = {
        "epochs": 200,
        "batch_size": 32,
        "verbose": True,
        "callbacks": [callback],
    }
# %%
def getNN(input_size, num_classes, layer_size=1000):
    network_base = keras.Sequential()
    initializer = keras.initializers.random_normal(seed=0)
    network_base.add(keras.layers.Dense(layer_size, kernel_initializer=initializer, input_shape=(input_size,)))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(layer_size, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(layer_size, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(layer_size, kernel_initializer=initializer))
    network_base.add(keras.layers.Activation(activations.relu))
    network_base.add(keras.layers.Dense(units=num_classes, activation="softmax", kernel_initializer=initializer))
    network_base.compile(**compile_kwargs)
    return network_base

#%%
dataset_id = 23517#44#11#22#11#40979#1067#1468#44#40979#1468#11#44#1050#
dataset = openml.datasets.get_dataset(dataset_id)
X, y, is_categorical, _ = dataset.get_data(
            dataset_format="array", target=dataset.default_target_attribute
        )

print(X.shape)
min_val = np.min(X,axis=0)
max_val = np.max(X, axis=0)
X = (X-min_val)/(max_val-min_val+1e-12)
_, y = np.unique(y, return_inverse=True)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, train_size=1000, random_state=10, stratify=y)
X_train, X_cal, y_train, y_cal = train_test_split(
                X_train, y_train, train_size=0.9, random_state=10, stratify=y_train)

#%%
model = rf(n_estimators=500)
model.fit(X_train, y_train)

#%%
model_kdf = kdf(rf_model=model)
model_kdf.fit(X_train, y_train, X_cal, y_cal)#, k=int(np.ceil(.4*1000)))
model_kdf.global_bias=-1e100

#%%
proba = model_kdf.predict_proba(X_test,distance='Geodesic')
print('Accuracy=',np.mean(y_test==np.argmax(proba,axis=1)))

print('ECE=',get_ece(proba,y_test, n_bins=15))

#%%
proba_rf = model_kdf.rf_model.predict_proba(X_test)
print('Accuracy=',np.mean(y_test==np.argmax(proba_rf,axis=1)))
print('ECE=',get_ece(proba_rf,y_test, n_bins=15))
#%%
nn = getNN(input_size=X_train.shape[1], num_classes=np.max(y_train)+1, layer_size=1000)
history = nn.fit(X_train, keras.utils.to_categorical(y_train), **fit_kwargs)
#%%
model_kdn = kdn(network=nn)
model_kdn.fit(X_train, y_train, X_cal, y_cal)#, k=int(np.ceil(.4*1000)))
model_kdn.global_bias=-1e100

#%%
proba = model_kdn.predict_proba(X_test,distance='Geodesic')
print('Accuracy=',np.mean(y_test==np.argmax(proba,axis=1)))

print('ECE=',get_ece(proba,y_test, n_bins=15))
#%%
proba_dn = model_kdn.network.predict(X_test)
print('Accuracy=',np.mean(y_test==np.argmax(proba_dn,axis=1)))
print('ECE=',get_ece(proba_dn,y_test, n_bins=15))
#%%
polytope_ids = model_kdn._get_polytope_ids(
                    np.array(X_train)
                )
w = 1- model_kdn._compute_geodesic(polytope_ids, polytope_ids)
#%%
k = 600#int(np.ceil(np.sqrt(X_train.shape[0])/2))#int(0.0006474*X_train.shape[0]*X.shape[1]**(1/1.323))
for id in range(100):
    #id = 15
    idx = np.argsort(w[id])[::-1]

    lbl = np.zeros(np.max(y_train)+1,dtype=float)
    for ii in idx[:k]:
        lbl[int(y_train[ii])] += 1#w[id,ii]
        #print(w[id,ii], int(y_train[ii]))

    print(lbl, w[id,ii], np.max(lbl)/np.sum(lbl))
print(X_train.shape)


# %%
def predict_proba(model, X, distance = 'Euclidean', return_likelihood=False, n_jobs=-1):
    r"""
    Calculate posteriors using the kernel density forest.
    Parameters
    ----------
    X : ndarray
        Input data matrix.
    """
    #X = check_array(X)
    
    total_polytope = len(model.polytope_means)
    log_likelihoods = np.zeros(
        (np.size(X,0), len(model.labels)),
        dtype=float
    )
    
    print('Calculating distance')
    if distance == 'Euclidean':
        distance = model._compute_euclidean(X)
        polytope_idx = np.argmin(distance, axis=1)
    elif distance == 'Geodesic':
        total_polytope = len(model.polytope_means)
        batch = total_polytope//1000 + 1
        batchsize = total_polytope//batch
        polytope_ids = model._get_polytope_ids(
                np.array(model.polytope_means[:batchsize])
            ) 

        indx_X2 = np.inf
        for ii in range(1,batch):
            #print("doing batch ", ii)
            indx_X1 = ii*batchsize
            indx_X2 = (ii+1)*batchsize
            polytope_ids = np.concatenate(
                (polytope_ids,
                model._get_polytope_ids(
                np.array(model.polytope_means[indx_X1:indx_X2])
                )),
                axis=0
            )
        
        if indx_X2 < len(model.polytope_means):
            polytope_ids = np.concatenate(
                    (polytope_ids,
                    model._get_polytope_ids(
                np.array(model.polytope_means[indx_X2:]))),
                    axis=0
                )

        total_sample = X.shape[0]
        batch = total_sample//1000 + 1
        batchsize = total_sample//batch
        test_ids = model._get_polytope_ids(X[:batchsize]) 

        indx_X2 = np.inf
        for ii in range(1,batch):
            #print("doing batch ", ii)
            indx_X1 = ii*batchsize
            indx_X2 = (ii+1)*batchsize
            test_ids = np.concatenate(
                (test_ids,
                model._get_polytope_ids(X[indx_X1:indx_X2])),
                axis=0
            )
        
        if indx_X2 < X.shape[0]:
            test_ids = np.concatenate(
                    (test_ids,
                    model._get_polytope_ids(X[indx_X2:])),
                    axis=0
                )
            
        print('Polytope extracted!')
        ####################################
        batch = total_sample//50000 + 1
        batchsize = total_sample//batch
        polytope_idx = []
        indx = [jj*batchsize for jj in range(batch+1)]
        if indx[-1] < total_sample:
            indx.append(total_sample)

        for ii in range(len(indx)-1):    
            polytope_idx.extend(
                list(np.argmin(
                    model._compute_geodesic(
                        test_ids[indx[ii]:indx[ii+1]],
                        polytope_ids,
                        batch=n_jobs
                    ), axis=1
                ))
            )
    else:
        raise ValueError("Unknown distance measure!")
    
    for ii,label in enumerate(model.labels):
        for jj in range(X.shape[0]):
            log_likelihoods[jj, ii] = model._compute_log_likelihood(X[jj], label, polytope_idx[jj])
            max_pow = max(log_likelihoods[jj, ii], model.global_bias)
            log_likelihoods[jj, ii] = np.log(
                (np.exp(log_likelihoods[jj, ii] - max_pow)\
                    + np.exp(model.global_bias - max_pow))
                    *model.prior[label]
            ) + max_pow
            
    max_pow = np.nan_to_num(
        np.max(log_likelihoods, axis=1).reshape(-1,1)@np.ones((1,len(model.labels)))
    )

    print(log_likelihoods)
    if return_likelihood:
        likelihood = np.exp(log_likelihoods)
    
    log_likelihoods -= max_pow
    likelihoods = np.exp(log_likelihoods)

    total_likelihoods = np.sum(likelihoods, axis=1)

    proba = (likelihoods.T/total_likelihoods).T
    
    if return_likelihood:
        return proba, likelihood
    else:
        return proba, polytope_idx
# %%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(_, _), (x_cifar100, y_cifar100) = cifar100.load_data()
x_noise = np.random.random_integers(0,high=255,size=(20,32,32,3)).astype('float')/255.0

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_cifar100 = x_cifar100.astype('float32') / 255

for channel in range(3):
    x_train_mean = np.mean(x_train[:,:,:,channel])
    x_train_std = np.std(x_train[:,:,:,channel])
    x_noise[:,:,:,channel] -= x_train_mean
    x_noise[:,:,:,channel] /= x_train_std
    x_test[:,:,:,channel] -= x_train_mean
    x_test[:,:,:,channel] /= x_train_std 
    x_cifar100[:,:,:,channel] -= x_train_mean
    x_cifar100[:,:,:,channel] /= x_train_std
# %%
model_kdn = joblib.load('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/resnet20_models/resnet_kdn_50000_100.joblib')
model_kdn.global_bias = -2.6e9
# %%
p, d = predict_proba(model_kdn, x_noise, distance='Geodesic')

# %%
p_in, d_in = predict_proba(model_kdn, x_test[:200], distance='Geodesic')

# %%
p_acet = acet.predict(x_cifar100[:40])
np.mean(np.max(p_acet,axis=1))
# %%
np.mean(np.max(p,axis=1))
# %%
np.mean(np.max(p_in,axis=1))
# %%
p_cifar100, d_cifar100 = predict_proba(model_kdn, x_cifar100[:40], distance='Geodesic')

# %%
np.mean(np.max(p_cifar100,axis=1))
# %%
p_noise_dn = model_kdn.network.predict(x_noise)
np.mean(np.max(p_noise_dn,axis=1))
# %%
import imageio as iio

img = iio.imread("/Users/jayantadey/Downloads/dtd/images/perforated/perforated_0082.jpg")
x = img.astype('float32')/255
x = tf.image.resize(x, input_shape[:-1],
    antialias=True)
x = x.numpy()


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
for channel in range(3):
    x_train_mean = np.mean(x_train[:,:,:,channel])
    x_train_std = np.std(x_train[:,:,:,channel])
    x[:,:,channel] -= x_train_mean
    x[:,:,channel] /= x_train_std

#%%
predict_proba(model_kdn, x.reshape(1,32,32,3), distance='Geodesic')
# %%
plt.imshow(x)
# %%
plt.imshow(model_kdn.polytope_means[2106])
# %%
p_svhn = model_kdn.predict_proba(x_svhn[:40], distance='Geodesic')
np.mean(np.max(p_svhn,axis=1))
# %%
