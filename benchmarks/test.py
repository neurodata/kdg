#%%
from numpy import dtype
from kdg import kdf, kdn, kdcnn
from kdg.utils import get_ece, plot_reliability
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
from sklearn.model_selection import train_test_split

#%%
dataset_id = 1067#44#1497#1067#1468#44#40979#1468#11#44#1050#
dataset = openml.datasets.get_dataset(dataset_id)
X, y, is_categorical, _ = dataset.get_data(
            dataset_format="array", target=dataset.default_target_attribute
        )
#%%
unique_classes, counts = np.unique(y, return_counts=True)

test_sample = 100
total_sample = X.shape[0]
indx = list(
    range(
        total_sample
        )
)

train_samples = np.logspace(
np.log10(2),
np.log10(total_sample-test_sample),
num=10,
endpoint=True,
dtype=int
)

train_sample = train_samples[-1]
np.random.shuffle(indx)
indx_to_take_train = indx[:train_sample]
indx_to_take_test = indx[-test_sample:]       
#%%
model_kdf = kdf(k=1e-300,kwargs={'n_estimators':500, 'min_samples_leaf':10})
model_kdf.fit(X[indx_to_take_train], y[indx_to_take_train])

# %%

#%%
print(np.mean(model_kdf.predict(X[indx_to_take_test])==y[indx_to_take_test]))
print(np.mean(model_kdf.rf_model.predict(X[indx_to_take_test])==y[indx_to_take_test]))
print(np.mean(model_kdf.predict(X[indx_to_take_train])==y[indx_to_take_train]))

# %%
def compute_pdf_1d(X, location, cov):
    return np.exp(-(X-location)**2/(2*cov))/(np.sqrt(2*np.pi*cov))
# %%
val = 1
pow = 0
for dim in range(X.shape[1]):
    location = model_kdf.polytope_means[0][0][dim]
    cov = model_kdf.polytope_cov[0][0][dim]

    val *= np.exp(model_kdf.pow_exp)*compute_pdf_1d(X[:1,dim], location, cov)


    print(val, pow)
# %%
from kdg.utils import generate_gaussian_parity, gaussian_sparse_parity

X, y = generate_gaussian_parity(1000)
model_kdf = kdf(k=1e3,kwargs={'n_estimators':500, 'min_samples_leaf':30})
model_kdf.fit(X, y)
#%%
def _compute_log_likelihood(model, X, label, polytope_idx):
    polytope_mean = model.polytope_means[label][polytope_idx]
    polytope_cov = model.polytope_cov[label][polytope_idx]
    likelihood = np.zeros(X.shape[0], dtype = float)

    for ii in range(model.feature_dim):
        likelihood += model._compute_log_likelihood_1d(X[:,ii], polytope_mean[ii], polytope_cov[ii])

    likelihood += np.log(model.polytope_cardinality[label][polytope_idx]) -\
        np.log(model.total_samples_this_label[label])

    #print(np.exp(likelihood))
    return likelihood

def predict_proba(model, X, return_likelihood=False):
    r"""
    Calculate posteriors using the kernel density forest.
    Parameters
    ----------
    X : ndarray
        Input data matrix.
    """
    

    log_likelihoods = np.zeros(
        (np.size(X,0), len(model.labels)),
        dtype=float
    )
    
    for ii,label in enumerate(model.labels):
        total_polytope_this_label = len(model.polytope_means[label])
        tmp_ = np.zeros((X.shape[0],total_polytope_this_label), dtype=float)

        for polytope_idx,_ in enumerate(model.polytope_means[label]):
            tmp_[:,polytope_idx] = _compute_log_likelihood(model, X, label, polytope_idx) 
        
        print(tmp_, 'tmp')
        max_pow = np.max(
            np.concatenate(
                (
                    tmp_,
                    model.global_bias*np.ones((X.shape[0],1), dtype=float)
                ),
                axis=1
            )
        )
        pow_exp = max_pow.reshape(-1,1)@np.ones((1,total_polytope_this_label), dtype=float)
        tmp_ -= pow_exp
        print(pow_exp, tmp_, 'pow exp, tmp')
        likelihoods = np.sum(np.exp(tmp_), axis=1) +\
                np.exp(model.global_bias - pow_exp[:,0]) 
        likelihoods *= model.prior[label] 
        print(likelihoods)
        log_likelihoods[:,ii] = np.log(likelihoods) + pow_exp[:,0]

    med_pow = np.max(log_likelihoods, axis=1).reshape(-1,1)@np.ones((1,len(model.labels)))
    log_likelihoods -= med_pow
    likelihoods = np.exp(log_likelihoods)
    total_likelihoods = np.sum(likelihoods, axis=1)

    proba = (likelihoods.T/total_likelihoods).T
    
    if return_likelihood:
        return proba, likelihoods
    else:
        return proba 

# %%
from kdg.utils import generate_gaussian_parity
X_test, y_test = generate_gaussian_parity(1000)
#predict_proba(model_kdf, X_test[:2,:])
np.mean(model_kdf.predict(X_test)==y_test)
# %%
def _compute_pdf_1d(model, X, location, variance):
    if variance == 0:
        return 1
        
    return np.exp(-(X-location)**2/(2*variance))/np.sqrt(2*np.pi*variance)

def _compute_pdf(model, X, label, polytope_idx):
    polytope_mean = model.polytope_means[label][polytope_idx]
    polytope_cov = model.polytope_cov[label][polytope_idx]
    likelihood = np.ones(X.shape[0], dtype = float)

    for ii in range(model.feature_dim):   
        likelihood *= _compute_pdf_1d(model, X[:,ii], polytope_mean[ii], polytope_cov[ii])

    likelihood *= model.polytope_cardinality[label][polytope_idx]/model.total_samples_this_label[label]
    return likelihood

def predict_proba(model, X, return_likelihood=False):
    r"""
    Calculate posteriors using the kernel density forest.
    Parameters
    ----------
    X : ndarray
        Input data matrix.
    """
    

    likelihoods = np.zeros(
        (np.size(X,0), len(model.labels)),
        dtype=float
    )
    
    for ii,label in enumerate(model.labels):
        for polytope_idx,_ in enumerate(model.polytope_means[label]):
            likelihoods[:,ii] += model.prior[label] * np.nan_to_num(_compute_pdf(model, X, label, polytope_idx))

        likelihoods[:,ii] += np.exp(model.global_bias)

    proba = (likelihoods.T/np.sum(likelihoods,axis=1)).T
    
    if return_likelihood:
        return proba, likelihoods
    else:
        return proba 

# %%
#test the fitting
X_t, y_t = X[indx_to_take_train], y[indx_to_take_train]
labels = np.unique(y)
rf_model = rf(n_estimators=5, min_samples_leaf=1).fit(X, y)
feature_dim = X_t.shape[1]

for label in labels:

    X_ = X_t[np.where(y_t==label)[0]]
    predicted_leaf_ids_across_trees = np.array(
        [tree.apply(X_) for tree in rf_model.estimators_]
                ).T
    polytopes, polytope_count = np.unique(
                predicted_leaf_ids_across_trees, return_counts=True, axis=0
            )
    total_polytopes_this_label = len(polytopes)
    print(X_.shape[0], 'total sample this label')

    for polytope in range(total_polytopes_this_label):
        matched_samples = np.sum(
            predicted_leaf_ids_across_trees == polytopes[polytope],
            axis=1
        )
        idx = np.where(
            matched_samples>0
        )[0]
        
        if len(idx) == 1:
            continue
        
        scales = matched_samples[idx]/np.max(matched_samples[idx])
        X_tmp = X_[idx].copy()
        location = np.average(X_tmp, axis=0, weights=scales)
        X_tmp -= location

        covariance = np.average(X_tmp**2, axis=0, weights=scales)
        break
    break
# %%
dataset = openml.datasets.get_dataset(458)
X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )

#feature = [475, 433]
#X = X[:,feature]
X /= np.max(np.linalg.norm(X,2,axis=1))
total_sample = X.shape[0]
train_samples = np.logspace(
            np.log10(100),
            np.log10(total_sample-1000),
            num=5,
            endpoint=True,
            dtype=int
        )
X_train, X_test, y_train, y_test = train_test_split(
                     X, y, test_size=.33, random_state=44)

#%%
model_kdf = kdf(k=1e100, kwargs={'n_estimators':500})
model_kdf.fit(X_train, y_train, epsilon=1e-4)

# %%
1 - np.mean(model_kdf.predict(X_test)==y_test)
# %%
1 - np.mean(model_kdf.rf_model.predict(X_test)==y_test)

# %%
proba_kdf = model_kdf.predict_proba(X_test)
proba_rf = model_kdf.rf_model.predict_proba(X_test)
predicted_label_kdf = np.argmax(proba_kdf, axis = 1)
predicted_label_rf = np.argmax(proba_rf, axis = 1)
# %%
get_ece(proba_rf, predicted_label_rf, y_test)
# %%
get_ece(proba_kdf, predicted_label_kdf, y_test)
#%%
compile_kwargs = {
        "loss": "binary_crossentropy",
        "optimizer": keras.optimizers.Adam(3e-4),
    }
callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10, verbose=True)
fit_kwargs = {
        "epochs": 2000,
        "batch_size": 32,
        "verbose": False,
        "callbacks": [callback],
    }
# %%
def getNN(input_size, num_classes, layer_size=2000):
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

# %%
nn = getNN(input_size=X.shape[1], num_classes=len(np.unique(y)), layer_size=1000)
history = nn.fit(X_train, keras.utils.to_categorical(y_train), **fit_kwargs)

#%%
model_kdn = kdn(network=nn)
model_kdn.fit(X_train, y_train, epsilon=1e-6, mul=20)
#%%
proba_kdn = model_kdn.predict_proba(X_test)
proba_dn = model_kdn.network.predict(X_test)
predicted_label_kdn = np.argmax(proba_kdn, axis = 1)
predicted_label_dn = np.argmax(proba_dn, axis = 1)
# %%
get_ece(proba_dn, predicted_label_dn, y_test)
# %%
get_ece(proba_kdn, predicted_label_kdn, y_test)

# %%
plot_reliability(proba_kdn, predicted_label_kdn, y_test)

# %%
plot_reliability(proba_dn, predicted_label_dn, y_test)
# %%
model_kdcnn = kdcnn(network=nn)
model_kdcnn.fit(X_train, y_train)
proba_kdcnn = model_kdcnn.predict_proba(X_test)
proba_dn = model_kdcnn.network.predict(X_test)
predicted_label_kdcnn = np.argmax(proba_kdcnn, axis = 1)
predicted_label_dn = np.argmax(proba_dn, axis = 1)

#%%
get_ece(proba_dn, predicted_label_dn, y_test)
#%%
get_ece(proba_kdcnn, predicted_label_kdcnn, y_test)

# %%
plot_reliability(proba_kdcnn, predicted_label_kdcnn, y_test)
# %%
data_id = []
benchmark_suite = openml.study.get_suite('OpenML-CC18')
datasets = list(openml.study.get_suite("OpenML-CC18").data)
for dataset_id in datasets:
    print(dataset_id)
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, is_categorical, _ = dataset.get_data(
                        dataset_format="array", target=dataset.default_target_attribute
                    )
    except:
        print('could not fetch!', dataset_id)
    
    print(np.mean(is_categorical))
    if np.mean(is_categorical) >0:
        continue

    if np.isnan(np.sum(y)):
        continue

    if np.isnan(np.sum(X)):
        continue
    
    print(np.isnan(np.sum(X)), np.isnan(np.sum(y)))
    print(X.shape)
    data_id.append(dataset_id)
# %%
res_folder_kdf = 'openml_res/openml_kdf_res_ood'
files = os.listdir(res_folder_kdf)
#files.remove('.DS_Store')
id_done = []
for file in files:
    id = file[:-4]
    id_done.append(int(id[8:]))
# %%
for id in data_id:
    if id not in id_done:
        print(id)

# %%
