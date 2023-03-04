#%%
from numpy import dtype
from kdg import kdf
from kdg.utils import get_ece
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

#%%
dataset_id = 44#1497#1067#1468#44#40979#1468#11#44#1050#
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
model_kdf = kdf(k=1e-2,kwargs={'n_estimators':500, 'min_samples_leaf':10})
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
