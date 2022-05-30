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
model_kdf = kdf(k=1e300,kwargs={'n_estimators':500, 'min_samples_leaf':10})
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
#
# Created on Mon Jan 31 2022 10:02:26 AM
# Author: Ashwin De Silva (ldesilv2@jhu.edu)
# Objective: Kernel Density Network
#

# import standard libraries
from .base import KernelDensityGraph
from sklearn.utils.validation import check_array, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf


class kdn(KernelDensityGraph):
    def __init__(
        self,
        network,
        weighting=True,
        k=1.0,
        T=1e-3,
        h=0.33,
        verbose=True,
    ):
        r"""[summary]

        Parameters
        ----------
        network : tf.keras.Model()
            trained neural network model
        weighting : bool, optional
            use weighting if true, by default True
        k : float, optional
            bias control parameter, by default 1
        T : float, optional
            neighborhood size control parameter, by default 1e-3
        h : float, optional
            variational parameters of the weighting, by default 0.33
        verbose : bool, optional
            print internal data, by default True
        """
        super().__init__()
        self.polytope_means = {}
        self.polytope_cov = {}
        self.polytope_cardinality = {}
        self.total_samples_this_label = {}
        self.prior = {}
        self.network = network
        self.weighting = weighting
        self.k = k
        self.h = h
        self.T = T
        self.bias = {}
        self.verbose = verbose
        self.is_fitted = False

        # total number of layers in the NN
        self.total_layers = len(self.network.layers)

        # get the sizes of each layer
        self.network_shape = []
        for layer in network.layers:
            self.network_shape.append(layer.output_shape[-1])

        # total number of units in the network (up to the penultimate layer)
        #self.num_neurons = sum(self.network_shape) - self.network_shape[-1]

        # get the weights and biases of the trained MLP
        self.weights = {}
        self.biases = {}
        for i in range(len(self.network.layers)):
            weight, bias = self.network.layers[i].get_weights()
            self.weights[i], self.biases[i] = weight, bias.reshape(1, -1)

    def _get_polytope_ids(self, X):
        r"""
        Obtain the polytope ID of each input sample
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        polytope_ids_tmp = []
        last_activations = X

        # Iterate through neural network manually, getting node activations at each step
        for l in range(self.total_layers):
            weights, bias = self.weights[l], self.biases[l]
            preactivation = np.matmul(last_activations, weights) + bias

            if l == self.total_layers - 1:
                binary_preactivation = (preactivation > 0.5).astype("int")
            else:
                binary_preactivation = (preactivation > 0).astype("int")

            if (
                l < self.total_layers - 1
            ):  # record the activation patterns only upto the penultimate layer
                polytope_ids_tmp.append(binary_preactivation)

            last_activations = preactivation * binary_preactivation

        # Concatenate all activations for given observation
        polytope_ids_tmp = np.concatenate(polytope_ids_tmp, axis=1)

        self.num_neurons = polytope_ids_tmp.shape[
            1
        ]  # get the number of total FC neurons under consideration
        total_samples = X.shape[0]
        polytope_ids = ["" for ii in range(total_samples)]
        #print(polytope_ids_tmp,"prev")
        for ii in range(total_samples):
            for jj in polytope_ids_tmp[ii]:
                polytope_ids[ii] += str(jj)
        #print(polytope_ids, "after")
        return polytope_ids

    def _get_activation_pattern(self, polytope_id):
        r"""get the ReLU activation pattern given the polytope ID

        Parameters
        ----------
        polytope_id : int
            polytope identifier

        Returns
        -------
        ndarray
            ReLU activation pattern (binary) corresponding to the given polytope ID
        """
        binary_string = np.binary_repr(polytope_id, width=self.num_neurons)[::-1]
        return np.array(list(binary_string)).astype("int")

    def compute_weights(self, X_, polytope_id):
        """compute weights based on the global network linearity measure
        Parameters
        ----------
        X_ : ndarray
            Input data matrix
        polytope_id : int
            refernce polytope identifier
        Returns
        -------
        ndarray
            weights of each input sample in the input data matrix
        """

        M_ref = polytope_id
        start = 0
        A = X_
        A_ref = X_
        d = 0
        #print(M_ref, 'M_ref')
        for l in range(len(self.network_shape) - 1):
            M_l = np.zeros((self.network_shape[l]))
            end = start + self.network_shape[l]

            for ii in range(self.network_shape[l]):
                M_l[ii] = int(M_ref[start+ii])
            #print(M_l, 'eeuu')
            start = end
            W, B = self.weights[l], self.biases[l]
            pre_A = A @ W + B
            A = np.maximum(0, pre_A)
            pre_A_ref = A_ref @ W + B
            A_ref = pre_A_ref @ np.diag(M_l)
            #print(A, A_ref)
            d += np.linalg.norm(A - A_ref, axis=1, ord=2)
            #print(d, 'distance')
        return np.exp(-d / self.h)

    def fit(self, X, y):
        r"""
        Fits the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response) data matrix.
        """
        if self.is_fitted:
            raise ValueError(
                "Model already fitted!"
            )
            return

        X, y = check_X_y(X, y)
        self.labels = np.unique(y)
        self.feature_dim = X.shape[1]

        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_cov[label] = []
            self.polytope_cardinality[label] = []

            X_ = X[np.where(y == label)[0]]  # data having the current label
            self.total_samples_this_label[label] = X_.shape[0]
            # get class prior probability
            self.prior[label] = len(X_) / len(X)

            # get polytope ids and unique polytope ids
            polytope_ids = self._get_polytope_ids(X_)
            unique_polytope_ids, counts = np.unique(polytope_ids, return_counts=True)
            print(counts)
            for polytope in unique_polytope_ids:
                weights = self.compute_weights(X_, polytope)
                print(len(np.where(weights==1)[0]))
                if not self.weighting:
                    weights[weights < 1] = 0
                weights[weights < self.T] = 0  # set very small weights to zero
                #points_with_nonzero_weights = len(np.where(weights > 0)[0])
                '''if points_with_nonzero_weights < 2:
                    continue'''

                # apply weights to the data
                X_tmp = X_.copy()
                polytope_mean_ = np.average(
                    X_tmp, axis=0, weights=weights
                )  # compute the weighted average of the samples
                #print(polytope_mean_, weights, 'gdgtdt')
                X_tmp -= polytope_mean_  # center the data
                polytope_cov_ = np.average(X_tmp**2, axis=0, weights=weights)

                polytope_samples_ = len(
                    np.where(polytope_ids == polytope)[0]
                )  # count the number of points in the polytope

                # store the mean, covariances, and polytope sample size
                self.polytope_means[label].append(polytope_mean_)
                self.polytope_cov[label].append(polytope_cov_)
                self.polytope_cardinality[label].append(polytope_samples_)

            ## calculate bias for each label
            likelihoods = np.zeros(
                (np.size(X_,0)),
                dtype=float
            )
            for polytope_idx,_ in enumerate(self.polytope_means[label]):
                likelihoods += self._compute_log_likelihood(X_, label, polytope_idx)

            #likelihoods -= np.log(self.total_samples_this_label[label]
            self.bias[label] = np.min(likelihoods) - np.log(self.k*self.total_samples_this_label[label])

        self.global_bias = min(self.bias.values())
        min_bias = -10**(np.log10(X.shape[1]) +1)- np.log(self.k) -np.log(X.shape[0])

        if self.global_bias < min_bias:
            self.global_bias = min_bias

        self.is_fitted = True

    def _compute_log_likelihood_1d(self, X, location, variance):                   
        return -(X-location)**2/(2*variance) - .5*np.log(2*np.pi*variance)

    def _compute_log_likelihood(self, X, label, polytope_idx):
        polytope_mean = self.polytope_means[label][polytope_idx]
        polytope_cov = self.polytope_cov[label][polytope_idx]
        likelihood = np.zeros(X.shape[0], dtype = float)

        for ii in range(self.feature_dim):
            likelihood += self._compute_log_likelihood_1d(X[:,ii], polytope_mean[ii], polytope_cov[ii])

        likelihood += np.log(self.polytope_cardinality[label][polytope_idx]) -\
            np.log(self.total_samples_this_label[label])

        return likelihood

    def predict_proba(self, X, return_likelihood=False):
        r"""
        Calculate posteriors using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        
        X = check_array(X)

        log_likelihoods = np.zeros(
            (np.size(X,0), len(self.labels)),
            dtype=float
        )
        
        for ii,label in enumerate(self.labels):
            total_polytope_this_label = len(self.polytope_means[label])
            tmp_ = np.zeros((X.shape[0],total_polytope_this_label), dtype=float)

            for polytope_idx,_ in enumerate(self.polytope_means[label]):
                tmp_[:,polytope_idx] = self._compute_log_likelihood(X, label, polytope_idx) 
            
            max_pow = np.max(
                    np.concatenate(
                        (
                            tmp_,
                            self.global_bias*np.ones((X.shape[0],1), dtype=float)
                        ),
                        axis=1
                    ),
                    axis=1
                )
            #print(max_pow, total_polytope_this_label, label)
            pow_exp = np.nan_to_num(
                max_pow.reshape(-1,1)@np.ones((1,total_polytope_this_label), dtype=float)
            )
            tmp_ -= pow_exp
            likelihoods = np.sum(np.exp(tmp_), axis=1) +\
                 np.exp(self.global_bias - pow_exp[:,0]) 
                
            likelihoods *= self.prior[label] 
            log_likelihoods[:,ii] = np.log(likelihoods) + pow_exp[:,0]

        max_pow = np.nan_to_num(
            np.max(log_likelihoods, axis=1).reshape(-1,1)@np.ones((1,len(self.labels)))
        )
        log_likelihoods -= max_pow
        likelihoods = np.exp(log_likelihoods)

        total_likelihoods = np.sum(likelihoods, axis=1)

        proba = (likelihoods.T/total_likelihoods).T
        
        if return_likelihood:
            return proba, likelihoods
        else:
            return proba 

    def predict(self, X):
        r"""
        Perform inference using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        return np.argmax(self.predict_proba(X), axis = 1)