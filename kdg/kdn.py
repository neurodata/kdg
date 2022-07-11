from matplotlib.pyplot import sca
from .base import KernelDensityGraph
from sklearn.utils.validation import check_array, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf
from tensorflow.keras.models import Model
from joblib import Parallel, delayed 
from tensorflow.keras import backend as bknd
from scipy.sparse import csr_matrix

class kdn(KernelDensityGraph):
    def __init__(
        self,
        network,
        k = 1.0,
        threshold = 0.6,
        verbose=True
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
        self.k = k
        self.threshold = threshold
        self.bias = {}
        self.verbose = verbose

        # total number of layers in the NN
        self.total_layers = 0

        # get the sizes of each layer
        self.network_shape = []
        for layer in network.layers:
            layer_name = layer.name

            if 'activation' in layer_name:
                self.total_layers += 1
                self.network_shape.append(
                    np.product(
                        layer.output_shape[1:]
                    )
                )

    def _get_polytope_ids(self, X):
        total_samples = X.shape[0]
           
        outputs = [] 
        inp = self.network.input

        for layer in self.network.layers:
            if 'activation' in layer.name:
                outputs.append(layer.output) 

        functor = bknd.function(inp, outputs)
        layer_outs = functor(X)

        activation = []
        for layer_out in layer_outs:
            activation.append(
                (layer_out>0).astype('int').reshape(total_samples, -1)
            )
        polytope_ids = np.concatenate(activation, axis=1)
        
        return polytope_ids
    
        
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
        #X, y = check_X_y(X, y)
        self.labels = np.unique(y)
        self.feature_dim = np.product(X.shape[1:])
        self.total_sample = X.shape[0]

        for label in self.labels:
            print('doing label ',label)
            self.polytope_means[label] = []
            self.polytope_cov[label] = []

            # data having the current label
            X_ = X[np.where(y == label)[0]]
            self.total_samples_this_label[label] = X_.shape[0]

            # get class prior probability
            self.prior[label] = \
                self.total_samples_this_label[label] / self.total_sample 
            
            # get polytope ids and unique polytope ids
            polytope_ids = self._get_polytope_ids(X_)
            unique_polytope_ids, polytope_samples_ = np.unique(polytope_ids, return_counts=True, axis=0)
            self.polytope_cardinality[label] = polytope_samples_
            
            for polytope in unique_polytope_ids:
                indx = np.where(polytope==0)[0]
                polytope_ = polytope.copy()
                polytope_[indx] = 2

                matched_pattern = (polytope_ids==polytope_)
                matched_nodes = np.zeros((len(polytope_ids),self.total_layers))
                end_node = 0
                normalizing_factor = 0
                for layer in range(self.total_layers):
                    end_node += self.network_shape[layer]
                    matched_nodes[:, layer] = \
                        np.sum(matched_pattern[:,end_node-self.network_shape[layer]:end_node], axis=1)\
                            + 1/self.network_shape[layer]

                    normalizing_factor += \
                        np.log(np.max(matched_nodes[:, layer]))
                    #print(normalizing_factor, np.where(polytope[end_node-self.network_shape[layer]:end_node])[0])
                #print(normalizing_factor, np.sum(np.log(matched_nodes), axis=1), 'match')
                scales = np.exp(np.sum(np.log(matched_nodes), axis=1)\
                    - normalizing_factor)
                indx_to_consider = np.where(
                    scales>self.threshold
                )[0]
                #print(np.sort(scales))
                if indx_to_consider.shape[0] == 1:
                    continue
                
                # apply weights to the data
                X_tmp = X_.reshape(
                    X_.shape[0], -1
                ).copy()
                polytope_mean_ = np.average(
                    X_tmp[indx_to_consider], axis=0, weights=scales[indx_to_consider]
                )  # compute the weighted average of the samples
                
                X_tmp[indx_to_consider] -= polytope_mean_  # center the data
                polytope_cov_ = np.average(
                    X_tmp[indx_to_consider]**2,
                    axis=0, 
                    weights=scales[indx_to_consider]
                )
                
                # store the mean, covariances, and polytope sample size
                self.polytope_means[label].append(polytope_mean_)
                self.polytope_cov[label].append(
                        polytope_cov_
                    )
                
            ## calculate bias for each label
            likelihoods = np.zeros(
                (np.size(X_,0)),
                dtype=float
            )
            for polytope_idx,_ in enumerate(self.polytope_means[label]):
                likelihoods += self._compute_log_likelihood(X_.reshape(X_.shape[0],-1), label, polytope_idx)

            #likelihoods -= np.log(self.total_samples_this_label[label]
            self.bias[label] = np.min(likelihoods) - np.log(self.k*self.total_samples_this_label[label])

        self.global_bias = min(self.bias.values())
        min_bias = -10**(np.log10(np.product(X.shape[1:])) +1)- np.log(self.k) -np.log(X.shape[0])
        
        if self.global_bias < min_bias:
            self.global_bias = min_bias

        self.is_fitted = True

    def _compute_log_likelihood_1d(self, X, location, variance):  
        if variance < 1e-200:
            return 0
        else:                
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
        
        #X = check_array(X)
        X = X.reshape(
            X.shape[0],
            -1
        )
        log_likelihoods = np.zeros(
            (np.size(X,0), len(self.labels)),
            dtype=float
        )
        
        for ii,label in enumerate(self.labels):
            total_polytope_this_label = len(self.polytope_means[label])
            tmp_ = np.zeros((X.shape[0],total_polytope_this_label), dtype=float)

            for polytope_idx,_ in enumerate(self.polytope_means[label]):
                tmp_[:,polytope_idx] = self._compute_log_likelihood(X, label, polytope_idx) 
            
            log_likelihood = np.max(tmp_, axis=1) 
            '''max_pow = np.max(
                    np.concatenate(
                        (
                            tmp_,
                            self.global_bias*np.ones((X.shape[0],1), dtype=float)
                        ),
                        axis=1
                    ),
                    axis=1
                )
            pow_exp = np.nan_to_num(
                max_pow.reshape(-1,1)@np.ones((1,total_polytope_this_label), dtype=float)
            )
            tmp_ -= pow_exp
            likelihoods = np.sum(np.exp(tmp_), axis=1) +\
                 np.exp(self.global_bias - pow_exp[:,0]) '''

            log_likelihoods[:,ii] = log_likelihood + np.log(self.prior[label])

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