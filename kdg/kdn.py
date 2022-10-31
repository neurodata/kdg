from matplotlib.pyplot import sca
from .base import KernelDensityGraph
from sklearn.utils.validation import check_array, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf
from tensorflow.keras.models import Model
from joblib import Parallel, delayed 
from tensorflow.keras import backend as bknd
#from scipy.sparse import csr_matrix

class kdn(KernelDensityGraph):
    def __init__(
        self,
        network,
        verbose=True
    ):
        r"""[summary]
        Parameters
        ----------
        network : tf.keras.Model()
            trained neural network model
        weighting : bool, optional
            use weighting if true, by default True
        verbose : bool, optional
            print internal data, by default True
        """
        super().__init__()
        self.polytope_means = []
        self.polytope_cov = []
        self.polytope_cardinality = {}
        self.total_samples_this_label = {}
        self.prior = {}
        self.network = network
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
                (layer_out>0).astype('bool').reshape(total_samples, -1)
            )
        polytope_ids = np.concatenate(activation, axis=1)
        
        return polytope_ids
       
    def fit(self, X, y, epsilon=1e-6, batch=10):
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
        X = X.astype('double')
        self.max_val = np.max(X, axis=0) 
        self.min_val = np.min(X, axis=0)

        X = (X-self.min_val)/(self.max_val-self.min_val+1e-8)

        self.labels = np.unique(y)
        self.feature_dim = np.product(X.shape[1:])
        self.global_bias = -10**(np.sqrt(self.feature_dim))
        
        for label in self.labels:
            self.polytope_cardinality[label] = []
            self.total_samples_this_label[label] = len(
                    np.where(y==label)[0]
                )
            self.prior[label] = self.total_samples_this_label[label]/X.shape[0]

        # get polytope ids and unique polytope ids
        batchsize = X.shape[0]//batch
        polytope_ids = self._get_polytope_ids(X[:batchsize])

        for ii in range(1,batch):
            print("doing batch ", ii)
            indx_X1 = ii*batchsize
            indx_X2 = (ii+1)*batchsize
            polytope_ids = np.concatenate(
                (polytope_ids,
                self._get_polytope_ids(X[indx_X1:indx_X2])),
                axis=0
            )
        
        if indx_X2 < X.shape[0]:
            polytope_ids = np.concatenate(
                    (polytope_ids,
                    self._get_polytope_ids(X[indx_X2:])),
                    axis=0
                )
        #print(polytope_ids.shape)
        polytopes = np.unique(
            polytope_ids, axis=0
            )
        
        for polytope in polytopes:
            #indx = np.where(polytope==0)[0]
            #polytope_ = polytope.copy()
            #polytope_[indx] = 2

            matched_pattern = (polytope_ids==polytope)
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

            scales = np.exp(np.sum(np.log(matched_nodes), axis=1)\
                - normalizing_factor)

            idx_with_scale_1 = np.where(
                    scales>.99999999999
                )[0]
            idx_with_scale_alpha = np.where(
                    scales>0
                )[0]

            location = np.mean(X[idx_with_scale_1], axis=0)
            X_tmp = X[idx_with_scale_alpha].copy() - location
            covariance = np.average(X_tmp**2+epsilon/np.sum(scales[idx_with_scale_alpha]), axis=0, weights=scales[idx_with_scale_alpha])
            self.polytope_cov.append(covariance)
            self.polytope_means.append(location)

            y_tmp = y[idx_with_scale_1]
            for label in self.labels:      
                self.polytope_cardinality[label].append(
                    len(np.where(y_tmp==label)[0])
                )

        self.global_bias = self.global_bias/X.shape[0]
        self.is_fitted = True    
        
    def _compute_mahalanobis(self, X, polytope):
            return np.sum(
                (X - self.polytope_means[polytope])**2\
                    *(1/self.polytope_cov[polytope]),
                axis=1
            )

    def _compute_log_likelihood_1d(self, X, location, variance):        
        return -(X-location)**2/(2*variance) - .5*np.log(2*np.pi*variance)

    def _compute_log_likelihood(self, X, label, polytope_idx):
        polytope_mean = self.polytope_means[polytope_idx]
        polytope_cov = self.polytope_cov[polytope_idx]
        likelihood = 0

        for ii in range(self.feature_dim):
            likelihood += self._compute_log_likelihood_1d(X[ii], polytope_mean[ii], polytope_cov[ii])
        
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
        X = (X-self.min_val)/(self.max_val-self.min_val+ 1e-8)

        total_polytope = len(self.polytope_means)
        log_likelihoods = np.zeros(
            (np.size(X,0), len(self.labels)),
            dtype=float
        )
        distance = np.zeros(
                (
                    np.size(X,0),
                    total_polytope
                ),
                dtype=float
            )
        
        for polytope in range(total_polytope):
            distance[:,polytope] = self._compute_mahalanobis(X, polytope)

        polytope_idx = np.argmin(distance, axis=1)

        for ii,label in enumerate(self.labels):
            for jj in range(X.shape[0]):
                log_likelihoods[jj, ii] = self._compute_log_likelihood(X[jj], label, polytope_idx[jj])
                max_pow = max(log_likelihoods[jj, ii], self.global_bias)
                log_likelihoods[jj, ii] = np.log(
                    (np.exp(log_likelihoods[jj, ii] - max_pow)\
                        + np.exp(self.global_bias - max_pow))
                        *self.prior[label]
                ) + max_pow
                
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