from matplotlib.pyplot import sca
from .base import KernelDensityGraph
from sklearn.utils.validation import check_array, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf
from tensorflow.keras.models import Model
from joblib import Parallel, delayed
from tensorflow.keras import backend as bknd
from scipy.spatial.distance import cdist as dist
from tqdm import tqdm

class kdn(KernelDensityGraph):
   def __init__(
       self,
       network
   ):
       r"""
       Parameters
       ----------
       network : tf.keras.Model()
           trained neural network model
       """
       super().__init__()
       self.polytope_means = []
       self.polytope_cov = []
       self.polytope_cardinality = {}
       self.total_samples_this_label = {}
       self.prior = {}
       self.network = network

 
       # total number of layers in the NN
       self.total_layers = 1
 
       # get the sizes of each layer
       self.network_shape = []
       for layer in network.layers:
           layer_name = layer.name
           #print(self.network_shape)
           if 'activation' in layer_name:
               self.total_layers += 1
               #print(layer.output_shape[1:],' fjeehe')
               self.network_shape.append(
                   np.product(
                       layer.output_shape[1:]
                   )
               )

       # add the final output layer
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

       # add the final layer
       outputs.append(layer.output)

       functor = bknd.function(inp, outputs)
       layer_outs = functor(X)
 
       activation = []
       for layer_out in layer_outs[:-1]:
           activation.append(
               (layer_out>0).astype('bool').reshape(total_samples, -1)
           )
        # add the last layer
       activation.append(
               (layer_outs[-1]>1/len(self.labels)).astype('bool').reshape(total_samples, -1)
           )
       polytope_ids = np.concatenate(activation, axis=1)
      
       return polytope_ids
   
   def _compute_geodesic(self, polytope_id_test, polytope_ids):
       total_layers = len(self.network_shape)
       id_thresholds = np.zeros(total_layers+1,dtype=int)
       id_thresholds[1:] = np.cumsum(self.network_shape)
       w = 1-np.array(Parallel(n_jobs=-1)(
            delayed(dist)(
                        polytope_id_test[:,id_thresholds[ii]:id_thresholds[ii+1]],
                        polytope_ids[:,id_thresholds[ii]:id_thresholds[ii+1]],
                        'hamming'
                    ) for ii in range(total_layers)
            ))    
       w = np.product(w,axis=0)

       return 1 - w

   def _compute_euclidean(self, test_X):
       total_samples = test_X.shape[0]
       total_polytopes = len(self.polytope_means)
       return dist(
           test_X.reshape(total_samples,-1),
           np.array(self.polytope_means).reshape(total_polytopes,-1)
        )

   def fit(self, X, y, k=1, epsilon=1e-6, batch=1, mul=1, n_jobs=-1):
       r"""
       Fits the kernel density forest.
       Parameters
       ----------
       X : ndarray
           Input data matrix.
       y : ndarray
           Output (i.e. response) data matrix.
       """
       X = X.astype('double') 
 
       self.labels = np.unique(y)
       self.total_samples = X.shape[0]
       self.feature_dim = np.product(X.shape[1:])
       self.global_bias = np.log(k) - 10**(self.feature_dim**(1/2)) 
       self.w = np.zeros(
                (
                    self.total_samples,
                    self.total_samples
                ),
                dtype=float
            )
       


       idx_with_label = {}
       for label in self.labels:
           self.polytope_cardinality[label] = []
           self.total_samples_this_label[label] = 0 
           idx_with_label[label] = np.where(y==label)[0]
           self.prior[label] = len(
                    idx_with_label[label]
                )/X.shape[0]
           idx_with_label[label] = np.where(
                    y == label
                )[0] 
 
       # get polytope ids and unique polytope ids
       batchsize = self.total_samples//batch
       polytope_ids = self._get_polytope_ids(X[:batchsize])
       indx_X2 = np.inf
       for ii in range(1,batch):
           #print("doing batch ", ii)
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
       
       total_layers = len(self.network_shape)
       id_thresholds = np.zeros(total_layers+1,dtype=int)
       id_thresholds[1:] = np.cumsum(self.network_shape)

       w = 1-np.array(Parallel(n_jobs=n_jobs,verbose=1)(
            delayed(dist)(
                        polytope_ids[:,id_thresholds[ii]:id_thresholds[ii+1]],
                        polytope_ids[:,id_thresholds[ii]:id_thresholds[ii+1]],
                        'hamming'
                    ) for ii in range(total_layers)
            ))    
       self.w = np.product(w,axis=0)
           
       used = []
       for ii in range(self.total_samples):
           if ii in used:
               continue
           scales = self.w[ii,:].copy()
           scales = scales**np.log2(self.total_samples*mul)
           
           idx_with_scale_1 = np.where(
                   scales>.9999999
               )[0]
           used.extend(idx_with_scale_1)
            
           location = np.mean(X[idx_with_scale_1], axis=0)
           X_tmp = X.copy() - location
           covariance = np.average(X_tmp**2+epsilon/np.sum(scales), axis=0, weights=scales)
           self.polytope_cov.append(covariance)
           self.polytope_means.append(location)
 
           for label in self.labels:     
               self.polytope_cardinality[label].append(
                    np.sum(scales[idx_with_label[label]])
                )
               self.total_samples_this_label[label] += self.polytope_cardinality[label][-1]
 
 
       self.global_bias = self.global_bias - np.log10(self.total_samples)
       self.is_fitted = True      

   def _compute_log_likelihood_1d(self, X, location, variance):        
        return -(X-location)**2/(2*variance) - .5*np.log(2*np.pi*variance)

   def _compute_log_likelihood(self, X, label, polytope_idx):
        polytope_mean = self.polytope_means[polytope_idx].reshape(-1)
        polytope_cov = self.polytope_cov[polytope_idx].reshape(-1)
        X = X.reshape(-1)
        likelihood = 0

        for ii in range(self.feature_dim):
            likelihood += self._compute_log_likelihood_1d(X[ii], polytope_mean[ii], polytope_cov[ii])
        
        likelihood += np.log(self.polytope_cardinality[label][polytope_idx]) -\
            np.log(self.total_samples_this_label[label])

        return likelihood

   def predict_proba(self, X, distance = 'Euclidean', return_likelihood=False):
        r"""
        Calculate posteriors using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        #X = check_array(X)
        
        total_polytope = len(self.polytope_means)
        log_likelihoods = np.zeros(
            (np.size(X,0), len(self.labels)),
            dtype=float
        )
        
        if distance == 'Euclidean':
            distance = self._compute_euclidean(X)
        elif distance == 'Geodesic':
            total_polytope = len(self.polytope_means)
            batch = total_polytope//1000 + 1
            batchsize = self.total_samples//batch
            polytope_ids = self._get_polytope_ids(self.polytope_means[:batchsize]) 

            indx_X2 = np.inf
            for ii in range(1,batch):
                #print("doing batch ", ii)
                indx_X1 = ii*batchsize
                indx_X2 = (ii+1)*batchsize
                polytope_ids = np.concatenate(
                    (polytope_ids,
                    self._get_polytope_ids(self.polytope_means[indx_X1:indx_X2])),
                    axis=0
                )
            
            if indx_X2 < len(self.polytope_means):
                polytope_ids = np.concatenate(
                        (polytope_ids,
                        self._get_polytope_ids(self.polytope_means[indx_X2:])),
                        axis=0
                    )

            total_sample = X.shape[0]
            batch = total_sample//1000 + 1
            batchsize = self.total_samples//batch
            test_ids = self._get_polytope_ids(X[:batchsize]) 

            indx_X2 = np.inf
            for ii in range(1,batch):
                #print("doing batch ", ii)
                indx_X1 = ii*batchsize
                indx_X2 = (ii+1)*batchsize
                test_ids = np.concatenate(
                    (test_ids,
                    self._get_polytope_ids(X[indx_X1:indx_X2])),
                    axis=0
                )
            
            if indx_X2 < X.shape[0]:
                test_ids = np.concatenate(
                        (test_ids,
                        self._get_polytope_ids(X[indx_X2:])),
                        axis=0
                    )
               
            
            distance = self._compute_geodesic(
                test_ids,
                polytope_ids
            )
        else:
            raise ValueError("Unknown distance measure!")
        
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

        if return_likelihood:
            likelihood = np.exp(log_likelihoods)

        log_likelihoods -= max_pow
        likelihoods = np.exp(log_likelihoods)

        total_likelihoods = np.sum(likelihoods, axis=1)

        proba = (likelihoods.T/total_likelihoods).T
        
        if return_likelihood:
            return proba, likelihood
        else:
            return proba 

   def predict(self, X, distance='Euclidean'):
        r"""
        Perform inference using the kernel density network.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        
        return np.argmax(self.predict_proba(X, distance=distance), axis = 1)

