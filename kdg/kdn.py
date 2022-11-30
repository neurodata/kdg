from matplotlib.pyplot import sca
from .base import KernelDensityGraph
from sklearn.utils.validation import check_array, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf
from tensorflow.keras.models import Model
from joblib import Parallel, delayed
from tensorflow.keras import backend as bknd
from scipy.sparse import csr_matrix, vstack
 
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
       polytope_ids = csr_matrix(np.concatenate(activation, axis=1))
      
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
       self.total_samples = X.shape[0]
       self.feature_dim = np.product(X.shape[1:])
       self.global_bias = -10**(np.sqrt(self.feature_dim))
      
       for label in self.labels:
           self.polytope_cardinality[label] = []
           self.total_samples_this_label[label] = len(
                   np.where(y==label)[0]
               )
           self.prior[label] = self.total_samples_this_label[label]/X.shape[0]
 
       # get polytope ids and unique polytope ids
       batchsize = self.total_samples//batch
       polytope_ids = self._get_polytope_ids(X[:batchsize])
 
       for ii in range(1,batch):
           #print("doing batch ", ii)
           indx_X1 = ii*batchsize
           indx_X2 = (ii+1)*batchsize
           polytope_ids = vstack(
               [polytope_ids,
               self._get_polytope_ids(X[indx_X1:indx_X2])]
           )
      
       if indx_X2 < X.shape[0]:
           polytope_ids = vstack(
                   [polytope_ids,
                   self._get_polytope_ids(X[indx_X2:])]
               )
 
       #create a matrix with mutual weights
       w = np.zeros((self.total_samples, self.total_samples), dtype=float)
       
       #define worker for parallel processing
       def worker(activation1, activation2, network_shape):
            activation1 = activation1.toarray()[0]
            activation2 = activation2.toarray()[0]
            end_node = 0
            scale = 0
            
            for layer in network_shape:
                end_node += layer
                act1_indx = set(np.where(
                      activation1[end_node-layer:end_node] == 1
                      )[0])
                act2_indx = set(np.where(
                      activation2[end_node-layer:end_node] == 1
                      )[0])
                scale += np.log(len
                      (
                          act1_indx.intersection(act2_indx)
                      )
                  )
                scale -= np.log(len(act1_indx.union(act2_indx)))
            return np.exp(scale)
        
       iterseq = []
       for ii in range(self.total_samples):
            for jj in range(ii, self.total_samples):
                iterseq.append((ii,jj))
        
       scales = Parallel(n_jobs=-1,verbose=1)(
                    delayed(worker)(
                   polytope_ids[ii],
                   polytope_ids[jj],
                   self.network_shape
               ) for (ii,jj) in iterseq
           )

       indx = 0
       for (ii,jj) in iterseq:
            w[ii,jj] = scales[indx]
            w[jj,ii] = scales[indx]
            indx += 1
 
       del polytope_ids #delete high memory using variables
       print(w)
       used = []
       for ii in range(self.total_samples):
           if ii in used:
               continue
          
           scales = w[ii,:]
           idx_with_scale_1 = np.where(
                   scales>.99999999999
               )[0]
           idx_with_scale_alpha = np.where(
                   scales>0
               )[0]
 
           used.extend(idx_with_scale_1)
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
 
       self.global_bias = self.global_bias -np.log(X.shape[0])
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

