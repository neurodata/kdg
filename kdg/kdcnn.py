from .base import KernelDensityGraph
from sklearn.utils.validation import check_array, check_X_y
import numpy as np
from tensorflow.keras.models import Model
from joblib import Parallel, delayed
import multiprocessing
from tensorflow.keras import backend as bknd
import tensorflow as tf
from scipy.spatial.distance import cdist as dist
from tqdm import tqdm
import pickle
import os
import gc
from joblib.externals.loky import get_reusable_executor
from .utils import get_ace

class kdcnn(KernelDensityGraph):
   def __init__(
       self,
       network,
       output_layer='flatten'
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
       self.output_layer = output_layer

 
       # total number of layers in the NN
       self.total_layers = 1
 
       # get the sizes of each layer
       self.network_shape = []
       ii = 0
       while self.output_layer not in self.network.layers[ii].name:
           ii += 1
       self.output_layer_id = ii

       for layer in network.layers[ii+1:]:
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

   def _get_layer_output(self, X, layer_name):
       layer_output = self.network.get_layer(layer_name).output
       intermediate_model = tf.keras.models.Model(inputs=self.network.input,outputs=layer_output)

       return intermediate_model.predict(X)

   def _get_polytope_ids(self, X):
       total_samples = X.shape[0]
       array_shape = [-1]
       array_shape.extend(
                list(self.network.get_layer(
                        self.output_layer
                    ).output.shape[1:]
                )
       )
       X = X.reshape(array_shape)
       outputs = []
       
       inp = self.network.layers[self.output_layer_id].output
       activation = []
       for layer in self.network.layers[self.output_layer_id+1:]:
           if 'activation' in layer.name:
               outputs.append(layer.output)

       # add the final layer
       outputs.append(layer.output)

       functor = bknd.function(inp, outputs)
       layer_outs = functor(X)
       
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
   
   def _compute_geodesic(self, polytope_id_test, polytope_ids, batch=-1):
       if batch == -1:
           batch = multiprocessing.cpu_count() if multiprocessing.cpu_count()<50 else 50

       total_layers = len(self.network_shape)
       total_test_samples = len(polytope_id_test)
       total_polytopes = len(polytope_ids)
       id_thresholds = np.zeros(total_layers+1,dtype=int)
       id_thresholds[1:] = np.cumsum(self.network_shape)

       sample_per_batch = total_test_samples//batch
       
       print("Calculating Geodesic...")
       w = np.ones((total_test_samples, total_polytopes), dtype=float)
       indx = [jj*sample_per_batch for jj in range(batch+1)]
       if indx[-1]<total_test_samples:
           indx.append(
               total_test_samples
           )
       for ii in tqdm(range(total_layers)):
           
            w_ = 1-np.array(Parallel(n_jobs=batch, prefer="threads")(
                            delayed(dist)(
                                        polytope_id_test[indx[jj]:indx[jj+1],id_thresholds[ii]:id_thresholds[ii+1]],
                                        polytope_ids[:,id_thresholds[ii]:id_thresholds[ii+1]],
                                        'hamming'
                                    ) for jj in range(len(indx)-1)
                            )
                )
            get_reusable_executor().shutdown(wait=True)
            gc.collect()
            w_ = np.concatenate(w_, axis=0) 
              
            w = w*w_
            del w_
           
           
       return 1 - w

   def _compute_euclidean(self, test_X):
       total_samples = test_X.shape[0]
       total_polytopes = len(self.polytope_means)
       return dist(
           test_X.reshape(total_samples,-1),
           np.array(self.polytope_means).reshape(total_polytopes,-1)
        )
   
   def _reset_param(self):
       for label in self.labels:
           self.polytope_cardinality[label] = []
           self.total_samples_this_label[label] = 0 


   def fit(self, X, y, X_val=None, y_val=None, epsilon=1e-6, batch=1, n_jobs=-1, k=None):
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
       X = self._get_layer_output(
                X,
                self.output_layer
           )
       self.labels = np.unique(y)
       self.total_samples = X.shape[0]
       self.feature_dim = np.product(X.shape[1:])
       self.global_bias = - 1e100 

       idx_with_label = {}
       for label in self.labels:
           self.polytope_cardinality[label] = []
           self.total_samples_this_label[label] = 0 
           idx_with_label[label] = np.where(y==label)[0]
           self.prior[label] = len(
                    idx_with_label[label]
                )/X.shape[0]
            
 
       # get polytope ids and unique polytope ids
       batchsize = self.total_samples//batch
       polytope_ids = self._get_polytope_ids( 
                            X[:batchsize]
                        )
       indx_X2 = np.inf
       for ii in range(1,batch):
           #print("doing batch ", ii)
           indx_X1 = ii*batchsize
           indx_X2 = (ii+1)*batchsize
           polytope_ids = np.concatenate(
               (polytope_ids,
               self._get_polytope_ids(
                   X[indx_X1:indx_X2]
                   )
                ),
               axis=0
           )
      
       if indx_X2 < X.shape[0]:
           polytope_ids = np.concatenate(
                   (polytope_ids,
                   self._get_polytope_ids(
                       X[indx_X2:]
                       )
                    ),
                   axis=0
               )

       w = 1- self._compute_geodesic(polytope_ids, polytope_ids, batch=n_jobs)
           
       used = []
       print('Fitting data!')

       #k = int(np.ceil(np.sqrt(self.total_samples)))
       polytope_filtered_id = []
       for ii in range(self.total_samples):
           if ii in used:
               continue
           
           polytope_filtered_id.append(ii)
           scales = w[ii,:].copy()
           
           #scale_indx_to_consider = np.where(scales>k)[0]
               
           idx_with_scale_1 = np.where(
                   scales>.9999999
               )[0]
           used.extend(idx_with_scale_1)
            
           location = np.mean(
               X[idx_with_scale_1],
               axis=0
               )
           X_tmp = X[idx_with_scale_1].copy() - location
           covariance = np.mean(
                        X_tmp**2+epsilon, 
                        axis=0
                    )
           self.polytope_cov.append(covariance)
           self.polytope_means.append(location)   
       
       self.global_bias = self.global_bias - np.log10(self.total_samples)
       ### cross validate for k
       def _count_polytope_cardinality(weight):
            for label in self.labels:     
               self.polytope_cardinality[label].append(
                    np.sum(weight[idx_with_label[label]])
                )
               self.total_samples_this_label[label] += self.polytope_cardinality[label][-1]

       
       def _get_likelihoods(min_id):
            log_likelihoods = np.zeros(
                (np.size(X_val,0), len(self.labels)),
                dtype=float
            )
            for ii,label in enumerate(self.labels):
                for jj in range(X_val.shape[0]):
                    log_likelihoods[jj, ii] = self._compute_log_likelihood(X_val[jj], label, min_id[jj])
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

            return proba


       if k == None:
            X_val = self._get_layer_output(
                X_val,
                self.output_layer
            )
            val_id = self._get_polytope_ids(X_val)
            distance = self._compute_geodesic(val_id, polytope_ids[polytope_filtered_id])
            min_dis_id = np.argmin(distance,axis=1)
            
            #k = int(np.ceil(np.sqrt(self.total_samples))
            min_ece = 1
            max_acc = 0
            for _ in range(2):
                if k==None:
                    k_ = np.arange(3,11,1)
                else:
                    k_ = np.arange(k-.6,k+.5,.1)
                for tmp_k in k_:
                    used = []
                    for ii in range(self.total_samples):

                        if ii in used:
                            continue

                        scales = w[ii,:].copy()
                        idx_with_scale_1 = np.where(
                                    scales>.9999999
                                )[0]
                        used.extend(idx_with_scale_1)
                        
                        _count_polytope_cardinality(scales**tmp_k)
                    
                    prob = _get_likelihoods(min_dis_id)
                    
                    accuracy = np.mean(np.argmax(prob,axis=1)==y_val.ravel())
                    ece = get_ace(prob, y_val.ravel())
                    print(k, ece, accuracy)
                    if ece < min_ece:
                        min_ece = ece
                        #max_acc = accuracy
                        k = tmp_k
                    # else:
                    #     break
                    
                    
                        #print('taken')
                    self._reset_param()
            
       used = []
       for ii in range(self.total_samples):
            if ii in used:
                continue
            
            scales = w[ii,:].copy()
            idx_with_scale_1 = np.where(
                        scales>.9999999
                    )[0]
            used.extend(idx_with_scale_1)

            _count_polytope_cardinality(scales**k)

             
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

   def predict_proba(self, X, distance = 'Euclidean', return_likelihood=False, n_jobs=-1):
        r"""
        Calculate posteriors using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        #X = check_array(X)
        X = self._get_layer_output(
                X,
                self.output_layer
           )
        total_polytope = len(self.polytope_means)
        log_likelihoods = np.zeros(
            (np.size(X,0), len(self.labels)),
            dtype=float
        )
        
        print('Calculating distance')
        if distance == 'Euclidean':
            distance = self._compute_euclidean(X)
            polytope_idx = np.argmin(distance, axis=1)
        elif distance == 'Geodesic':
            total_polytope = len(self.polytope_means)
            batch = total_polytope//1000 + 1
            batchsize = total_polytope//batch
            polytope_ids = self._get_polytope_ids(
                    np.array(self.polytope_means[:batchsize])
                ) 

            indx_X2 = np.inf
            for ii in range(1,batch):
                #print("doing batch ", ii)
                indx_X1 = ii*batchsize
                indx_X2 = (ii+1)*batchsize
                polytope_ids = np.concatenate(
                    (polytope_ids,
                    self._get_polytope_ids(
                    np.array(self.polytope_means[indx_X1:indx_X2])
                    )),
                    axis=0
                )
            
            if indx_X2 < len(self.polytope_means):
                polytope_ids = np.concatenate(
                        (polytope_ids,
                        self._get_polytope_ids(
                    np.array(self.polytope_means[indx_X2:]))),
                        axis=0
                    )

            total_sample = X.shape[0]
            batch = total_sample//1000 + 1
            batchsize = total_sample//batch
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
                        self._compute_geodesic(
                            test_ids[indx[ii]:indx[ii+1]],
                            polytope_ids,
                            batch=n_jobs
                        ), axis=1
                    ))
                )
        else:
            raise ValueError("Unknown distance measure!")
        
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