from numpy import min_scalar_type
from .base import KernelDensityGraph
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
from .utils import get_ece
import warnings
warnings.filterwarnings("ignore")

class kdf(KernelDensityGraph):

    def __init__(self, rf_model):
        super().__init__()

        self.polytope_means = []
        self.polytope_cov = []
        self.polytope_cardinality = {}
        self.total_samples_this_label = {}
        self.prior = {}
        self.rf_model = rf_model
        self.is_fitted = False

    def _get_polytope_ids(self, X):
       predicted_leaf_ids_across_trees = np.array(
                [tree.apply(X) for tree in self.rf_model.estimators_]
                ).T

       return predicted_leaf_ids_across_trees
     
    def _compute_geodesic(self, polytope_id_test, polytope_ids):
       total_samples = polytope_id_test.shape[0]
       total_polytopes = polytope_ids.shape[0]
       total_trees = polytope_id_test.shape[1]
       w = np.zeros((total_samples, total_polytopes), dtype=float)
       for ii in range(total_samples):
           matched_samples = np.sum(
                    polytope_ids == polytope_id_test[ii],
                    axis=1
                )
           w[ii,:] = (matched_samples/total_trees) + 1e-30
       return 1 - w

    def _compute_euclidean(self, test_X):
        total_samples = test_X.shape[0]
        total_polytopes = len(self.polytope_means)
        w = np.zeros((total_samples, total_polytopes), dtype=float)
        for ii in range(total_samples):
           w[ii,:] = np.sum(
               ((test_X[ii] - self.polytope_means)**2).reshape(total_polytopes,-1),
               axis=1
           )
        return w

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
        if self.is_fitted:
            raise ValueError(
                "Model already fitted!"
            )
            return

        X, y = check_X_y(X, y)
        X = X.astype('double')

        self.total_samples = X.shape[0]
        self.labels = np.unique(y)
        self.feature_dim = X.shape[1] 
        self.global_bias = - 1e100 
        
        ### change code to calculate one kernel per polytope
        idx_with_label = {}
        for label in self.labels:
            self.polytope_cardinality[label] = []
            self.total_samples_this_label[label] = 0 
            idx_with_label[label] = np.where(y==label)[0]
            self.prior[label] = len(
                    idx_with_label[label]
                )/X.shape[0]
        
        polytope_ids = self._get_polytope_ids(X)
        w = 1 - self._compute_geodesic(
            polytope_ids,
            polytope_ids
        )
        
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
                val_id = self._get_polytope_ids(X_val)
                distance = self._compute_geodesic(val_id, polytope_ids[polytope_filtered_id])
                min_dis_id = np.argmin(distance,axis=1)
                
                #k = int(np.ceil(np.sqrt(self.total_samples))
                min_ece = 1
                max_acc = 0
                for _ in range(2):
                    if k==None:
                        k_ = np.arange(1,20,2)
                    else:
                        k_ = np.arange(k,k+2,.2)
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
                        
                        #accuracy = np.mean(np.argmax(prob,axis=1)==y_val)
                        ece = get_ece(prob, y_val)
                        #print(k, ece)
                        if ece < min_ece:
                            min_ece = ece
                            #max_acc = accuracy
                            k = tmp_k
                        
                        
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
        
    def _compute_distance(self, X, polytope):
        return np.sum(
               (X - self.polytope_means[polytope])**2/self.polytope_cov[polytope],
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
            distance = self._compute_geodesic(
                self._get_polytope_ids(X),
                self._get_polytope_ids(
                    np.array(
                    self.polytope_means
                    )
                )
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
        Perform inference using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        
        return np.argmax(self.predict_proba(X, distance=distance), axis = 1)