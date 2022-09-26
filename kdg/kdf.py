from numpy import min_scalar_type
from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.stats import multivariate_normal
import warnings
from sklearn.covariance import MinCovDet, fast_mcd, GraphicalLassoCV, LedoitWolf, EmpiricalCovariance, OAS, EllipticEnvelope, log_likelihood
warnings.filterwarnings("ignore")

class kdf(KernelDensityGraph):

    def __init__(self, k = 1, kwargs={}):
        super().__init__()

        self.polytope_means = []
        self.polytope_cov = []
        self.polytope_cardinality = {}
        self.total_samples_this_label = {}
        self.polytope_mean_cov = {}
        self.prior = {}
        self.bias = {}
        self.global_bias = -np.inf
        self.kwargs = kwargs
        self.k = k
        self.is_fitted = False

    def fit(self, X, y, epsilon=1e-8, alpha=0.5):
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
        self.max_val = np.max(X, axis=0) 
        self.min_val = np.min(X, axis=0)

        X = (X-self.min_val)/(self.max_val-self.min_val+1e-8)
        
        self.labels = np.unique(y)
        self.rf_model = rf(**self.kwargs).fit(X, y)
        self.feature_dim = X.shape[1]   

        ### change code to calculate one kernel per polytope
        for label in self.labels:
            self.polytope_cardinality[label] = []
            self.total_samples_this_label[label] = len(
                    np.where(y==label)[0]
                )
            self.prior[label] = self.total_samples_this_label[label]/X.shape[0]


        predicted_leaf_ids_across_trees = np.array(
                [tree.apply(X) for tree in self.rf_model.estimators_]
                ).T

        polytopes = np.unique(
                predicted_leaf_ids_across_trees, axis=0
            )
        total_polytopes = len(polytopes)
            
        for polytope in range(total_polytopes):
            matched_samples = np.sum(
                    predicted_leaf_ids_across_trees == polytopes[polytope],
                    axis=1
                )
            
            scales = matched_samples/np.max(matched_samples)
            #X_tmp = X[idx].copy()
            idx_with_scale_1 = np.where(
                    scales==1
                )[0]
            idx_with_scale_alpha = np.where(
                    scales>alpha
                )[0]

            location = np.mean(X[idx_with_scale_1], axis=0)
            X_tmp = X[idx_with_scale_alpha].copy() - location
            covariance = np.average(X_tmp**2+epsilon, axis=0, weights=scales[idx_with_scale_alpha])
            self.polytope_cov.append(covariance)
            self.polytope_means.append(location)

            y_tmp = y[idx_with_scale_1]
            for label in self.labels:      
                self.polytope_cardinality[label].append(
                    len(np.where(y_tmp==label)[0])
                )

        #########################################################
        for label in self.labels:
            ## calculate bias for each label
            idx = np.where(y==label)[0]
            likelihoods = np.zeros(
                (np.size(X[idx],0)),
                dtype=float
            )
            for polytope_idx,_ in enumerate(self.polytope_means):
                if self.polytope_cardinality[label][polytope_idx] != 0:
                    likelihoods += self._compute_log_likelihood(X[idx], label, polytope_idx)

            #likelihoods -= np.log(self.total_samples_this_label[label]
            self.bias[label] = np.min(likelihoods) - np.log(self.k*self.total_samples_this_label[label])

        self.global_bias = min(self.bias.values())
        self.is_fitted = True
        

    def _compute_log_likelihood_1d(self, X, location, variance):        
        return -(X-location)**2/(2*variance) - .5*np.log(2*np.pi*variance)

    def _compute_log_likelihood(self, X, label, polytope_idx):
        if self.polytope_cardinality[label][polytope_idx] == 0:
            return -np.inf*np.ones(X.shape[0], dtype = float)

        polytope_mean = self.polytope_means[polytope_idx]
        polytope_cov = self.polytope_cov[polytope_idx]
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
        X = (X-self.min_val)/(self.max_val-self.min_val+ 1e-8)

        log_likelihoods = np.zeros(
            (np.size(X,0), len(self.labels)),
            dtype=float
        )
        
        for ii,label in enumerate(self.labels):
            total_polytope = len(self.polytope_means)
            tmp_ = np.zeros((X.shape[0],total_polytope), dtype=float)

            for polytope_idx,_ in enumerate(self.polytope_means):
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
            pow_exp = np.nan_to_num(
                max_pow.reshape(-1,1)@np.ones((1,total_polytope), dtype=float)
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