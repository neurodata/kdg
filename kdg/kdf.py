from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.stats import multivariate_normal
import warnings
from sklearn.covariance import MinCovDet, fast_mcd, GraphicalLassoCV, LedoitWolf, EmpiricalCovariance, OAS, EllipticEnvelope

class kdf(KernelDensityGraph):

    def __init__(self, k = 1, kwargs={}):
        super().__init__()

        self.polytope_means = {}
        self.polytope_cov = {}
        self.polytope_cardinality = {}
        self.polytope_mean_cov = {}
        self.prior = {}
        self.bias = {}
        self.global_bias = np.zeros(self.feature_dim, dtype=float)
        self.kwargs = kwargs
        self.k = k
        self.is_fitted = False

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
        self.rf_model = rf(**self.kwargs).fit(X, y)
        self.feature_dim = X.shape[1]
        self.total_training_points = X.shape[0]
        min_bias = np.ones(self.feature_dim, dtype=float)*1e15

        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_cov[label] = []
            self.polytope_cardinality[label] = []

            X_ = X[np.where(y==label)[0]]
            predicted_leaf_ids_across_trees = np.array(
                [tree.apply(X_) for tree in self.rf_model.estimators_]
                ).T
            polytopes, polytope_count = np.unique(
                predicted_leaf_ids_across_trees, return_inverse=True, axis=0
            )
            self.polytope_cardinality[label].extend(
                polytope_count
            )
            total_polytopes_this_label = len(polytopes)
            total_samples_this_label = X_.shape[0]
            self.prior[label] = total_samples_this_label/X.shape[0]

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
                location_ = np.average(X_tmp, axis=0, weights=scales)
                X_tmp -= location_
                
                #sqrt_scales = np.sqrt(scales).reshape(-1,1) @ np.ones(self.feature_dim).reshape(1,-1)
                #X_tmp *= sqrt_scales

                covariance = np.average(X_tmp**2, weights=scales)
                self.polytope_means[label].append(
                    location_
                )
                self.polytope_cov[label].append(
                    covariance
                )

            ## calculate bias for each label
            #likelihoods = np.zeros(self.feature_dim, dtype=float)
            for d in range(self.feature_dim):
                likelihoods = 0
                for polytope_idx,_ in enumerate(self.polytope_means[label]):
                    likelihoods += np.nan_to_num(self._compute_pdf_1d(X_[:,d], label, polytope_idx, d))

                if likelihoods < min_bias[d]:
                    min_bias[d] = likelihoods

        self.global_bias = min_bias/(self.k*np.log(self.total_training_points))
        self.is_fitted = True
        
    
    def _compute_pdf_1d(self, X, label, polytope_idx, dim):
        mean_1d = self.polytope_means[label][polytope_idx][dim]
        var_1d = self.polytope_cov[label][polytope_idx][dim]
        bias_1d = self.global_bias[dim]

        likelihood = (bias_1d + \
                np.exp(-(X-mean_1d)**2/(2*var_1d))/(np.sqrt(2*np.pi*var_1d)))

        likelihood = likelihood*(self.polytope_cardinality[label][polytope_idx]/self.total_training_points)**(1/self.feature_dim)
        
    def _compute_pdf(self, X, label, polytope_idx):
        
        likelihood = np.ones(X.shape[0], dtype=float)
        for ii in range(self.feature_dim):
            likelihood *= self._compute_pdf_1d(
                    X[:,ii], label, polytope_idx, ii
            )

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

        likelihoods = np.zeros(
            (np.size(X,0), len(self.labels)),
            dtype=float
        )
        
        for ii,label in enumerate(self.labels):
            total_polytopes = len(self.polytope_means[label])
            for polytope_idx,_ in enumerate(self.polytope_means[label]):
                likelihoods[:,ii] += self.prior[label] * np.nan_to_num(self._compute_pdf(X, label, polytope_idx))

        proba = (likelihoods.T/np.sum(likelihoods,axis=1)).T
        
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