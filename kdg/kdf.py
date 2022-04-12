from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.stats import multivariate_normal
import warnings

class kdf(KernelDensityGraph):

    def __init__(self, k = 1, kwargs={}):
        super().__init__()

        self.polytope_means = {}
        self.polytope_cov = {}
        self.polytope_cardinality = {}
        self.polytope_mean_cov = {}
        self.prior = {}
        self.bias = {}
        self.total_samples_this_label = {}
        self.global_bias = 0
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

        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_cov[label] = []
            self.polytope_cardinality[label] = []

            X_ = X[np.where(y==label)[0]]
            self.total_samples_this_label[label] = X_.shape[0]
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
                location = np.average(X_tmp, axis=0, weights=scales)
                X_tmp -= location

                '''covariance_model = LedoitWolf(assume_centered=True)
                covariance_model.fit(X_tmp)'''

                covariance = np.average(X_tmp**2, axis=0, weights=scales)
                self.polytope_means[label].append(
                    location
                )
                self.polytope_cov[label].append(
                    covariance
                )

            ## calculate bias for each label
            likelihoods = np.zeros(
                (np.size(X_,0)),
                dtype=float
            )
            for polytope_idx,_ in enumerate(self.polytope_means[label]):
                likelihoods += np.nan_to_num(self._compute_pdf(X_, label, polytope_idx))

            likelihoods /= total_samples_this_label
            self.bias[label] = np.min(likelihoods)/(self.k*total_samples_this_label)

        self.global_bias = min(self.bias.values())
        self.is_fitted = True
        
    def _compute_pdf_1d(self, X, label, polytope_idx, dim):
        mean_1d = self.polytope_means[label][polytope_idx][dim]
        var_1d = self.polytope_cov[label][polytope_idx][dim]

        likelihood = np.exp(-(X-mean_1d)**2/(2*var_1d))/(np.sqrt(2*np.pi*var_1d))

        return likelihood

    def _compute_pdf(self, X, label, polytope_idx):
        
        likelihood = np.ones(X.shape[0], dtype=float)
        for ii in range(self.feature_dim):
            likelihood *= self._compute_pdf_1d(
                    X[:,ii], label, polytope_idx, ii
            )

        likelihood = likelihood*(self.polytope_cardinality[label][polytope_idx]/self.total_samples_this_label[label])
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
            for polytope_idx,_ in enumerate(self.polytope_means[label]):
                likelihoods[:,ii] += self.prior[label] * np.nan_to_num(self._compute_pdf(X, label, polytope_idx))

            likelihoods[:,ii] += self.global_bias

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
