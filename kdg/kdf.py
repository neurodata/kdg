from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.stats import multivariate_normal
import warnings
from sklearn.covariance import MinCovDet, fast_mcd, GraphicalLassoCV, LedoitWolf, EmpiricalCovariance, OAS, EllipticEnvelope

class kdf(KernelDensityGraph):

    def __init__(self, criterion='bic', kwargs={}):
        super().__init__()

        self.polytope_means = {}
        self.polytope_cov = {}
        self.polytope_cardinality = {}
        self.polytope_mean_cov = {}
        self.scale_sum = {}
        self.criterion = criterion
        self.kwargs = kwargs
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
        feature_dim = X.shape[1]

        polytope_full_cov = {}
        polytope_diag_cov = {}
        polytope_mean_cov = {}

        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_cov[label] = []
            polytope_full_cov[label] = []
            polytope_diag_cov[label] = []
            polytope_mean_cov[label] = [np.zeros((feature_dim,feature_dim), dtype=float)]
            scale_sum = 0

            X_ = X[np.where(y==label)[0]]
            predicted_leaf_ids_across_trees = np.array(
                [tree.apply(X_) for tree in self.rf_model.estimators_]
                ).T
            _, polytope_idx = np.unique(
                predicted_leaf_ids_across_trees, return_inverse=True, axis=0
            )
            total_polytopes_this_label = np.max(polytope_idx)+1

            for polytope in range(total_polytopes_this_label):
                matched_samples = np.sum(
                    predicted_leaf_ids_across_trees == predicted_leaf_ids_across_trees[polytope],
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
                
                sqrt_scales = np.sqrt(scales).reshape(-1,1) @ np.ones(feature_dim).reshape(1,-1)
                X_tmp *= sqrt_scales

                covariance_model = LedoitWolf(assume_centered=True)
                covariance_model.fit(X_tmp)

                self.polytope_means[label].append(
                    location_
                )

                polytope_full_cov[label].append(
                    covariance_model.covariance_*len(idx)/sum(scales)
                )
                polytope_diag_cov[label].append(
                    np.eye(feature_dim)*np.diag(
                        covariance_model.covariance_*len(idx)/sum(scales)
                    )
                )
                
                polytope_mean_cov[label][0] += covariance_model.covariance_*len(idx)
                scale_sum += sum(scales)
            
            polytope_mean_cov[label][0] /= scale_sum

            # see the best fit according to the criterion     
            self.polytope_cov[label] = polytope_full_cov[label]
            if self.criterion == 'aic':
                constraint = self.aic(X_, label, total_polytopes_this_label*feature_dim*(feature_dim+1)/2)
            else:
                constraint = self.bic(X_, label, total_polytopes_this_label*feature_dim*(feature_dim+1)/2)

            method = 'full'
            
            self.polytope_cov[label] = polytope_diag_cov[label]
            if self.criterion == 'aic':
                constraint_ = self.aic(X_, label, total_polytopes_this_label*feature_dim)
            else:
                constraint_ = self.bic(X_, label, total_polytopes_this_label*feature_dim)

            if constraint_ < constraint:
                method = 'diag'
                constraint = constraint_
            
            self.polytope_cov[label] = polytope_mean_cov[label]
            if self.criterion == 'aic':
                constraint_ = self.aic(X_, label, feature_dim*(feature_dim+1)/2)
            else:
                constraint_ = self.bic(X_, label, feature_dim*(feature_dim+1)/2)

            if constraint_ < constraint:
                method = 'tied'
                constraint = constraint_
            
            if method == 'full':
                self.polytope_cov[label] = polytope_full_cov[label]
            elif method == 'diag':
                self.polytope_cov[label] = polytope_diag_cov[label]
                     
        self.is_fitted = True
         
            
    def _compute_pdf(self, X, label, polytope_idx):
        polytope_mean = self.polytope_means[label][polytope_idx]

        if len(self.polytope_cov[label]) > 1:
            polytope_cov = self.polytope_cov[label][polytope_idx]
        else:
            polytope_cov = self.polytope_cov[label][0]

        var = multivariate_normal(
            mean=polytope_mean, 
            cov=polytope_cov, 
            allow_singular=True
            )

        likelihood = var.pdf(X)
        return likelihood

    def score(self, X, label):
        likelihood = 0
        for polytope_idx,_ in enumerate(self.polytope_means[label]):
                likelihood += np.nan_to_num(self._compute_pdf(X, label, polytope_idx))

        return likelihood

    def predict_proba(self, X):
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
            likelihoods[:,ii] = self.score(X, label)

        return likelihoods

    def aic(self, X, label, n_parameters):
        likelihood = self.score(X, label)
        loglikelihood = np.sum(
                np.log(
                likelihood[likelihood>0]
            )
        )

        return -2 * loglikelihood + 2 * n_parameters

    def bic(self, X, label, n_parameters):
        likelihood = self.score(X, label)
        loglikelihood = np.sum(
                np.log(
                likelihood[likelihood>0]
            )
        )

        return -2 * loglikelihood + 2 * n_parameters*np.log(
            X.shape[0]
        )

    def predict(self, X):
        r"""
        Perform inference using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        likelihoods = self.predict_proba(X)
        proba = (likelihoods.T/(np.sum(likelihoods,axis=1)+1e-100)).T

        return np.argmax(proba, axis = 1)