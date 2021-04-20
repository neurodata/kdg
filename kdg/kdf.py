from .base import KernelDensityGraph
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf

class kdf(KernelDensityGraph):

    def __init__(self, kwargs={}):
        super().__init__()
        self.polytope_means = {}
        self.polytope_vars = {}
        self.polytope_cardinality = {}
        self.polytope_mean_cov = {}
        self.kwargs = kwargs

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
        X, y = check_X_y(X, y)
        self.labels = np.unique(y)
        self.rf_model = rf(**self.kwargs).fit(X, y)

        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_vars[label] = []
            self.polytope_cardinality[label] = []
            self.polytope_mean_cov[label] = []

            X_ = X[np.where(y==label)]
            predicted_leaf_ids_across_trees = np.array(
                [tree.apply(X_) for tree in self.rf_model.estimators_]
                ).T
            total_polytopes_this_label = len(X_)

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

                cov = LedoitWolf().fit(X_[idx])
                self.polytope_means[label].append(
                    cov.location_
                )
                
                self.polytope_vars[label].append(
                    cov.covariance_
                )
                self.polytope_cardinality[label].append(len(idx))

        for label in self.labels:
            if np.sum(self.polytope_cardinality[label]) == 0:
                self.polytope_mean_cov[label] = 0
            else:
                self.polytope_mean_cov[label] = np.average(
                    self.polytope_vars[label],
                    weights = self.polytope_cardinality[label],
                    axis = 0
                    )

    def _compute_pdf(self, X, label, polytope_idx):
        polytope_mean = self.polytope_means[label][polytope_idx]
        polytope_cov = self.polytope_mean_cov[label]
        polytope_cardinality = self.polytope_cardinality[label]

        var = multivariate_normal(
            mean=polytope_mean, 
            cov=polytope_cov, 
            allow_singular=True
            )

        likelihood = var.pdf(X)*polytope_cardinality[polytope_idx]/np.sum(polytope_cardinality)
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
            for polytope_idx,_ in enumerate(self.polytope_cardinality[label]):
                likelihoods[:,ii] += self._compute_pdf(X, label, polytope_idx)

        proba = (likelihoods.T/(np.sum(likelihoods,axis=1)+1e-200)).T
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