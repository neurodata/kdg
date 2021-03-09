from .base import KernelDensityGraph
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.stats import multivariate_normal
from numba import jit

class kdf(KernelDensityGraph):

    def __init__(self, kwargs={}):
        super().__init__()
        self.polytope_means = {}
        self.polytope_vars = {}
        self.polytope_cardinality = {}
        self.kwargs = kwargs

    @jit(nopython=True)
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.labels = np.unique(y)
        self.rf_model = rf(**self.kwargs).fit(X, y)

        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_vars[label] = []
            self.polytope_cardinality[label] = []

        predicted_leaf_ids_across_trees = [tree.apply(X) for tree in self.rf_model.estimators_]

        for polytopes_in_a_tree in predicted_leaf_ids_across_trees:
            for polytope in np.unique(polytopes_in_a_tree):
                for label in self.labels:
                    polytope_label_idx = np.where((y==label) & (polytopes_in_a_tree==polytope))
                    
                    if polytope_label_idx[0].size == 0:
                        continue
                    
                    self.polytope_means[label].append(
                        np.mean(
                            X[polytope_label_idx],
                            axis=0
                        )
                    )
                    self.polytope_vars[label].append(
                        np.var(
                            X[polytope_label_idx],
                            axis=0
                        )
                    )
                    self.polytope_cardinality[label].append(
                        len(polytope_label_idx)
                    )

    def _compute_pdf(self, X, label, polytope_idx):
        polytope_mean = self.polytope_means[label][polytope_idx]
        polytope_cov = np.eye(len(self.polytope_vars[label][polytope_idx]), dtype=float)*self.polytope_vars[label][polytope_idx]
        polytope_cardinality = self.polytope_cardinality[label]

        var = multivariate_normal(
            mean=polytope_mean, 
            cov=polytope_cov, 
            allow_singular=True
            )

        likelihood = var.pdf(X)*polytope_cardinality[polytope_idx]/np.sum(polytope_cardinality)
        return likelihood

    @jit(nopython=True)
    def predict_proba(self, X):
        X = check_array(X)

        likelihoods = np.zeros(
            (np.size(X,0), len(self.labels)),
            dtype=float
        )
        
        for ii,label in enumerate(self.labels):
            for polytope_idx,_ in enumerate(self.polytope_cardinality[label]):
                likelihoods[:,ii] += self._compute_pdf(X, label, polytope_idx)

        proba = (likelihoods.T/np.sum(likelihoods,axis=1)).T
        return proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis = 1)
