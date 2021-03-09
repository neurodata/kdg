from .base KernelDensityGraph
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np

class kdf(KernelDensityGraph):

    def __init__(self, kwargs={}):
        super().__init__()
        self.polytope_means = {}
        self.polytope_vars = {}
        self.kwargs = kwargs

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.labels = np.unique(y)
        self.rf_model = rf(**self.kwargs).fit(X, y)

        for label in self.labels:
            self.polytope_means[label] = []

        predicted_leaf_ids_across_trees = [tree.apply(X) for tree in self.rf_model.estimators_]

        for polytopes_in_a_tree in predicted_leaf_ids_across_trees:
            for label in self.labels:
                for polytope in np.unique(polytopes_in_a_tree):
                    polytope_label_idx = np.where((y==label) & (polytopes_in_a_tree==polytope))
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

