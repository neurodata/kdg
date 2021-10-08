#%%
import numpy as np
from joblib import Parallel, delayed
# %%
class random_tree():

    def __init__(self, max_depth=2000):
        self.children_left = {}
        self.children_right = {}
        self.feature = {}
        self.threshold = {}
        self.leaf_to_posterior = {}

    def fit(self, X, y):
        self.total_classes = len(np.unique(y))
        self.feature_dim = X.shape[1]
        self.node_count = 1

        def build_tree(X_, y_, node):

            if len(y_) == 0:
                return

            classes = np.unique(y_)
            if len(classes) == 1:
                self.node_count += 1
                posterior = np.zeros(self.total_classes, dtype=float)
                posterior[classes] = 1
                self.leaf_to_posterior[node] = posterior
                self.children_left[node] = -2
                self.children_right[node] = -2
                self.feature[node] = -2
                return 

            dim = np.random.choice(self.feature_dim)
            self.feature[node] = dim

            X_max, X_min = np.max(X_[:, dim]), np.min(X_[:, dim])
            threshold_ = np.random.uniform(X_min, X_max)

            left_child = self.node_count
            right_child = self.node_count + 1
            self.node_count += 2
            
            self.children_left[node] = left_child
            self.children_right[node] = right_child
            self.threshold[node] = threshold_
            idx_left = np.where(X_[:,dim]<threshold_)[0]
            idx_right = np.where(X_[:,dim]>=threshold_)[0]

            build_tree(X_[idx_left].copy(), y_[idx_left].copy(), left_child)
            build_tree(X_[idx_right].copy(), y_[idx_right], right_child)

        build_tree(X.copy(), y.copy(), 0)

    def predict_proba(self, X):

        def find_leaf(X, node):
            if self.children_left[node] == self.children_right[node]:
                return self.leaf_to_posterior[node]

            selected_feature = self.feature[node]
            threshold = self.threshold[node]

            if X[selected_feature] < threshold:
                return find_leaf(X, self.children_left[node])
            else:
                return find_leaf(X, self.children_right[node])

        return Parallel(n_jobs=-2)(
            delayed(find_leaf)(
                    X[i,:],0
                    ) for i in range(X.shape[0])
                )
# %%
