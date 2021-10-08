#%%
import numpy as np

# %%
class random_tree():

    def __init__(self):
        self.children_left = []
        self.children_right = []
        self.feature = []
        self.threshold = []
        self.leaf_to_posterior = {}

    def _
    def fit(self, X, y):
        feature_dim = X.shape[1]
        
        