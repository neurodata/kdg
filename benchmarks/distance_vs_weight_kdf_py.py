# -*- coding: utf-8 -*-
# import modules
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rf 
from kdg.kdf import *
from kdg.utils import gaussian_sparse_parity, trunk_sim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# define the experimental setup
p = 20 # total dimensions of the data vector
p_star = 3 # number of signal dimensions of the data vector
'''sample_size = np.logspace(
        np.log10(10),
        np.log10(5000),
        num=10,
        endpoint=True,
        dtype=int
        )'''
sample_size = 5000 # sample size under consideration
n_test = 1000 # test set size
reps = 10 # number of replicates

df = pd.DataFrame()
reps_list = []
accuracy_kdn = []
accuracy_kdn_ = []
accuracy_nn = []
accuracy_nn_ = []
sample_list = []

nTrees = 100
compile_kwargs = {
    "n_estimators": nTrees
    }

def get_kdf_weights(X, y, forest):
    scales = []
    scale_idx = []
    X, y = check_X_y(X, y)
    labels = np.unique(y)

    for label in labels:
        X_ = X[np.where(y==label)[0]]
        predicted_leaf_ids_across_trees = np.array(
            [tree.apply(X_) for tree in forest.estimators_]
            ).T
        _, polytope_idx = np.unique(
            predicted_leaf_ids_across_trees, return_inverse=True, axis=0
        )
        total_polytopes_this_label = np.max(polytope_idx)+1

        print(total_polytopes_this_label)

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
            
            scale = matched_samples[idx]/np.max(matched_samples[idx])
            #print(scale.shape)
            #print(X_[idx].shape)
            scales.append(scale)
            scale_idx.append(idx)
      
    return scales, scale_idx

X, y = gaussian_sparse_parity(
    sample_size,
    p_star=p_star,
    p=p
)
X_test, y_test = gaussian_sparse_parity(
    n_test,
    p_star=p_star,
    p=p
)
#%%
# train Vanilla RF
vanilla_rf = rf(**compile_kwargs).fit(X, y)

# pick a random point from testing data 
#np.random.seed(0)
#idx = np.random.randint(0, sample_size)
#X_star = X[idx]

# sort the points in testing data according to the distance from X_star (consider only the signal dimension)
#distance_vector_ = np.linalg.norm(X[:, :3] - X_star[:3], ord=2, axis=1)
#X_ = X[distance_vector_.argsort()]
#distance_vector_sorted = distance_vector_[distance_vector_.argsort()]

# get the polytope weights
weights, idxs = get_kdf_weights(X, y, vanilla_rf)
distance_vectors = [np.linalg.norm(X[idxs[i], :3] - X[i, :3], ord=2, axis=1) for i in range(len(weights))]

# plot distance vs. weights

sns.set_context('talk')
for sample in range(0, 5):
    fig, ax = plt.subplots(1,1, figsize=(8,8))

    ax.scatter(distance_vectors[sample], weights[sample], c="r", label=f'RF_{sample_size}')

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)

    ax.set_xlabel('Distance')
    ax.set_ylabel('Weight')
    ax.legend(frameon=True)

    # median plots

    RF_unique_weights = np.sort(np.unique(weights[sample]))
    RF_distance_medians = []
    for i in range(len(RF_unique_weights)):
        RF_distance_medians.append(np.median(distance_vectors[sample][weights[sample]==RF_unique_weights[i]]))
    RF_distance_medians = np.array(RF_distance_medians)
    sort = RF_distance_medians.argsort()
    ax.plot(RF_distance_medians[sort], RF_unique_weights[sort], c="k", label='RF')

RF_weight_sizes = np.array([w.shape[0] for w in weights])
RF_weight_mean = np.array([np.mean(w) for w in weights])
RF_weight_median = np.array([np.median(w) for w in weights])
print(f"Number of polytopes: {len(weights)}")
print(f"Mean number of samples per polytopes: {np.mean(RF_weight_sizes)} (range: {np.min(RF_weight_sizes)}, {np.max(RF_weight_sizes)})")
print(f"Mean weight of samples in polytopes: {np.mean(RF_weight_mean):.4f}")
print(f"Median weight of samples in polytopes: {np.median(RF_weight_median)}")