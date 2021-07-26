#%%
import numpy as np
from kdg import kdf
from kdg.utils import gaussian_sparse_parity
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
import matplotlib.pyplot as plt
import seaborn as sns
# %%
X, y = gaussian_sparse_parity(10000)
n_estimators = 500
# %%
model_rf = rf(n_estimators=500, max_features=5).fit(X, y)
features = []

for estimator in model_rf.estimators_:
    features.extend(list(estimator.tree_.feature))

features = np.array(features)
features = features[features!=-2]
# %%
sns.set_context("talk")
fig, ax = plt.subplots(1,1, figsize=(8,8))

plt.hist(features, density=True)
plt.xticks(range(0,20,2))
plt.xlabel('Features')
plt.ylabel('Probability of Selection')
plt.savefig('plots/feature_at_node.pdf')
# %%
