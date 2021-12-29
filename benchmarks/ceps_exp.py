#%%
import numpy as np
from sklearn.datasets import make_blobs
# %%
def multiclass_guassian(n_samples, k=98):
    samples_per_blob = np.random.multinomial(
        n_samples, 1 / k * np.ones(k)
    )
    sqrt_cls = np.ceil(np.sqrt(k))

    center_x = np.arange(0,(sqrt_cls-1)*.5,step=.5)
    center_y = np.arange(0,(sqrt_cls-1)*.5,step=.5)
    grid_samples = np.concatenate(
            (
                center_x.reshape(-1,1),
                center_y.reshape(-1,1)
            ),
            axis=1
    ) 
    centers = grid_samples[:k]

    X, y = make_blobs(
        n_samples=samples_per_blob,
        n_features=2,
        centers=centers,
        cluster_std=0.25
    )

    return X, y