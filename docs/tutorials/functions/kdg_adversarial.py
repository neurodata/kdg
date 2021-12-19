import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kdg.utils import generate_gaussian_parity
import random


def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c


def plot_gaussians(Values, Classes, ax=None):
    X = Values
    y = Classes
    colors = sns.color_palette("Dark2", n_colors=2)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(X[:, 0], X[:, 1], c=get_colors(colors, y), s=50)
    ax.set_title("Created Gaussians", fontsize=30)
    plt.tight_layout()


def label_noise_trial_clf(n_samples, p=0.10, n_estimators=500, clf=None):
    """
    Runs a single trial of the label noise experiment at
    a given contamination level for any given classifiers.

    Parameters
    ---
    n_samples : int
        The number of training samples to be generated
    p : float
        The proportion of flipped training labels
    n_estimators : int
        Number of trees in the KDF and RF forests
    clf : classifier class 

    Returns
    ---
    err : list
        A collection of the estimated error of 
        a given classifier on a test distribution
    """
    X, y = generate_gaussian_parity(n_samples, cluster_std=0.5)
    X_test, y_test = generate_gaussian_parity(1000, cluster_std=0.5)

    # Generate noise and flip labels
    n_noise = np.int32(np.round(len(X) * p))
    noise_indices = random.sample(range(len(X)), n_noise)
    y[noise_indices] = 1 - y[noise_indices]

    err = []

    for c in clf:
        c.fit(X, y)
        err.append(1 - np.mean(c.predict(X_test) == y_test))

    return err