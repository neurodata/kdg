import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from tensorflow import keras
from tensorflow.keras import layers

def getNN(dense_size, input_size, **kwargs):
    network_base = keras.Sequential()
    network_base.add(layers.Dense(dense_size, activation='relu', input_shape=(input_size,)))
    network_base.add(layers.Dense(dense_size, activation='relu'))
    network_base.add(layers.Dense(units=2, activation = 'softmax'))
    network_base.compile(**kwargs)
    return network_base

def _nCr(n, r):
    return math.factorial(n) / math.factorial(r) / math.factorial(n - r)

def hypergeom_weight(n, m):
    # k = nodes drawn before mismatch occurs
    if m == n:  # perfect match
        weight = 1
    else:  # imperfect match, add scaled layer weight and break
        weight = 0
        for k in range(m + 1):
            prob_k = 1 / (k + 1) * (_nCr(m, k) * (n - m)) / _nCr(n, k + 1)
            # print(k/n, 1/(k+1), _nCr(m, k), n-m, _nCr(n, k+1))
            weight += k / n * prob_k
    return weight

def simple_weight(n, m):
    return m / (n * (n - m + 1))

def plot_accuracy(sample_size, pct_errs, labels):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    color_list = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    color_list.extend(sns.color_palette("Set1")[1:])

    for i, err in enumerate(pct_errs):
        ax.plot(sample_size, np.mean(err, axis=1), c=color_list[i], label=labels[i])
        ax.fill_between(
            sample_size,
            np.quantile(err, 0.25, axis=1),
            np.quantile(err, 0.75, axis=1),
            facecolor=color_list[i],
            alpha=0.3,
        )
    
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)

    ax.set_xscale("log")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False)
