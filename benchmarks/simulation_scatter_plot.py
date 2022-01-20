#%%
import numpy as np
from kdg.utils import generate_gaussian_parity, generate_ellipse, generate_spirals, generate_sinewave, generate_polynomial
from kdg.utils import plot_2dsim
from kdg import kdf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
# %%
n_samples = 1e4
X, y = {}, {}
#%%
X['gxor'], y['gxor'] = generate_gaussian_parity(n_samples)
X['spiral'], y['spiral'] = generate_spirals(n_samples)
X['circle'], y['circle'] = generate_ellipse(n_samples)
X['sine'], y['sine'] = generate_sinewave(n_samples)
X['poly'], y['poly'] = generate_polynomial(n_samples, a=[1,3])
#%%
sns.set_context('talk')
fig, ax = plt.subplots(1,5, figsize=(40,8), sharey=True)
title_size = 45
ticksize = 30

plot_2dsim(X['gxor'], y['gxor'], ax=ax[0])
ax[0].set_xticks([-1,0,1])
ax[0].set_yticks([-1,0,1])
ax[0].tick_params(labelsize=ticksize)
ax[0].set_title('Gaussian XOR', fontsize=title_size)

plot_2dsim(X['spiral'], y['spiral'], ax=ax[1])
ax[1].set_xticks([-1,0,1])
ax[1].set_yticks([-1,0,1])
ax[1].tick_params(labelsize=ticksize)
ax[1].set_title('Spiral', fontsize=title_size)

plot_2dsim(X['circle'], y['circle'], ax=ax[2])
ax[2].set_xticks([-1,0,1])
ax[2].set_yticks([-1,0,1])
ax[2].tick_params(labelsize=ticksize)
ax[2].set_title('Circle', fontsize=title_size)

plot_2dsim(X['sine'], y['sine'], ax=ax[3])
ax[3].set_xticks([-1,0,1])
ax[3].set_yticks([-1,0,1])
ax[3].tick_params(labelsize=ticksize)
ax[3].set_title('Sinewave', fontsize=title_size)

plot_2dsim(X['poly'], y['poly'], ax=ax[4])
ax[4].set_xticks([-1,0,1])
ax[4].set_yticks([-1,0,1])
ax[4].tick_params(labelsize=ticksize)
ax[4].set_title('Polynomial', fontsize=title_size)

plt.savefig('plots/simulations.pdf')
# %%
