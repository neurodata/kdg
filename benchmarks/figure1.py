#%%
import numpy as np
from kdg.utils import generate_gaussian_parity, generate_ellipse, generate_spirals, generate_sinewave, generate_polynomial
from kdg.utils import plot_2dsim
from kdg import kdf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import pickle
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
fig, ax = plt.subplots(6,5, figsize=(40,48), sharex=True)
title_size = 55
ticksize = 30

plot_2dsim(X['gxor'], y['gxor'], ax=ax[0][0])
ax[0][0].set_ylabel('Simulation Data', fontsize=title_size-5)
ax[0][0].set_xlim([-2,2])
ax[0][0].set_ylim([-2,2])
ax[0][0].set_xticks([])
ax[0][0].set_yticks([])
ax[0][0].tick_params(labelsize=ticksize)
ax[0][0].set_title('Gaussian XOR', fontsize=title_size)

plot_2dsim(X['spiral'], y['spiral'], ax=ax[0][1])
ax[0][1].set_xlim([-2,2])
ax[0][1].set_ylim([-2,2])
ax[0][1].set_xticks([])
ax[0][1].set_yticks([])
ax[0][1].tick_params(labelsize=ticksize)
ax[0][1].set_title('Spiral', fontsize=title_size)

plot_2dsim(X['circle'], y['circle'], ax=ax[0][2])
ax[0][2].set_xlim([-2,2])
ax[0][2].set_ylim([-2,2])
ax[0][2].set_xticks([])
ax[0][2].set_yticks([])
ax[0][2].tick_params(labelsize=ticksize)
ax[0][2].set_title('Circle', fontsize=title_size)

plot_2dsim(X['sine'], y['sine'], ax=ax[0][3])
ax[0][3].set_xlim([-2,2])
ax[0][3].set_ylim([-2,2])
ax[0][3].set_xticks([])
ax[0][3].set_yticks([])
ax[0][3].tick_params(labelsize=ticksize)
ax[0][3].set_title('Sinewave', fontsize=title_size)

plot_2dsim(X['poly'], y['poly'], ax=ax[0][4])
ax[0][4].set_xlim([-2,2])
ax[0][4].set_ylim([-2,2])
ax[0][4].set_xticks([])
ax[0][4].set_yticks([])
ax[0][4].tick_params(labelsize=ticksize)
ax[0][4].set_title('Polynomial', fontsize=title_size)

################################################
#define grids
p = np.arange(-2, 2, step=0.01)
q = np.arange(-2, 2, step=0.01)
xx, yy = np.meshgrid(p, q)

# get true posterior
tp_df = pd.read_csv("true_posterior/Gaussian_xor_pdf.csv")
proba_true = 0.5*np.ones((400, 400))
tmp = np.array([tp_df["posterior"][x] for x in range(40000)])
tmp = tmp.reshape(200, 200)
proba_true[100:300, 100:300] = tmp

ax0 = ax[1][0].imshow(
    proba_true,
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
#ax[1][0].set_title("True Class Posteriors", fontsize=24)
ax[1][0].set_aspect("equal")
ax[1][0].tick_params(labelsize=ticksize)
ax[1][0].set_yticks([])
ax[1][0].set_xticks([])
ax[1][0].set_ylabel('True Posteriors',fontsize=title_size-5)

tp_df = pd.read_csv("true_posterior/spiral_pdf.csv")
proba_true = 0.5*np.ones((400, 400))
tmp = np.array([tp_df["posterior"][x] for x in range(40000)])
tmp = tmp.reshape(200, 200)
proba_true[100:300, 100:300] = 1 - tmp

ax0 = ax[1][1].imshow(
    np.flip(proba_true, axis=0),
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
#ax[1][1].set_title("True Class Posteriors", fontsize=24)
ax[1][1].set_aspect("equal")
ax[1][1].tick_params(labelsize=ticksize)
ax[1][1].set_yticks([])
ax[1][1].set_xticks([])


tp_df = pd.read_csv("true_posterior/ellipse_pdf.csv")
proba_true = 0.5*np.ones((400, 400))
tmp = np.array([tp_df["posterior"][x] for x in range(40000)])
tmp = tmp.reshape(200, 200)
proba_true[100:300, 100:300] = tmp

ax0 = ax[1][2].imshow(
    proba_true,
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
#ax[1][2].set_title("True Class Posteriors", fontsize=24)
ax[1][2].set_aspect("equal")
ax[1][2].tick_params(labelsize=ticksize)
ax[1][2].set_yticks([])
ax[1][2].set_xticks([])


tp_df = pd.read_csv("true_posterior/sinewave_pdf.csv")
proba_true = 0.5*np.ones((400, 400))
tmp = np.array([tp_df["posterior"][x] for x in range(40000)])
tmp = np.flip(tmp.reshape(200, 200),axis=0)
proba_true[100:300, 100:300] = tmp

ax0 = ax[1][3].imshow(
    proba_true,
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
#ax[1][3].set_title("True Class Posteriors", fontsize=24)
ax[1][3].set_aspect("equal")
ax[1][3].tick_params(labelsize=ticksize)
ax[1][3].set_yticks([])
ax[1][3].set_xticks([])


tp_df = pd.read_csv("true_posterior/polynomial_pdf.csv")
proba_true = 0.5*np.ones((400, 400))
tmp = np.array([tp_df["posterior"][x] for x in range(40000)])
tmp = np.flip(tmp.reshape(200, 200),axis=0)
proba_true[100:300, 100:300] = tmp

ax0 = ax[1][4].imshow(
    proba_true,
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
#ax[1][4].set_title("True Class Posteriors", fontsize=24)
ax[1][4].set_aspect("equal")
ax[1][4].tick_params(labelsize=ticksize)
ax[1][4].set_yticks([])
ax[1][4].set_xticks([])

#########################################################
with open('kdf_simulations/results/gxor.pickle', 'rb') as f:
    df = pickle.load(f)

ax1 = ax[2][0].imshow(
    df['posterior_rf'],
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[2][0].set_ylabel("RF Posteriors", fontsize=title_size-5)
ax[2][0].set_aspect("equal")
ax[2][0].tick_params(labelsize=ticksize)
ax[2][0].set_yticks([])
ax[2][0].set_xticks([])


ax1 = ax[3][0].imshow(
    df['posterior_kdf'],
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[3][0].set_ylabel('KDF Posteriors', fontsize=title_size-5)
ax[3][0].set_aspect("equal")
ax[3][0].tick_params(labelsize=ticksize)
ax[3][0].set_yticks([])
ax[3][0].set_xticks([])

############################################
with open('kdf_simulations/results/spiral.pickle', 'rb') as f:
    df = pickle.load(f)

ax1 = ax[2][1].imshow(
    1-np.flip(df['posterior_rf'],axis=0),
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[2][1].set_aspect("equal")
ax[2][1].tick_params(labelsize=ticksize)
ax[2][1].set_yticks([])
ax[2][1].set_xticks([])


ax1 = ax[3][1].imshow(
    1-np.flip(df['posterior_kdf'],axis=0),
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[3][1].set_aspect("equal")
ax[3][1].tick_params(labelsize=ticksize)
ax[3][1].set_yticks([])
ax[3][1].set_xticks([])

#############################################
with open('kdf_simulations/results/circle.pickle', 'rb') as f:
    df = pickle.load(f)

ax1 = ax[2][2].imshow(
    df['posterior_rf'],
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[2][2].set_aspect("equal")
ax[2][2].tick_params(labelsize=ticksize)
ax[2][2].set_yticks([])
ax[2][2].set_xticks([])


ax1 = ax[3][2].imshow(
    df['posterior_kdf'],
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[3][2].set_aspect("equal")
ax[3][2].tick_params(labelsize=ticksize)
ax[3][2].set_yticks([])
ax[3][2].set_xticks([])


##################################################
with open('kdf_simulations/results/sinewave.pickle', 'rb') as f:
    df = pickle.load(f)

ax1 = ax[2][3].imshow(
    np.flip(df['posterior_rf'],axis=0),
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[2][3].set_aspect("equal")
ax[2][3].tick_params(labelsize=ticksize)
ax[2][3].set_yticks([])
ax[2][3].set_xticks([])


ax1 = ax[3][3].imshow(
    np.flip(df['posterior_kdf'], axis=0),
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[3][3].set_aspect("equal")
ax[3][3].tick_params(labelsize=ticksize)
ax[3][3].set_yticks([])
ax[3][3].set_xticks([])

###################################################
with open('kdf_simulations/results/polynomial.pickle', 'rb') as f:
    df = pickle.load(f)

ax1 = ax[2][4].imshow(
    np.flip(df['posterior_rf'],axis=0),
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[2][4].set_aspect("equal")
ax[2][4].tick_params(labelsize=ticksize)
ax[2][4].set_yticks([])
ax[2][4].set_xticks([])


ax1 = ax[3][4].imshow(
    np.flip(df['posterior_kdf'],axis=0),
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[3][4].set_aspect("equal")
ax[3][4].tick_params(labelsize=ticksize)
ax[3][4].set_yticks([])
ax[3][4].set_xticks([])


##############################################
##############################################
with open('kdn_simulations/results/gxor.pickle', 'rb') as f:
    df = pickle.load(f)

proba_nn = 1-np.flip(df["posterior_dn"], axis=1)
proba_kdn = 1-np.flip(df["posterior_kdn"], axis=1)

ax1 = ax[4][0].imshow(
    proba_nn,
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[4][0].set_aspect("equal")
ax[4][0].tick_params(labelsize=ticksize)
ax[4][0].set_ylabel('DN Posteriors',fontsize=title_size-5)
ax[4][0].set_yticks([])
ax[4][0].set_xticks([])


ax1 = ax[5][0].imshow(
    proba_kdn,
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[5][0].set_aspect("equal")
ax[5][0].set_ylabel('KDN Posteriors',fontsize=title_size-5)
ax[5][0].tick_params(labelsize=ticksize)
ax[5][0].set_yticks([])
ax[5][0].set_xticks([])


########################################
with open('kdn_simulations/results/spiral.pickle', 'rb') as f:
    df = pickle.load(f)

proba_nn = np.flip(df["posterior_dn"], axis=1)
proba_kdn = np.flip(df["posterior_kdn"], axis=1)

ax1 = ax[4][1].imshow(
    proba_nn,
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[4][1].set_aspect("equal")
ax[4][1].tick_params(labelsize=ticksize)
ax[4][1].set_yticks([])
ax[4][1].set_xticks([])


ax1 = ax[5][1].imshow(
    proba_kdn,
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[5][1].set_aspect("equal")
ax[5][1].tick_params(labelsize=ticksize)
ax[5][1].set_yticks([])
ax[5][1].set_xticks([])

########################################################
with open('kdn_simulations/results/circle.pickle', 'rb') as f:
    df = pickle.load(f)

proba_nn = np.flip(df["posterior_dn"], axis=1)
proba_kdn = np.flip(df["posterior_kdn"], axis=1)

ax1 = ax[4][2].imshow(
    proba_nn,
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[4][2].set_aspect("equal")
ax[4][2].tick_params(labelsize=ticksize)
ax[4][2].set_yticks([])
ax[4][2].set_xticks([])


ax1 = ax[5][2].imshow(
    proba_kdn,
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[5][2].set_aspect("equal")
ax[5][2].tick_params(labelsize=ticksize)
ax[5][2].set_yticks([])
ax[5][2].set_xticks([])

####################################################
with open('kdn_simulations/results/sinewave.pickle', 'rb') as f:
    df = pickle.load(f)

proba_nn = np.flip(df["posterior_dn"], axis=0)
proba_kdn = np.flip(df["posterior_kdn"], axis=0)

ax1 = ax[4][3].imshow(
    proba_nn,
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[4][3].set_aspect("equal")
ax[4][3].tick_params(labelsize=ticksize)
ax[4][3].set_yticks([])
ax[4][3].set_xticks([])


ax1 = ax[5][3].imshow(
    proba_kdn,
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
ax[5][3].set_aspect("equal")
ax[5][3].tick_params(labelsize=ticksize)
ax[5][3].set_yticks([])
ax[5][3].set_xticks([])

#######################################################
with open('kdn_simulations/results/polynomial.pickle', 'rb') as f:
    df = pickle.load(f)

proba_nn = np.flip(df["posterior_dn"], axis=1)
proba_kdn = np.flip(df["posterior_kdn"], axis=1)

ax1 = ax[4][4].imshow(
    proba_nn,
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
#fig.colorbar(ax1, ax=ax[4][4], anchor=(0, 0.3), shrink=0.85)
ax[4][4].set_aspect("equal")
ax[4][4].tick_params(labelsize=ticksize)
ax[4][4].set_yticks([])
ax[4][4].set_xticks([])


ax1 = ax[5][4].imshow(
    proba_kdn,
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="bwr",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto",
)
#fig.colorbar(ax1, anchor=(0, 0.3), shrink=0.85)

ax[5][4].set_aspect("equal")
ax[5][4].tick_params(labelsize=ticksize)
ax[5][4].set_yticks([])
ax[5][4].set_xticks([])

plt.savefig('plots/simulations.pdf')
# %%
