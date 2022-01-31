#%%
import numpy as np
from kdg.utils import generate_gaussian_parity, generate_ellipse, generate_spirals, generate_sinewave, generate_polynomial
from kdg.utils import plot_2dsim
from kdg import kdf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from scipy.io import savemat, loadmat
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
title_size = 45
ticksize = 30

plot_2dsim(X['gxor'], y['gxor'], ax=ax[0][0])
ax[0][0].set_ylabel('Simulation Data', fontsize=title_size-5)
ax[0][0].set_xlim([-2,2])
ax[0][0].set_ylim([-2,2])
ax[0][0].set_xticks([])
ax[0][0].set_yticks([-2,-1,0,1,2])
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
ax[1][0].set_yticks([-2,-1,0,1,2])
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
df = loadmat('kdf_experiments/results/gxor_plot_data.mat')
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
ax[2][0].set_yticks([-2,-1,0,1,2])
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
ax[3][0].set_yticks([-2,-1,0,1,2])
ax[3][0].set_xticks([])

############################################
df = loadmat('kdf_experiments/results/spiral_plot_data.mat')
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
df = loadmat('kdf_experiments/results/circle_plot_data.mat')
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
df = loadmat('kdf_experiments/results/sinewave_plot_data.mat')
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
df = loadmat('kdf_experiments/results/polynomial_plot_data.mat')
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
df = loadmat('kdn_experiments/results/gxor_plot_data.mat')
proba_nn = 1-np.flip(df["nn_proba"][:, 0].reshape(400, 400), axis=1)
proba_kdn = 1-np.flip(df["kdn_proba"][:, 0].reshape(400, 400), axis=1)

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
ax[4][0].set_ylabel('NN Posteriors',fontsize=title_size-5)
ax[4][0].set_yticks([-2,-1,0,1,2])
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
ax[5][0].set_yticks([-2,-1,0,1,2])
ax[5][0].set_xticks([-2,-1,0,1,2])


########################################
df = loadmat('kdn_experiments/results/spiral_plot_data.mat')
proba_nn = np.flip(df["nn_proba"][:, 0].reshape(400, 400), axis=1)
proba_kdn = np.flip(df["kdn_proba"][:, 0].reshape(400, 400), axis=1)

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
ax[5][1].set_xticks([-2,-1,0,1,2])

########################################################
df = loadmat('kdn_experiments/results/circle_plot_data.mat')
proba_nn = np.flip(df["nn_proba"][:, 0].reshape(400, 400), axis=1)
proba_kdn = np.flip(df["kdn_proba"][:, 0].reshape(400, 400), axis=1)

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
ax[5][2].set_xticks([-2,-1,0,1,2])

####################################################
df = loadmat('kdn_experiments/results/sinewave_plot_data.mat')
proba_nn = np.flip(df["nn_proba"][:, 0].reshape(400, 400), axis=0)
proba_kdn = np.flip(df["kdn_proba"][:, 0].reshape(400, 400), axis=0)

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
ax[5][3].set_xticks([-2,-1,0,1,2])

#######################################################
df = loadmat('kdn_experiments/results/polynomial_plot_data.mat')
proba_nn = 1-np.flip(df["nn_proba"][:, 0].reshape(400, 400), axis=1)
proba_kdn = np.flip(df["kdn_proba"][:, 0].reshape(400, 400), axis=1)

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
ax[5][4].set_xticks([-2,-1,0,1,2])

#plt.savefig('plots/simulations.pdf')
# %%
def calc_stat(a, reps=45):
    a_med = []
    a_25 = []
    a_75 = []
    a = a.reshape(-1,reps)
    return np.median(a,axis=1), np.quantile(a,[.25], axis=1)[0], np.quantile(a,[.75], axis=1)[0]
# %%
sns.set_context('talk')
sample_size = [50, 100, 500, 1000, 5000, 10000]

fig, ax = plt.subplots(5,4, figsize=(45,40))
title_size = 45
ticksize = 30

for ax_ in ax:
    for ax__ in ax_:
        ax__.tick_params(labelsize=ticksize)

        
df = loadmat('kdn_experiments/results/graphs/gxor.mat')

med, a_25, a_75 = calc_stat(1-df['kdn_acc'])
med_nn, nn_25, nn_75 = calc_stat(1-df['nn_acc'])

ax[0][0].plot(sample_size, med[1:], c="b", label='KDN')
ax[0][0].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[0][0].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[0][0].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[0][0].set_xscale('log')
#ax[0][0].set_xlabel('Sample size')
ax[0][0].set_xticks([])
ax[0][0].set_ylabel('Generalization Error', fontsize=ticksize)

right_side = ax[0][0].spines["right"]
right_side.set_visible(False)
top_side = ax[0][0].spines["top"]
top_side.set_visible(False)

df_ = loadmat('kdf_experiments/results/gxor_plot_data.mat')
ax[0][0].plot(sample_size, df_['error_kdf_med'].ravel(), c="r", label='KDF')
ax[0][0].plot(sample_size, df_['error_rf_med'].ravel(), c="k", label='RF')

ax[0][0].fill_between(sample_size, df_["error_kdf_25"].ravel(), df_["error_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[0][0].fill_between(sample_size, df_["error_rf_25"].ravel(), df_["error_rf_75"].ravel(), facecolor='k', alpha=.3)

ax[0][0].legend(fontsize=ticksize, frameon=False)

##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_hd'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_hd'])

ax[0][1].plot(sample_size, med[1:], c="b", label='KDN')
ax[0][1].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[0][1].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[0][1].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[0][1].set_xscale('log')
ax[0][1].set_xticks([])
ax[0][1].set_ylabel('Hellinger Distance', fontsize=ticksize)

right_side = ax[0][1].spines["right"]
right_side.set_visible(False)
top_side = ax[0][1].spines["top"]
top_side.set_visible(False)

ax[0][1].plot(sample_size, df_['hellinger_kdf_med'].ravel(), c="r", label='KDF')
ax[0][1].plot(sample_size, df_['hellinger_rf_med'].ravel(), c="k", label='RF')

ax[0][1].fill_between(sample_size, df_["hellinger_kdf_25"].ravel(), df_["hellinger_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[0][1].fill_between(sample_size, df_["hellinger_rf_25"].ravel(), df_["hellinger_rf_75"].ravel(), facecolor='k', alpha=.3)
ax[0][1].set_title('Gaussian XOR', fontsize=title_size)

##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_mmcIn'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_mmcIn'])

ax[0][2].plot(sample_size, med[1:], c="b", label='KDN')
ax[0][2].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[0][2].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[0][2].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[0][2].set_xscale('log')
ax[0][2].set_xticks([])
ax[0][2].set_ylabel('Mean Max Confidence\n (In Distribution)', fontsize=ticksize)

right_side = ax[0][2].spines["right"]
right_side.set_visible(False)
top_side = ax[0][2].spines["top"]
top_side.set_visible(False)

ax[0][2].plot(sample_size, df_['mmcIn_kdf_med'].ravel(), c="r", label='KDF')
ax[0][2].plot(sample_size, df_['mmcIn_rf_med'].ravel(), c="k", label='RF')

ax[0][2].fill_between(sample_size, df_["mmcIn_kdf_25"].ravel(), df_["mmcIn_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[0][2].fill_between(sample_size, df_["mmcIn_rf_25"].ravel(), df_["mmcIn_rf_75"].ravel(), facecolor='k', alpha=.3)

##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_mmcOut'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_mmcOut'])

ax[0][3].plot(sample_size, med[1:], c="b", label='KDN')
ax[0][3].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[0][3].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[0][3].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[0][3].set_xscale('log')
ax[0][3].set_xticks([])
ax[0][3].set_ylabel('Mean Max Confidence\n (Out Distribution)', fontsize=ticksize)

right_side = ax[0][3].spines["right"]
right_side.set_visible(False)
top_side = ax[0][3].spines["top"]
top_side.set_visible(False)

ax[0][3].plot(sample_size, df_['mmcOut_kdf_med'].ravel(), c="r", label='KDF')
ax[0][3].plot(sample_size, df_['mmcOut_rf_med'].ravel(), c="k", label='RF')

ax[0][3].fill_between(sample_size, df_["mmcOut_kdf_25"].ravel(), df_["mmcOut_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[0][3].fill_between(sample_size, df_["mmcOut_rf_25"].ravel(), df_["mmcOut_rf_75"].ravel(), facecolor='k', alpha=.3)

#########################################################
#########################################################
df = loadmat('kdn_experiments/results/graphs/spiral.mat')

med, a_25, a_75 = calc_stat(1-df['kdn_acc'])
med_nn, nn_25, nn_75 = calc_stat(1-df['nn_acc'])

ax[1][0].plot(sample_size, med[1:], c="b", label='KDN')
ax[1][0].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[1][0].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[1][0].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[1][0].set_xscale('log')
ax[1][0].set_xticks([])
ax[1][0].set_ylabel('Generalization Error', fontsize=ticksize)

right_side = ax[1][0].spines["right"]
right_side.set_visible(False)
top_side = ax[1][0].spines["top"]
top_side.set_visible(False)

df_ = loadmat('kdf_experiments/results/spiral_plot_data.mat')
ax[1][0].plot(sample_size, df_['error_kdf_med'].ravel(), c="r", label='KDF')
ax[1][0].plot(sample_size, df_['error_rf_med'].ravel(), c="k", label='RF')

ax[1][0].fill_between(sample_size, df_["error_kdf_25"].ravel(), df_["error_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[1][0].fill_between(sample_size, df_["error_rf_25"].ravel(), df_["error_rf_75"].ravel(), facecolor='k', alpha=.3)


##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_hd'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_hd'])

ax[1][1].plot(sample_size, med[1:], c="b", label='KDN')
ax[1][1].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[1][1].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[1][1].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[1][1].set_xscale('log')
ax[1][1].set_xticks([])
ax[1][1].set_ylabel('Hellinger Distance', fontsize=ticksize)

right_side = ax[1][1].spines["right"]
right_side.set_visible(False)
top_side = ax[1][1].spines["top"]
top_side.set_visible(False)

ax[1][1].plot(sample_size, df_['hellinger_kdf_med'].ravel(), c="r", label='KDF')
ax[1][1].plot(sample_size, df_['hellinger_rf_med'].ravel(), c="k", label='RF')

ax[1][1].fill_between(sample_size, df_["hellinger_kdf_25"].ravel(), df_["hellinger_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[1][1].fill_between(sample_size, df_["hellinger_rf_25"].ravel(), df_["hellinger_rf_75"].ravel(), facecolor='k', alpha=.3)
ax[1][1].set_title('Spiral', fontsize=title_size)
##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_mmcIn'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_mmcIn'])

ax[1][2].plot(sample_size, med[1:], c="b", label='KDN')
ax[1][2].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[1][2].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[1][2].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[1][2].set_xscale('log')
ax[1][2].set_xticks([])
ax[1][2].set_ylabel('Mean Max Confidence\n (In Distribution)', fontsize=ticksize)

right_side = ax[1][2].spines["right"]
right_side.set_visible(False)
top_side = ax[1][2].spines["top"]
top_side.set_visible(False)

ax[1][2].plot(sample_size, df_['mmcIn_kdf_med'].ravel(), c="r", label='KDF')
ax[1][2].plot(sample_size, df_['mmcIn_rf_med'].ravel(), c="k", label='RF')

ax[1][2].fill_between(sample_size, df_["mmcIn_kdf_25"].ravel(), df_["mmcIn_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[1][2].fill_between(sample_size, df_["mmcIn_rf_25"].ravel(), df_["mmcIn_rf_75"].ravel(), facecolor='k', alpha=.3)

##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_mmcOut'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_mmcOut'])

ax[1][3].plot(sample_size, med[1:], c="b", label='KDN')
ax[1][3].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[1][3].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[1][3].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[1][3].set_xscale('log')
ax[1][3].set_xticks([])
ax[1][3].set_ylabel('Mean Max Confidence\n (Out Distribution)', fontsize=ticksize)

right_side = ax[1][3].spines["right"]
right_side.set_visible(False)
top_side = ax[1][3].spines["top"]
top_side.set_visible(False)

ax[1][3].plot(sample_size, df_['mmcOut_kdf_med'].ravel(), c="r", label='KDF')
ax[1][3].plot(sample_size, df_['mmcOut_rf_med'].ravel(), c="k", label='RF')

ax[1][3].fill_between(sample_size, df_["mmcOut_kdf_25"].ravel(), df_["mmcOut_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[1][3].fill_between(sample_size, df_["mmcOut_rf_25"].ravel(), df_["mmcOut_rf_75"].ravel(), facecolor='k', alpha=.3)

#########################################################
#########################################################
df = loadmat('kdn_experiments/results/graphs/circle.mat')

med, a_25, a_75 = calc_stat(1-df['kdn_acc'])
med_nn, nn_25, nn_75 = calc_stat(1-df['nn_acc'])

ax[2][0].plot(sample_size, med[1:], c="b", label='KDN')
ax[2][0].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[2][0].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[2][0].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[2][0].set_xscale('log')
ax[2][0].set_xticks([])
ax[2][0].set_ylabel('Generalization Error', fontsize=ticksize)

right_side = ax[2][0].spines["right"]
right_side.set_visible(False)
top_side = ax[2][0].spines["top"]
top_side.set_visible(False)

df_ = loadmat('kdf_experiments/results/circle_plot_data.mat')
ax[2][0].plot(sample_size, df_['error_kdf_med'].ravel(), c="r", label='KDF')
ax[2][0].plot(sample_size, df_['error_rf_med'].ravel(), c="k", label='RF')

ax[2][0].fill_between(sample_size, df_["error_kdf_25"].ravel(), df_["error_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[2][0].fill_between(sample_size, df_["error_rf_25"].ravel(), df_["error_rf_75"].ravel(), facecolor='k', alpha=.3)


##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_hd'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_hd'])

ax[2][1].plot(sample_size, med[1:], c="b", label='KDN')
ax[2][1].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[2][1].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[2][1].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[2][1].set_xscale('log')
ax[2][1].set_xticks([])
ax[2][1].set_ylabel('Hellinger Distance', fontsize=ticksize)

right_side = ax[2][1].spines["right"]
right_side.set_visible(False)
top_side = ax[2][1].spines["top"]
top_side.set_visible(False)

ax[2][1].plot(sample_size, df_['hellinger_kdf_med'].ravel(), c="r", label='KDF')
ax[2][1].plot(sample_size, df_['hellinger_rf_med'].ravel(), c="k", label='RF')

ax[2][1].fill_between(sample_size, df_["hellinger_kdf_25"].ravel(), df_["hellinger_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[2][1].fill_between(sample_size, df_["hellinger_rf_25"].ravel(), df_["hellinger_rf_75"].ravel(), facecolor='k', alpha=.3)
ax[2][1].set_title('Circle', fontsize=title_size)
##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_mmcIn'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_mmcIn'])

ax[2][2].plot(sample_size, med[1:], c="b", label='KDN')
ax[2][2].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[2][2].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[2][2].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[2][2].set_xscale('log')
ax[2][2].set_xticks([])
ax[2][2].set_ylabel('Mean Max Confidence\n (In Distribution)', fontsize=ticksize)

right_side = ax[2][2].spines["right"]
right_side.set_visible(False)
top_side = ax[2][2].spines["top"]
top_side.set_visible(False)

ax[2][2].plot(sample_size, df_['mmcIn_kdf_med'].ravel(), c="r", label='KDF')
ax[2][2].plot(sample_size, df_['mmcIn_rf_med'].ravel(), c="k", label='RF')

ax[2][2].fill_between(sample_size, df_["mmcIn_kdf_25"].ravel(), df_["mmcIn_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[2][2].fill_between(sample_size, df_["mmcIn_rf_25"].ravel(), df_["mmcIn_rf_75"].ravel(), facecolor='k', alpha=.3)

##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_mmcOut'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_mmcOut'])

ax[2][3].plot(sample_size, med[1:], c="b", label='KDN')
ax[2][3].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[2][3].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[2][3].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[2][3].set_xscale('log')
ax[2][3].set_xticks([])
ax[2][3].set_ylabel('Mean Max Confidence\n (Out Distribution)', fontsize=ticksize)

right_side = ax[2][3].spines["right"]
right_side.set_visible(False)
top_side = ax[2][3].spines["top"]
top_side.set_visible(False)

ax[2][3].plot(sample_size, df_['mmcOut_kdf_med'].ravel(), c="r", label='KDF')
ax[2][3].plot(sample_size, df_['mmcOut_rf_med'].ravel(), c="k", label='RF')

ax[2][3].fill_between(sample_size, df_["mmcOut_kdf_25"].ravel(), df_["mmcOut_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[2][3].fill_between(sample_size, df_["mmcOut_rf_25"].ravel(), df_["mmcOut_rf_75"].ravel(), facecolor='k', alpha=.3)


#########################################################
#########################################################
df = loadmat('kdn_experiments/results/graphs/sinewave.mat')

med, a_25, a_75 = calc_stat(1-df['kdn_acc'])
med_nn, nn_25, nn_75 = calc_stat(1-df['nn_acc'])

ax[3][0].plot(sample_size, med[1:], c="b", label='KDN')
ax[3][0].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[3][0].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[3][0].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[3][0].set_xscale('log')
ax[3][0].set_xticks([])
ax[3][0].set_ylabel('Generalization Error', fontsize=ticksize)

right_side = ax[3][0].spines["right"]
right_side.set_visible(False)
top_side = ax[3][0].spines["top"]
top_side.set_visible(False)

df_ = loadmat('kdf_experiments/results/sinewave_plot_data.mat')
ax[3][0].plot(sample_size, df_['error_kdf_med'].ravel(), c="r", label='KDF')
ax[3][0].plot(sample_size, df_['error_rf_med'].ravel(), c="k", label='RF')

ax[3][0].fill_between(sample_size, df_["error_kdf_25"].ravel(), df_["error_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[3][0].fill_between(sample_size, df_["error_rf_25"].ravel(), df_["error_rf_75"].ravel(), facecolor='k', alpha=.3)


##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_hd'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_hd'])

ax[3][1].plot(sample_size, med[1:], c="b", label='KDN')
ax[3][1].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[3][1].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[3][1].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[3][1].set_xscale('log')
ax[3][1].set_xticks([])
ax[3][1].set_ylabel('Hellinger Distance', fontsize=ticksize)

right_side = ax[3][1].spines["right"]
right_side.set_visible(False)
top_side = ax[3][1].spines["top"]
top_side.set_visible(False)

ax[3][1].plot(sample_size, df_['hellinger_kdf_med'].ravel(), c="r", label='KDF')
ax[3][1].plot(sample_size, df_['hellinger_rf_med'].ravel(), c="k", label='RF')

ax[3][1].fill_between(sample_size, df_["hellinger_kdf_25"].ravel(), df_["hellinger_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[3][1].fill_between(sample_size, df_["hellinger_rf_25"].ravel(), df_["hellinger_rf_75"].ravel(), facecolor='k', alpha=.3)
ax[3][1].set_title('Sinewave', fontsize=title_size)
##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_mmcIn'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_mmcIn'])

ax[3][2].plot(sample_size, med[1:], c="b", label='KDN')
ax[3][2].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[3][2].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[3][2].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[3][2].set_xscale('log')
ax[3][2].set_xticks([])
ax[3][2].set_ylabel('Mean Max Confidence\n (In Distribution)', fontsize=ticksize)

right_side = ax[3][2].spines["right"]
right_side.set_visible(False)
top_side = ax[3][2].spines["top"]
top_side.set_visible(False)

ax[3][2].plot(sample_size, df_['mmcIn_kdf_med'].ravel(), c="r", label='KDF')
ax[3][2].plot(sample_size, df_['mmcIn_rf_med'].ravel(), c="k", label='RF')

ax[3][2].fill_between(sample_size, df_["mmcIn_kdf_25"].ravel(), df_["mmcIn_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[3][2].fill_between(sample_size, df_["mmcIn_rf_25"].ravel(), df_["mmcIn_rf_75"].ravel(), facecolor='k', alpha=.3)

##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_mmcOut'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_mmcOut'])

ax[3][3].plot(sample_size, med[1:], c="b", label='KDN')
ax[3][3].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[3][3].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[3][3].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[3][3].set_xscale('log')
ax[3][3].set_xticks([])
ax[3][3].set_ylabel('Mean Max Confidence\n (Out Distribution)', fontsize=ticksize)

right_side = ax[3][3].spines["right"]
right_side.set_visible(False)
top_side = ax[3][3].spines["top"]
top_side.set_visible(False)

ax[3][3].plot(sample_size, df_['mmcOut_kdf_med'].ravel(), c="r", label='KDF')
ax[3][3].plot(sample_size, df_['mmcOut_rf_med'].ravel(), c="k", label='RF')

ax[3][3].fill_between(sample_size, df_["mmcOut_kdf_25"].ravel(), df_["mmcOut_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[3][3].fill_between(sample_size, df_["mmcOut_rf_25"].ravel(), df_["mmcOut_rf_75"].ravel(), facecolor='k', alpha=.3)


#########################################################
#########################################################
df = loadmat('kdn_experiments/results/graphs/polynomial.mat')

med, a_25, a_75 = calc_stat(1-df['kdn_acc'])
med_nn, nn_25, nn_75 = calc_stat(1-df['nn_acc'])

ax[4][0].plot(sample_size, med[1:], c="b", label='KDN')
ax[4][0].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[4][0].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[4][0].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[4][0].set_xscale('log')
ax[4][0].set_xlabel('Sample size', fontsize=ticksize)
ax[4][0].set_ylabel('Generalization Error', fontsize=ticksize)

right_side = ax[4][0].spines["right"]
right_side.set_visible(False)
top_side = ax[4][0].spines["top"]
top_side.set_visible(False)

df_ = loadmat('kdf_experiments/results/polynomial_plot_data.mat')
ax[4][0].plot(sample_size, df_['error_kdf_med'].ravel(), c="r", label='KDF')
ax[4][0].plot(sample_size, df_['error_rf_med'].ravel(), c="k", label='RF')

ax[4][0].fill_between(sample_size, df_["error_kdf_25"].ravel(), df_["error_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[4][0].fill_between(sample_size, df_["error_rf_25"].ravel(), df_["error_rf_75"].ravel(), facecolor='k', alpha=.3)


##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_hd'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_hd'])

ax[4][1].plot(sample_size, med[1:], c="b", label='KDN')
ax[4][1].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[4][1].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[4][1].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[4][1].set_xscale('log')
ax[4][1].set_xlabel('Sample size', fontsize=ticksize)
ax[4][1].set_ylabel('Hellinger Distance', fontsize=ticksize)

right_side = ax[4][1].spines["right"]
right_side.set_visible(False)
top_side = ax[4][1].spines["top"]
top_side.set_visible(False)

ax[4][1].plot(sample_size, df_['hellinger_kdf_med'].ravel(), c="r", label='KDF')
ax[4][1].plot(sample_size, df_['hellinger_rf_med'].ravel(), c="k", label='RF')

ax[4][1].fill_between(sample_size, df_["hellinger_kdf_25"].ravel(), df_["hellinger_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[4][1].fill_between(sample_size, df_["hellinger_rf_25"].ravel(), df_["hellinger_rf_75"].ravel(), facecolor='k', alpha=.3)
ax[4][1].set_title('Polynomial', fontsize=title_size)
##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_mmcIn'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_mmcIn'])

ax[4][2].plot(sample_size, med[1:], c="b", label='KDN')
ax[4][2].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[4][2].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[4][2].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[4][2].set_xscale('log')
ax[4][2].set_xlabel('Sample size', fontsize=ticksize)
ax[4][2].set_ylabel('Mean Max Confidence\n (In Distribution)', fontsize=ticksize)

right_side = ax[4][2].spines["right"]
right_side.set_visible(False)
top_side = ax[4][2].spines["top"]
top_side.set_visible(False)

ax[4][2].plot(sample_size, df_['mmcIn_kdf_med'].ravel(), c="r", label='KDF')
ax[4][2].plot(sample_size, df_['mmcIn_rf_med'].ravel(), c="k", label='RF')

ax[4][2].fill_between(sample_size, df_["mmcIn_kdf_25"].ravel(), df_["mmcIn_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[4][2].fill_between(sample_size, df_["mmcIn_rf_25"].ravel(), df_["mmcIn_rf_75"].ravel(), facecolor='k', alpha=.3)

##################################################################################################
med, a_25, a_75 = calc_stat(df['kdn_mmcOut'])
med_nn, nn_25, nn_75 = calc_stat(df['nn_mmcOut'])

ax[4][3].plot(sample_size, med[1:], c="b", label='KDN')
ax[4][3].plot(sample_size, med_nn[1:], c="c", label='NN')

ax[4][3].fill_between(sample_size, a_25[1:], a_75[1:], facecolor='b', alpha=.3)
ax[4][3].fill_between(sample_size, nn_25[1:], nn_75[1:], facecolor='c', alpha=.3)

ax[4][3].set_xscale('log')
ax[4][3].set_xlabel('Sample size', fontsize=ticksize)
ax[4][3].set_ylabel('Mean Max Confidence\n (Out Distribution)', fontsize=ticksize)

right_side = ax[4][3].spines["right"]
right_side.set_visible(False)
top_side = ax[4][3].spines["top"]
top_side.set_visible(False)

ax[4][3].plot(sample_size, df_['mmcOut_kdf_med'].ravel(), c="r", label='KDF')
ax[4][3].plot(sample_size, df_['mmcOut_rf_med'].ravel(), c="k", label='RF')

ax[4][3].fill_between(sample_size, df_["mmcOut_kdf_25"].ravel(), df_["mmcOut_kdf_75"].ravel(), facecolor='r', alpha=.3)
ax[4][3].fill_between(sample_size, df_["mmcOut_rf_25"].ravel(), df_["mmcOut_rf_75"].ravel(), facecolor='k', alpha=.3)

plt.savefig('plots/simulation_res.pdf')
# %%
