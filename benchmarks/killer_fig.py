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
n_estimators = 500
X, y = {}, {}
models  = {}
#%%
X['gxor'], y['gxor'] = generate_gaussian_parity(n_samples)
X['spiral'], y['spiral'] = generate_spirals(n_samples)
X['circle'], y['circle'] = generate_ellipse(n_samples)
X['sine'], y['sine'] = generate_sinewave(n_samples)
X['poly'], y['poly'] = generate_polynomial(n_samples, a=[1,3])

# %%
models['gxor'] = kdf(kwargs={"n_estimators": n_estimators})
models['spiral'] = kdf(kwargs={"n_estimators": n_estimators})
models['circle'] = kdf(kwargs={"n_estimators": n_estimators})
models['sine'] = kdf(kwargs={"n_estimators": n_estimators})
models['poly'] = kdf(kwargs={"n_estimators": n_estimators})

# %%
models['gxor'].fit(
    X['gxor'], y['gxor']
)

models['spiral'].fit(
    X['spiral'], y['spiral']
)

models['circle'].fit(
    X['circle'], y['circle']
)

models['sine'].fit(
    X['sine'], y['sine']
)

models['poly'].fit(
    X['poly'], y['poly']
)
# %%
fig = plt.figure(constrained_layout=True,figsize=(24,30))
gs = fig.add_gridspec(30, 24)

sns.set_context("talk")
p = np.arange(-2,2,step=0.1)
q = np.arange(-2,2,step=0.1)
xx, yy = np.meshgrid(p,q)

grid_samples = np.concatenate(
            (
                xx.reshape(-1,1),
                yy.reshape(-1,1)
            ),
            axis=1
    )



### GXOR plot ###
ax = fig.add_subplot(gs[:6,:6])
plot_2dsim(X['gxor'], y['gxor'], ax=ax)

ax = fig.add_subplot(gs[:6,6:12])
df = pd.read_csv('true_posterior/Gaussian_xor_pdf.csv')
grid_samples0 = df['X1']
grid_samples1 = df['X2']
posterior = df['posterior']
data = pd.DataFrame(data={'x':grid_samples0, 'y':grid_samples1, 'z':posterior})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
ax.set_title('True Posterior',fontsize=24)

ax = fig.add_subplot(gs[:6,12:18])
pdf_gxor = models['gxor'].rf_model.predict_proba(grid_samples)
data = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':pdf_gxor[:,0]})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
ax.set_title('RF Posterior',fontsize=24)

ax = fig.add_subplot(gs[:6,18:24])
pdf_gxor = models['gxor'].predict_proba(grid_samples)
data = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':pdf_gxor[:,0]})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
ax.set_title('KDF Posterior',fontsize=24)



### circle plot ###
ax = fig.add_subplot(gs[6:12,:6])
plot_2dsim(X['circle'], y['circle'], ax=ax)

ax = fig.add_subplot(gs[6:12,6:12])
df = pd.read_csv('true_posterior/ellipse_pdf.csv')
grid_samples0 = df['X1']
grid_samples1 = df['X2']
posterior = df['posterior']
data = pd.DataFrame(data={'x':grid_samples0, 'y':grid_samples1, 'z':posterior})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
#ax.set_title('True Posterior',fontsize=24)

ax = fig.add_subplot(gs[6:12,12:18])
pdf_circle = models['circle'].rf_model.predict_proba(grid_samples)
data = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':pdf_circle[:,0]})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
#ax.set_title('RF Posterior',fontsize=24)

ax = fig.add_subplot(gs[6:12,18:24])
pdf_circle = models['circle'].predict_proba(grid_samples)
data = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':pdf_circle[:,0]})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
#ax.set_title('KDF Posterior',fontsize=24)




### spiral plot ###
ax = fig.add_subplot(gs[12:18,:6])
plot_2dsim(X['spiral'], y['spiral'], ax=ax)

ax = fig.add_subplot(gs[12:18,6:12])
df = pd.read_csv('true_posterior/spiral_pdf.csv')
grid_samples0 = df['X1']
grid_samples1 = df['X2']
posterior = df['posterior']
data = pd.DataFrame(data={'x':grid_samples0, 'y':grid_samples1, 'z':posterior})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
#ax.set_title('True Posterior',fontsize=24)

ax = fig.add_subplot(gs[12:18,12:18])
pdf_spiral = models['spiral'].rf_model.predict_proba(grid_samples)
data = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':pdf_spiral[:,0]})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
#ax.set_title('RF Posterior',fontsize=24)

ax = fig.add_subplot(gs[12:18,18:24])
pdf_spiral = models['spiral'].predict_proba(grid_samples)
data = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':pdf_spiral[:,0]})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
#ax.set_title('KDF Posterior',fontsize=24)



### sinewave plot ###
ax = fig.add_subplot(gs[18:24,:6])
plot_2dsim(X['sine'], y['sine'], ax=ax)

ax = fig.add_subplot(gs[18:24,6:12])
df = pd.read_csv('true_posterior/sinewave_pdf.csv')
grid_samples0 = df['X1']
grid_samples1 = df['X2']
posterior = df['posterior']
data = pd.DataFrame(data={'x':grid_samples0, 'y':grid_samples1, 'z':posterior})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
#ax.set_title('True Posterior',fontsize=24)

ax = fig.add_subplot(gs[18:24,12:18])
pdf_sine = models['sine'].rf_model.predict_proba(grid_samples)
data = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':pdf_sine[:,0]})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
#ax.set_title('RF Posterior',fontsize=24)

ax = fig.add_subplot(gs[18:24,18:24])
pdf_sine = models['sine'].predict_proba(grid_samples)
data = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':pdf_sine[:,0]})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
#ax.set_title('KDF Posterior',fontsize=24)



### polynomial plot ###
ax = fig.add_subplot(gs[24:30,:6])
plot_2dsim(X['poly'], y['poly'], ax=ax)

ax = fig.add_subplot(gs[24:30,6:12])
df = pd.read_csv('true_posterior/polynomial_pdf.csv')
grid_samples0 = df['X1']
grid_samples1 = df['X2']
posterior = df['posterior']
data = pd.DataFrame(data={'x':grid_samples0, 'y':grid_samples1, 'z':posterior})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
#ax.set_title('True Posterior',fontsize=24)

ax = fig.add_subplot(gs[24:30,12:18])
pdf_poly = models['poly'].rf_model.predict_proba(grid_samples)
data = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':pdf_poly[:,0]})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
#ax.set_title('RF Posterior',fontsize=24)

ax = fig.add_subplot(gs[24:30,18:24])
pdf_poly = models['poly'].predict_proba(grid_samples)
data = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':pdf_poly[:,0]})
data = data.pivot(index='x', columns='y', values='z')
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
#ax.set_title('KDF Posterior',fontsize=24)
# %%
