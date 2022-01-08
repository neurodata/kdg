#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from kdg.utils import generate_gaussian_parity
# %%
p = np.arange(-1,1,step=0.01)
q = np.arange(-1,1,step=0.01)
xx, yy = np.meshgrid(p,q)
tmp = np.ones(xx.shape)

grid_samples = np.concatenate(
            (
                xx.reshape(-1,1),
                yy.reshape(-1,1)
            ),
            axis=1
    ) 
total_grid_points = grid_samples.shape[0]    
# %%
N = 100
epsilon = 1e-2
# %%
mc_reps = 3

def posterior_calc(N, total_grid_points):
    posterior = np.zeros(total_grid_points, dtype=float)
    X, y = generate_gaussian_parity(N)
    
    for jj in range(total_grid_points):
        points = 0
        class1 = 0
        for kk in range(N):
            if X[kk,0]<grid_samples[jj,0] + epsilon and X[kk,0]>grid_samples[jj,0] - epsilon and X[kk,1]<grid_samples[jj,1] + epsilon and X[kk,1]>grid_samples[jj,1] - epsilon:
                points += 1

                if y[kk] == 0:
                    class1 += 1

        if points == 0:
            posterior[jj] = 0.5
        else:
            posterior[jj] = class1/points

    return posterior

##########################################
posterior = Parallel(n_jobs=-1,verbose=1)(
            delayed(posterior_calc)(
                    N,
                    total_grid_points
                    ) for _ in range(mc_reps)
                )

print(posterior.shape, grid_samples.shape)
posterior = np.mean(posterior, axis=1)
df = pd.DataFrame()
df['posterior'] = posterior
df['X1'] = grid_samples[:,0]
df['X2'] = grid_samples[:,1]
df.to_csv('Gaussian_xor_pdf.csv')
# %%
'''data = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':posterior})
data = data.pivot(index='x', columns='y', values='z')


sns.set_context("talk")
fig, ax = plt.subplots(1,1, figsize=(8,8))
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax, vmin=0, vmax=1,cmap=cmap)
ax1.set_xticklabels(['-1','' , '', '', '', '', '','','','','0','','','','','','','','','1'])
ax1.set_yticklabels(['-1','' , '', '', '', '', '','','','','','','0','','','','','','','','','','','','','1'])
#ax1.set_yticklabels(['-1','' , '', '', '', '', '','','','' , '', '', '', '', '', '','','','','', '0','','' , '', '', '', '', '','','','','','','','','','','','','1'])
#ax.set_title('Estimated PDF of xor-nxor simulation data',fontsize=24)
ax.invert_yaxis()'''

# %%
