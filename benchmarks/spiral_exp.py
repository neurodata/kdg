#%%
from kdg.utils import generate_spirals
from kdg import kdf
# %%
n_estimators = 500
X, y = generate_spirals(500, n_class=3)

model_kdf = kdf(kwargs={'n_estimators':n_estimators})
model_kdf.fit(X, y)
# %%
import seaborn as sns
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

p = np.arange(-1,1,step=0.006)
q = np.arange(-1,1,step=0.006)
xx, yy = np.meshgrid(p,q)
tmp = np.ones(xx.shape)

grid_samples = np.concatenate(
            (
                xx.reshape(-1,1),
                yy.reshape(-1,1)
            ),
            axis=1
    ) 
    
proba_kdf = model_kdf.predict_proba(grid_samples)
proba_rf = model_kdf.rf_model.predict_proba(grid_samples)

data = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':proba_kdf[:,0]})
data = data.pivot(index='x', columns='y', values='z')

data_rf = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':proba_rf[:,0]})
data_rf = data_rf.pivot(index='x', columns='y', values='z')

sns.set_context("talk")
fig, ax = plt.subplots(1,2, figsize=(16,8))
#cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax[0])
ax1.set_xticklabels(['-1','' , '', '', '', '', '','','','','0','','','','','','','','','1'])
ax1.set_yticklabels(['-1','' , '', '', '', '', '','','','','','','0','','','','','','','','','','','','','1'])
#ax1.set_yticklabels(['-1','' , '', '', '', '', '','','','' , '', '', '', '', '', '','','','','', '0','','' , '', '', '', '', '','','','','','','','','','','','','1'])
ax[0].set_title('KDF',fontsize=24)
ax[0].invert_yaxis()

ax1 = sns.heatmap(data_rf, ax=ax[1])
ax1.set_xticklabels(['-1','' , '', '', '', '', '','','','','0','','','','','','','','','1'])
ax1.set_yticklabels(['-1','' , '', '', '', '', '','','','','','','0','','','','','','','','','','','','','1'])
#ax1.set_yticklabels(['-1','' , '', '', '', '', '','','','' , '', '', '', '', '', '','','','','', '0','','' , '', '', '', '', '','','','','','','','','','','','','1'])
ax[1].set_title('RF',fontsize=24)
ax[1].invert_yaxis()
# %%
