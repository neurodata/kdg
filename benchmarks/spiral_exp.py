#%%
from kdg.utils import generate_spirals, generate_gaussian_parity
from kdg import kdf,kdn
from keras import layers
import keras
# %%
network = keras.Sequential()
#network.add(layers.Dense(2, activation="relu", input_shape=(2)))
network.add(layers.Dense(3, activation='relu', input_shape=(2,)))
network.add(layers.Dense(3, activation='relu'))
network.add(layers.Dense(units=2, activation = 'softmax'))

#%%
n_estimators = 200
X, y = generate_gaussian_parity(sample, cluster_std=0.5)#generate_spirals(5000, noise=.8, n_class=2)

model_kdf = kdn(network,fit_kwargs = {
            "epochs": 100,
            "batch_size": 32,
            "verbose": False
            }) #kdf(k=1/2.5, kwargs={'n_estimators':n_estimators})
model_kdf.fit(X, y)
# %%
import seaborn as sns
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

p = np.arange(-2,2,step=0.006)
q = np.arange(-2,2,step=0.006)
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
proba_rf = model_kdf.network.predict_proba(grid_samples)

data = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':proba_kdf[:,0]})
data = data.pivot(index='x', columns='y', values='z')

data_rf = pd.DataFrame(data={'x':grid_samples[:,0], 'y':grid_samples[:,1], 'z':proba_rf[:,0]})
data_rf = data_rf.pivot(index='x', columns='y', values='z')
#%%
sns.set_context("talk")
fig, ax = plt.subplots(2,2, figsize=(16,16))
cmap= sns.diverging_palette(240, 10, n=9)
ax1 = sns.heatmap(data, ax=ax[0][0], vmin=0, vmax=1,cmap=cmap)
ax1.set_xticklabels(['-2','' , '', '', '', '', '','','','','0','','','','','','','','','2'])
ax1.set_yticklabels(['-2','' , '', '', '', '', '','','','','','','0','','','','','','','','','','','','','2'])
#ax1.set_yticklabels(['-1','' , '', '', '', '', '','','','' , '', '', '', '', '', '','','','','', '0','','' , '', '', '', '', '','','','','','','','','','','','','1'])
ax[0][0].set_title('KDF',fontsize=24)
#ax[0][0].invert_yaxis()


ax1 = sns.heatmap(data_rf, ax=ax[0][1], vmin=0, vmax=1,cmap=cmap)
ax1.set_xticklabels(['-2','' , '', '', '', '', '','','','','0','','','','','','','','','2'])
ax1.set_yticklabels(['-2','' , '', '', '', '', '','','','','','','0','','','','','','','','','','','','','2'])
#ax1.set_yticklabels(['-1','' , '', '', '', '', '','','','' , '', '', '', '', '', '','','','','', '0','','' , '', '', '', '', '','','','','','','','','','','','','1'])
ax[0][1].set_title('RF',fontsize=24)
#ax[0][1].invert_yaxis()

colors = sns.color_palette("Dark2", n_colors=2)
clr = [colors[i] for i in y]
ax[1][0].scatter(X[:, 0], X[:, 1], c=clr, s=50)

plt.savefig('plots/spiral_pdf_kdn.pdf')
plt.show()
# %%
