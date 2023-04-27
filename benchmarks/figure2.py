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
def unpickle(filename):
    with open(filename, 'rb') as f:
        df = pickle.load(f)
    return df

# %%
simulations = ['gxor', 'spiral', 'circle', 'sinewave', 'polynomial']
models = ['kdn', 'kdf']
sample_size = [50, 100, 500, 1000, 5000, 10000]
r = r = np.arange(0,10.5,.5)
linewidth = [6,3]

#sns.set_context('talk')
ticksize = 45
labelsize = 50
fig1, ax = plt.subplots(5, 6, figsize=(60, 40))

for ii, model in enumerate(models):
    for jj, simulation in enumerate(simulations):
        filename = model + '_simulations/results/' + simulation + '.pickle'
        df = unpickle(filename)

        parent = 'rf' if model == 'kdf' else 'dn'
        model_key = 'error_' + model + '_'
        parent_key = 'error_' + parent + '_' 

        
        ax[jj][ii*3+0].plot(sample_size, df[model_key+'med'], c="r", linewidth=linewidth[0], label=model.upper())
        ax[jj][ii*3+0].plot(sample_size, df[parent_key+'med'], c="k", label=parent.upper())
        ax[jj][ii*3+0].fill_between(sample_size, df[model_key+'25'], df[model_key+'75'], facecolor='r', alpha=.3)
        ax[jj][ii*3+0].fill_between(sample_size, df[parent_key+'25'], df[parent_key+'75'], facecolor='k', alpha=.3)

        ax[jj][ii*3+0].set_xscale('log')

        if jj==4:
            ax[jj][ii*3+0].set_xlabel('Sample size', fontsize=labelsize)
        else:
            ax[jj][ii*3+0].set_xticks([])

        ax[jj][ii*3+0].set_ylabel('Classification error', fontsize=labelsize)

        tot_val = np.concatenate((df[model_key+'med'], df[parent_key+'med']))
        min_val = np.round(np.min(tot_val),1)
        max_val = np.round(np.max(tot_val),1)
        
        ax[jj][ii*3+0].set_yticks([min_val,max_val])

        if len(model)>6:
            offset=0
        else:
            offset= min_val+(max_val-min_val)/3

        if ii == 0:
            ax[jj][0].text(1,offset, simulation, fontsize=labelsize+25, rotation=90)
            
        ax[jj][ii*3+0].tick_params(labelsize=ticksize)
        right_side = ax[jj][ii*3+0].spines["right"]
        right_side.set_visible(False)
        top_side = ax[jj][ii*3+0].spines["top"]
        top_side.set_visible(False)

        model_key = 'hellinger_' + model + '_'
        parent_key = 'hellinger_' + parent + '_' 
        ax[jj][ii*3+1].plot(sample_size, df[model_key+'med'], c="r", linewidth=linewidth[0], label=model.upper())
        ax[jj][ii*3+1].plot(sample_size, df[parent_key+'med'], c="k", label=parent.upper())
        ax[jj][ii*3+1].fill_between(sample_size, df[model_key+'25'], df[model_key+'75'], facecolor='r', alpha=.3)
        ax[jj][ii*3+1].fill_between(sample_size, df[parent_key+'25'], df[parent_key+'75'], facecolor='k', alpha=.3)

        ax[jj][ii*3+1].set_xscale('log')

        if jj==4:
            ax[jj][ii*3+1].set_xlabel('Sample size', fontsize=labelsize)
        else:
            ax[jj][ii*3+1].set_xticks([])

        ax[jj][ii*3+1].set_ylabel('Hellinger Dist.', fontsize=labelsize)

        tot_val = np.concatenate((df[model_key+'med'], df[parent_key+'med']))
        min_val = np.round(np.min(tot_val),1)
        max_val = np.round(np.max(tot_val),1)
        ax[jj][ii*3+1].set_yticks([min_val, max_val])
        ax[jj][ii*3+1].tick_params(labelsize=ticksize)

        if jj==0:
            ax[jj][ii*3+1].set_title(model.upper(), fontsize=labelsize+40)

        if jj==4 and ii==0:
            leg = ax[jj][ii*3+1].legend(bbox_to_anchor=(0.3, 0.07), bbox_transform=plt.gcf().transFigure,
                        ncol=2, loc='upper center', fontsize=labelsize+20)
            leg.get_frame().set_linewidth(0.0)

            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)

        if jj==4 and ii==1:
            leg = ax[jj][ii*3+1].legend(bbox_to_anchor=(0.7, 0.07), bbox_transform=plt.gcf().transFigure,
                        ncol=2, loc='upper center', fontsize=labelsize+20)
            leg.get_frame().set_linewidth(0.0)

            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)

        right_side = ax[jj][ii*3+1].spines["right"]
        right_side.set_visible(False)
        top_side = ax[jj][ii*3+1].spines["top"]
        top_side.set_visible(False)

        model_key = 'mmcOut_' + model + '_'
        parent_key = 'mmcOut_' + parent + '_' 
        ax[jj][ii*3+2].plot(r, df[model_key+'med'], c="r", linewidth=linewidth[0], label=model.upper())
        ax[jj][ii*3+2].plot(r, df[parent_key+'med'], c="k", label=parent.upper())
        ax[jj][ii*3+2].fill_between(r, df[model_key+'25'], df[model_key+'75'], facecolor='r', alpha=.3)
        ax[jj][ii*3+2].fill_between(r, df[parent_key+'25'], df[parent_key+'75'], facecolor='k', alpha=.3)

        if jj==4:
            ax[jj][ii*3+2].set_xlabel('Distance', fontsize=labelsize)
            ax[jj][ii*3+2].set_xticks([0,2,4,6,8,10])
        else:
            ax[jj][ii*3+2].set_xticks([])

        ax[jj][ii*3+2].set_ylabel('Mean Max Conf.', fontsize=labelsize)
        ax[jj][ii*3+2].set_yticks([.5,1])
        
        ax[jj][ii*3+2].tick_params(labelsize=ticksize)
        right_side = ax[jj][ii*3+2].spines["right"]
        right_side.set_visible(False)
        top_side = ax[jj][ii*3+2].spines["top"]
        top_side.set_visible(False)

plt.subplots_adjust(hspace=.2,wspace=.2)
plt.tight_layout()
plt.savefig('plots/simulation_res.pdf')
# %%
