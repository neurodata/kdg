#%%
import numpy as np
from kdg.utils import generate_gaussian_parity, generate_ellipse, generate_spirals, generate_sinewave, generate_polynomial, trunk_sim
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

def get_stat(data, reps=10):
    total_dim = len(data)//reps

    med = []
    quantile_25 = []
    quantile_75 = []
    for ii in range(total_dim):
        med.append(
                np.median(data[ii*reps:(ii+1)*reps])
            )
        quantile_25.append(
        np.quantile(data[ii*reps:(ii+1)*reps], [0.25])[0]
            )
        quantile_75.append(
        np.quantile(data[ii*reps:(ii+1)*reps], [0.75])[0]
            )
    return med, quantile_25, quantile_75

#%%
n_samples = 5000
X, y = {}, {}

X['gxor'], y['gxor'] = generate_gaussian_parity(n_samples)
X['spiral'], y['spiral'] = generate_spirals(n_samples)
X['circle'], y['circle'] = generate_ellipse(n_samples)
X['sinewave'], y['sinewave'] = generate_sinewave(n_samples)
X['polynomial'], y['polynomial'] = generate_polynomial(n_samples, a=[1,3])
X['trunk'], y['trunk'] = trunk_sim(1000, p_star=2, p=2)
# %%
simulations = ['gxor', 'spiral', 'circle', 'sinewave', 'polynomial']
models = ['kdf', 'kdn']
sample_size = [50, 100, 500, 1000, 5000, 10000]
r = r = np.arange(0,10.5,.5)
linewidth = [6,3]

#sns.set_context('talk')
ticksize = 50
labelsize = 50
fig1, ax = plt.subplots(6, 7, figsize=(65, 50))

for ii, model in enumerate(models):
    for jj, simulation in enumerate(simulations):

        if ii == 0:
            plot_2dsim(X[simulation], y[simulation],ax=ax[jj][ii])

            ax[jj][ii].set_xlim([-2,2])
            ax[jj][ii].set_ylim([-2,2])
            ax[jj][ii].set_xticks([])
            ax[jj][ii].set_yticks([-2,-1,0,1,2])

            ax[jj][ii].tick_params(labelsize=ticksize)

        filename = '/Users/jayantadey/kdg/benchmarks/'+model + '_simulations/results/' + simulation + '.pickle'
        df = unpickle(filename)

        clr = 'r' if model == 'kdn' else 'b'
        parent = 'rf' if model == 'kdf' else 'dn'
        model_key = 'error_' + model + '_'
        parent_key = 'error_' + parent + '_' 

        ax[jj][ii*3+1].plot(sample_size, df[model_key+'geod_med'], c=clr, linewidth=linewidth[0], label=model.upper()+'-Geodesic')
        ax[jj][ii*3+1].plot(sample_size, df[model_key+'med'], c=clr, linewidth=linewidth[0], label=model.upper()+'-Euclidean', linestyle='dashed')
        ax[jj][ii*3+1].plot(sample_size, df[parent_key+'med'], c="k", label=parent.upper())
        ax[jj][ii*3+1].fill_between(sample_size, df[model_key+'25'], df[model_key+'75'], facecolor=clr, alpha=.3)
        ax[jj][ii*3+1].fill_between(sample_size, df[model_key+'geod_25'], df[model_key+'geod_75'], facecolor=clr, alpha=.3)
        ax[jj][ii*3+1].fill_between(sample_size, df[parent_key+'25'], df[parent_key+'75'], facecolor='k', alpha=.3)

        ax[jj][ii*3+1].set_xscale('log')

        if jj==4:
            ax[jj][ii*3+1].set_xlabel('Sample size', fontsize=labelsize)
        else:
            ax[jj][ii*3+1].set_xticks([])

        #ax[jj][ii*3+1].set_ylabel('Error', fontsize=labelsize)

        tot_val = np.concatenate((df[model_key+'med'], df[parent_key+'med']))
        min_val = np.round(np.min(tot_val),1)
        max_val = np.round(np.max(tot_val),1)
        
        ax[jj][ii*3+1].set_yticks([min_val,max_val])

        if len(simulation)>8:
            offset=.1
        elif len(simulation)>6:
            offset= 0
        else:
            offset= min_val+(max_val-min_val)/3

        if ii == 0:
            ax[jj][0].text(-3.5,offset, simulation[0].upper()+simulation[1:], fontsize=labelsize+10, rotation=90)
            
        ax[jj][ii*3+1].tick_params(labelsize=ticksize)
        right_side = ax[jj][ii*3+1].spines["right"]
        right_side.set_visible(False)
        top_side = ax[jj][ii*3+1].spines["top"]
        top_side.set_visible(False)

        model_key = 'hellinger_' + model + '_'
        parent_key = 'hellinger_' + parent + '_' 
        ax[jj][ii*3+2].plot(sample_size, df[model_key+'geod_med'], c=clr, linewidth=linewidth[0], label=model.upper()+'-Geodesic')
        ax[jj][ii*3+2].plot(sample_size, df[model_key+'med'], c=clr, linewidth=linewidth[0], label=model.upper()+'-Euclidean', linestyle='dashed')
        ax[jj][ii*3+2].plot(sample_size, df[parent_key+'med'], c="k", label=parent.upper())
        ax[jj][ii*3+2].fill_between(sample_size, df[model_key+'25'], df[model_key+'75'], facecolor=clr, alpha=.3)
        ax[jj][ii*3+2].fill_between(sample_size, df[model_key+'geod_25'], df[model_key+'geod_75'], facecolor=clr, alpha=.3)
        ax[jj][ii*3+2].fill_between(sample_size, df[parent_key+'25'], df[parent_key+'75'], facecolor='k', alpha=.3)

        ax[jj][ii*3+2].set_xscale('log')

        if jj==4:
            ax[jj][ii*3+2].set_xlabel('Sample Size', fontsize=labelsize)
        else:
            ax[jj][ii*3+2].set_xticks([])

        #ax[jj][ii*3+2].set_ylabel('Hellinger Dist.', fontsize=labelsize)

        tot_val = np.concatenate((df[model_key+'med'], df[parent_key+'med']))
        min_val = np.round(np.min(tot_val),1)
        max_val = np.round(np.max(tot_val),1)

        if min_val == max_val:
            max_val += .1
            
        ax[jj][ii*3+2].set_yticks([min_val, max_val])
        ax[jj][ii*3+2].tick_params(labelsize=ticksize)

        '''if jj==0:
            ax[jj][ii*3+2].set_title(model.upper() + ' and '+parent.upper(), fontsize=labelsize+30)'''

        if jj==4 and ii==0:
            leg = ax[jj][ii*3+2].legend(bbox_to_anchor=(0.4, 0.07), bbox_transform=plt.gcf().transFigure,
                        ncol=3, loc='upper center', fontsize=labelsize)
            leg.get_frame().set_linewidth(0.0)

            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)

        if jj==4 and ii==1:
            leg = ax[jj][ii*3+2].legend(bbox_to_anchor=(0.75, 0.07), bbox_transform=plt.gcf().transFigure,
                        ncol=3, loc='upper center', fontsize=labelsize)
            leg.get_frame().set_linewidth(0.0)

            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)

        right_side = ax[jj][ii*3+2].spines["right"]
        right_side.set_visible(False)
        top_side = ax[jj][ii*3+2].spines["top"]
        top_side.set_visible(False)

        model_key = 'mmcOut_' + model + '_'
        parent_key = 'mmcOut_' + parent + '_' 
        ax[jj][ii*3+3].plot(r, np.array(df[model_key+'geod_med']).ravel(), c=clr, linewidth=linewidth[0], label=model.upper()+'-Geodesic')
        ax[jj][ii*3+3].plot(r, np.array(df[model_key+'med']).ravel(), c=clr, linewidth=linewidth[0], label=model.upper()+'-Euclidean', linestyle='dashed')
        ax[jj][ii*3+3].plot(r, np.array(df[parent_key+'med']).ravel(), c="k", label=parent.upper())
        ax[jj][ii*3+3].fill_between(r, np.array(df[model_key+'25']).ravel(), np.array(df[model_key+'75']).ravel(), facecolor=clr, alpha=.3)
        ax[jj][ii*3+3].fill_between(r, np.array(df[model_key+'geod_25']).ravel(), np.array(df[model_key+'geod_75']).ravel(), facecolor=clr, alpha=.3)
        ax[jj][ii*3+3].fill_between(r, np.array(df[parent_key+'25']).ravel(), np.array(df[parent_key+'75']).ravel(), facecolor='k', alpha=.3)

        if jj==4:
            ax[jj][ii*3+3].set_xlabel('Distance', fontsize=labelsize)
            ax[jj][ii*3+3].set_xticks([0,2,4,6,8,10])
        else:
            ax[jj][ii*3+3].set_xticks([])

        #ax[jj][ii*3+3].set_ylabel('Mean Max Conf.', fontsize=labelsize)
        ax[jj][ii*3+3].set_yticks([.5,1])
        
        ax[jj][ii*3+3].tick_params(labelsize=ticksize)
        right_side = ax[jj][ii*3+3].spines["right"]
        right_side.set_visible(False)
        top_side = ax[jj][ii*3+3].spines["top"]
        top_side.set_visible(False)

df_trunk = unpickle('/Users/jayantadey/kdg/benchmarks/high_dim_exp/trunk2.pickle')
dim = np.unique(df_trunk['dimension'])


plot_2dsim(X['trunk'], y['trunk'],ax=ax[5][0])

ax[5][0].set_xlim([-2,2])
ax[5][0].set_ylim([-2,2])
ax[5][0].set_xticks([-2,-1,0,1,2])
ax[5][0].set_yticks([-2,-1,0,1,2])
ax[5][0].tick_params(labelsize=ticksize)


err_rf_med, err_rf_25, err_rf_75 = get_stat(df_trunk['err_rf'])
err_kdf_med, err_kdf_25, err_kdf_75 = get_stat(df_trunk['err_kdf'])
err_kdf_geod_med, err_kdf_geod_25, err_kdf_geod_75 = get_stat(df_trunk['err_kdf_geodesic'])
ax[5][1].plot(dim, err_kdf_geod_med, c='b', linewidth=linewidth[0], label='KDF-Geodesic')
ax[5][1].plot(dim, err_kdf_med, c='b', linewidth=linewidth[0], label='KDF-Euclidean', linestyle='dashed')
ax[5][1].plot(dim, err_rf_med, c="k", label='RF')
ax[5][1].fill_between(dim, err_kdf_25, err_kdf_75, facecolor='b', alpha=.3)
ax[5][1].fill_between(dim, err_kdf_25, err_kdf_25, facecolor='b', alpha=.3)
ax[5][1].fill_between(dim, err_rf_25, err_rf_75, facecolor='k', alpha=.3)

ax[5][1].set_xscale('log')

ax[5][1].tick_params(labelsize=ticksize)
ax[5][0].text(-3.5, 0, 'Trunk', fontsize=labelsize+10, rotation=90)
#ax[5][1].set_ylabel('Error', fontsize=labelsize)
ax[5][1].set_xlabel('Dimensions', fontsize=labelsize)

right_side = ax[5][1].spines["right"]
right_side.set_visible(False)
top_side = ax[5][1].spines["top"]
top_side.set_visible(False)


err_rf_med, err_rf_25, err_rf_75 = get_stat(df_trunk['rf_hellinger'])
err_kdf_med, err_kdf_25, err_kdf_75 = get_stat(df_trunk['kdf_hellinger'])
err_kdf_geod_med, err_kdf_geod_25, err_kdf_geod_75 = get_stat(df_trunk['kdf_geodesic_hellinger'])
ax[5][2].plot(dim, err_kdf_geod_med, c='b', linewidth=linewidth[0], label='KDF-Geodesic')
ax[5][2].plot(dim, err_kdf_med, c='b', linewidth=linewidth[0], label='KDF-Euclidean', linestyle='dashed')
ax[5][2].plot(dim, err_rf_med, c="k", label='RF')
ax[5][2].fill_between(dim, err_kdf_25, err_kdf_75, facecolor='b', alpha=.3)
ax[5][2].fill_between(dim, err_kdf_25, err_kdf_25, facecolor='b', alpha=.3)
ax[5][2].fill_between(dim, err_rf_25, err_rf_75, facecolor='k', alpha=.3)

ax[5][2].set_xscale('log')

ax[5][2].tick_params(labelsize=ticksize)
#ax[5][2].set_ylabel('Hellinger Dist.', fontsize=labelsize)
ax[5][2].set_xlabel('Dimensions', fontsize=labelsize)

right_side = ax[5][2].spines["right"]
right_side.set_visible(False)
top_side = ax[5][2].spines["top"]
top_side.set_visible(False)




err_rf_med, err_rf_25, err_rf_75 = get_stat(df_trunk['conf_rf_ood'])
err_kdf_med, err_kdf_25, err_kdf_75 = get_stat(df_trunk['conf_kdf_ood'])
err_kdf_geod_med, err_kdf_geod_25, err_kdf_geod_75 = get_stat(df_trunk['conf_kdf_geodesic_ood'])
ax[5][3].plot(dim, err_kdf_geod_med, c='b', linewidth=linewidth[0], label='KDF-Geodesic')
ax[5][3].plot(dim, err_kdf_med, c='b', linewidth=linewidth[0], label='KDF-Euclidean', linestyle='dashed')
ax[5][3].plot(dim, err_rf_med, c="k", label='RF')
ax[5][3].fill_between(dim, err_kdf_25, err_kdf_75, facecolor='b', alpha=.3)
ax[5][3].fill_between(dim, err_kdf_25, err_kdf_25, facecolor='b', alpha=.3)
ax[5][3].fill_between(dim, err_rf_25, err_rf_75, facecolor='k', alpha=.3)

ax[5][3].set_xscale('log')

ax[5][3].tick_params(labelsize=ticksize)
#ax[5][3].set_ylabel('Mean Max Conf.', fontsize=labelsize)
ax[5][3].set_xlabel('Dimensions', fontsize=labelsize)

right_side = ax[5][3].spines["right"]
right_side.set_visible(False)
top_side = ax[5][3].spines["top"]
top_side.set_visible(False)


err_dn_med, err_dn_25, err_dn_75 = get_stat(df_trunk['err_dn'])
err_kdn_med, err_kdn_25, err_kdn_75 = get_stat(df_trunk['err_kdn'])
err_kdn_geod_med, err_kdn_geod_25, err_kdn_geod_75 = get_stat(df_trunk['err_kdn_geodesic'])
ax[5][4].plot(dim, err_kdn_geod_med, c='r', linewidth=linewidth[0], label='KDN-Geodesic')
ax[5][4].plot(dim, err_kdn_med, c='r', linewidth=linewidth[0], label='KDN-Euclidean', linestyle='dashed')
ax[5][4].plot(dim, err_dn_med, c="k", label='DN')
ax[5][4].fill_between(dim, err_kdn_25, err_kdn_75, facecolor='r', alpha=.3)
ax[5][4].fill_between(dim, err_kdn_25, err_kdn_25, facecolor='r', alpha=.3)
ax[5][4].fill_between(dim, err_dn_25, err_dn_75, facecolor='k', alpha=.3)

ax[5][4].set_xscale('log')

ax[5][4].tick_params(labelsize=ticksize)
#ax[5][4].set_ylabel('Error', fontsize=labelsize)
ax[5][4].set_xlabel('Dimensions', fontsize=labelsize)

right_side = ax[5][4].spines["right"]
right_side.set_visible(False)
top_side = ax[5][4].spines["top"]
top_side.set_visible(False)


err_dn_med, err_dn_25, err_dn_75 = get_stat(df_trunk['dn_hellinger'])
err_kdn_med, err_kdn_25, err_kdn_75 = get_stat(df_trunk['kdn_hellinger'])
err_kdn_geod_med, err_kdn_geod_25, err_kdn_geod_75 = get_stat(df_trunk['kdn_geodesic_hellinger'])
ax[5][5].plot(dim, err_kdn_geod_med, c='r', linewidth=linewidth[0], label='KDN-Geodesic')
ax[5][5].plot(dim, err_kdn_med, c='r', linewidth=linewidth[0], label='KDN-Euclidean', linestyle='dashed')
ax[5][5].plot(dim, err_dn_med, c="k", label='DN')
ax[5][5].fill_between(dim, err_kdn_25, err_kdn_75, facecolor='r', alpha=.3)
ax[5][5].fill_between(dim, err_kdn_25, err_kdn_25, facecolor='r', alpha=.3)
ax[5][5].fill_between(dim, err_dn_25, err_dn_75, facecolor='k', alpha=.3)

ax[5][5].set_xscale('log')

ax[5][5].tick_params(labelsize=ticksize)
#ax[5][5].set_ylabel('Hellinger Dist.', fontsize=labelsize)
ax[5][5].set_xlabel('Dimensions', fontsize=labelsize)

right_side = ax[5][5].spines["right"]
right_side.set_visible(False)
top_side = ax[5][5].spines["top"]
top_side.set_visible(False)

err_dn_med, err_dn_25, err_dn_75 = get_stat(df_trunk['conf_dn_ood'])
err_kdn_med, err_kdn_25, err_kdn_75 = get_stat(df_trunk['conf_kdn_ood'])
err_kdn_geod_med, err_kdn_geod_25, err_kdn_geod_75 = get_stat(df_trunk['conf_kdn_geodesic_ood'])
ax[5][6].plot(dim, err_kdn_geod_med, c='r', linewidth=linewidth[0], label='KDN-Geodesic')
ax[5][6].plot(dim, err_kdn_med, c='r', linewidth=linewidth[0], label='KDN-Euclidean', linestyle='dashed')
ax[5][6].plot(dim, err_dn_med, c="k", label='DN')
ax[5][6].fill_between(dim, err_kdn_25, err_kdn_75, facecolor='r', alpha=.3)
ax[5][6].fill_between(dim, err_kdn_25, err_kdn_25, facecolor='r', alpha=.3)
ax[5][6].fill_between(dim, err_dn_25, err_dn_75, facecolor='k', alpha=.3)

ax[5][6].set_xscale('log')

ax[5][6].tick_params(labelsize=ticksize)
#ax[5][6].set_ylabel('Mean Max Conf.', fontsize=labelsize)
ax[5][6].set_xlabel('Dimensions', fontsize=labelsize)

right_side = ax[5][6].spines["right"]
right_side.set_visible(False)
top_side = ax[5][6].spines["top"]
top_side.set_visible(False)


ax[0][1].set_title('Classification Error', fontsize=labelsize+4)
ax[0][2].set_title('Hellinger Distance', fontsize=labelsize+4)
ax[0][3].set_title('Mean Max Conf.', fontsize=labelsize+4)

ax[0][4].set_title('Classification Error', fontsize=labelsize+4)
ax[0][5].set_title('Hellinger Distance', fontsize=labelsize+4)
ax[0][6].set_title('Mean Max Conf.', fontsize=labelsize+4)

ax[0][0].text(0, 0, 'Simulations', fontsize=labelsize+20)
ax[0][2].text(.1, .35, 'KDF and RF', fontsize=labelsize+20)
ax[0][4].text(.5, .15, 'KDN and DN', fontsize=labelsize+20)

plt.subplots_adjust(hspace=.5,wspace=.5)
#plt.tight_layout()
plt.savefig('/Users/jayantadey/kdg/benchmarks/plots/simulation_res.pdf', bbox_inches='tight')
# %%