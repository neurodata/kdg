#%%
from kdg import kdn
import keras
from keras import layers
from kdg.utils import generate_gaussian_parity, pdf, hellinger
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
# %%
reps = 100
sample_size = np.logspace(
        np.log10(10),
        np.log10(5000),
        num=10,
        endpoint=True,
        dtype=int
        )
covarice_types = {'diag', 'full', 'spherical'}

#%%
def experiment_kdn(sample, cov_type, criterion=None):
    network = keras.Sequential()
    network.add(layers.Dense(15, activation='relu'))
    network.add(layers.Dense(15, activation='relu'))
    network.add(layers.Dense(units=2, activation = 'softmax'))

    X, y = generate_gaussian_parity(sample, cluster_std=0.5)
    X_test, y_test = generate_gaussian_parity(1000, cluster_std=0.5)
    p = np.arange(-1,1,step=0.006)
    q = np.arange(-1,1,step=0.006)
    xx, yy = np.meshgrid(p,q)
    grid_samples = np.concatenate(
            (
                xx.reshape(-1,1),
                yy.reshape(-1,1)
            ),
            axis=1
    ) 
    model_kdn = kdn(network, covariance_types = cov_type, criterion = criterion)
    model_kdn.fit(X, y)
    proba_kdn = model_kdn.predict_proba(grid_samples)
    true_pdf_class1 = np.array([pdf(x, cov_scale=0.5) for x in grid_samples]).reshape(-1,1)
    true_pdf = np.concatenate([true_pdf_class1, 1-true_pdf_class1], axis = 1)

    error = 1 - np.mean(model_kdn.predict(X_test)==y_test)
    return hellinger(proba_kdn, true_pdf), error

def experiment_nn(sample):
    network_base = keras.Sequential()
    network_base.add(layers.Dense(15, activation='relu'))
    network_base.add(layers.Dense(15, activation='relu'))
    network_base.add(layers.Dense(units=2, activation = 'softmax'))
    network_base.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(3e-4))

    X, y = generate_gaussian_parity(sample, cluster_std=0.5)
    X_test, y_test = generate_gaussian_parity(1000, cluster_std=0.5)
    p = np.arange(-1,1,step=0.006)
    q = np.arange(-1,1,step=0.006)
    xx, yy = np.meshgrid(p,q)
    grid_samples = np.concatenate(
            (
                xx.reshape(-1,1),
                yy.reshape(-1,1)
            ),
            axis=1
    ) 
    network_base.fit(X, keras.utils.to_categorical(y), epochs=100, batch_size=32, verbose=False)
    proba_nn = network_base.predict_proba(grid_samples)
    true_pdf_class1 = np.array([pdf(x, cov_scale=0.5) for x in grid_samples]).reshape(-1,1)
    true_pdf = np.concatenate([true_pdf_class1, 1-true_pdf_class1], axis = 1)

    error = 1 - np.mean(network_base.predict(X_test)==y_test)
    return hellinger(proba_nn, true_pdf), error
    
        
# %%
for cov_type in covarice_types:
    df = pd.DataFrame()
    hellinger_dist_kdn = []
    hellinger_dist_nn = []
    err_kdn = []
    err_nn = []
    sample_list = []
    
    for sample in sample_size:
        print('Doing sample %d for %s'%(sample,cov_type))

        for ii in range(reps):
            dist, err = experiment_kdn(
                    sample,
                    cov_type=cov_type,
                    criterion=None
                    )
            hellinger_dist_kdn.append(
                    dist
                )
            err_kdn.append(
                    err
                )

            dist, err = experiment_nn(sample)
            hellinger_dist_nn.append(
                    dist
                )
            err_nn.append(
                    err
                )

        sample_list.extend([sample]*reps)

    df['hellinger dist kdn'] = hellinger_dist_kdn
    df['hellinger dist nn'] = hellinger_dist_nn
    df['error kdn'] = err_kdn
    df['error nn'] = err_nn
    df['sample'] = sample_list
    df.to_csv('simulation_res_nn_'+cov_type+'.csv')
# %%
df = pd.DataFrame()
hellinger_dist_kdn = []
hellinger_dist_nn = []
err_kdn = []
err_nn = []
sample_list = []
    
for sample in sample_size:
    print('Doing sample %d for %s'%(sample,covarice_types))

    for ii in range(reps):
        dist, err = experiment_kdn(
                sample,
                cov_type=covarice_types,
                criterion='aic'
                )
        hellinger_dist_kdn.append(
                dist
            )
        err_kdn.append(
                err
            )

        dist, err = experiment_nn(sample)
        hellinger_dist_nn.append(
                dist
            )
        err_nn.append(
                err
            )

    sample_list.extend([sample]*reps)

df['hellinger dist kdn'] = hellinger_dist_kdn
df['hellinger dist nn'] = hellinger_dist_nn
df['error kdn'] = err_kdn
df['error nn'] = err_nn
df['sample'] = sample_list
df.to_csv('simulation_res_nn_AIC.csv')

#%%
df = pd.DataFrame()
hellinger_dist_kdn = []
hellinger_dist_nn = []
err_kdn = []
err_nn = []
sample_list = []
    
for sample in sample_size:
    print('Doing sample %d for %s'%(sample,covarice_types))

    for ii in range(reps):
        dist, err = experiment_kdn(
                sample,
                cov_type=covarice_types,
                criterion='bic'
                )
        hellinger_dist_kdn.append(
                dist
            )
        err_kdn.append(
                err
            )

        dist, err = experiment_nn(sample)
        hellinger_dist_nn.append(
                dist
            )
        err_nn.append(
                err
            )

    sample_list.extend([sample]*reps)

df['hellinger dist kdn'] = hellinger_dist_kdn
df['hellinger dist nn'] = hellinger_dist_nn
df['error kdn'] = err_kdn
df['error nn'] = err_nn
df['sample'] = sample_list
df.to_csv('simulation_res_nn_BIC.csv')