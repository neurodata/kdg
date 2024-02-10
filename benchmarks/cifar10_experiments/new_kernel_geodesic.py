# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle
from keras.models import Model
from kdg import kdf, kdn, get_ece
import pickle
from tensorflow.keras.datasets import cifar10, cifar100
import timeit
from scipy.io import loadmat
import random
import joblib
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.metrics import roc_auc_score
import multiprocessing
from tensorflow.keras import backend as bknd
from scipy.spatial.distance import cdist as dist
from tqdm import tqdm
import pickle
import os
import gc
from joblib.externals.loky import get_reusable_executor
#%%
def fpr_at_95_tpr(conf_in, conf_out):
    TPR = 95
    PERC = np.percentile(conf_in, 100-TPR)
    #FP = np.sum(conf_out >=  PERC)
    FPR = np.sum(conf_out >=  PERC)/len(conf_out)
    return FPR, PERC
#%%
# Load the CIFAR10 and CIFAR100 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(_, _), (x_cifar100, y_cifar100) = cifar100.load_data()
x_noise = np.random.random_integers(0,high=255,size=(1000,32,32,3)).astype('float')/255.0
x_svhn = loadmat('/Users/jayantadey/svhn/train_32x32.mat')['X']
y_svhn = loadmat('/Users/jayantadey/svhn/train_32x32.mat')['y']
test_ids =  random.sample(range(0, x_svhn.shape[3]), 2000)
x_svhn = x_svhn[:,:,:,test_ids].astype('float32')
x_tmp = np.zeros((2000,32,32,3), dtype=float)

for ii in range(2000):
   x_tmp[ii,:,:,:] = x_svhn[:,:,:,ii]

x_svhn = x_tmp
del x_tmp
# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_cifar100 = x_cifar100.astype('float32') / 255
x_svhn = x_svhn.astype('float32') / 255


for channel in range(3):
    x_train_mean = np.mean(x_train[:,:,:,channel])
    x_train_std = np.std(x_train[:,:,:,channel])

    print(x_train_mean, x_train_std)
    
    x_train[:,:,:,channel] -= x_train_mean
    x_train[:,:,:,channel] /= x_train_std

    x_test[:,:,:,channel] -= x_train_mean
    x_test[:,:,:,channel] /= x_train_std

    x_cifar100[:,:,:,channel] -= x_train_mean
    x_cifar100[:,:,:,channel] /= x_train_std

    x_svhn[:,:,:,channel] -= x_train_mean #+ 1
    x_svhn[:,:,:,channel] /= x_train_std

    x_noise[:,:,:,channel] -= x_train_mean
    x_noise[:,:,:,channel] /= x_train_std

    im[:,:,:,channel] -= x_train_mean
    im[:,:,:,channel] /= x_train_std

#%%
def _compute_geodesic(model, polytope_id_test, polytope_ids, batch=-1, save_temp=False):
    
    if save_temp:
        try:
            os.mkdir('temp')
        except:
            print('Temporary result dirctory already exists!!!')

    if batch == -1:
        batch = 10

    total_layers = len(model.network_shape)
    total_test_samples = len(polytope_id_test)
    total_polytopes = len(polytope_ids)
    id_thresholds = np.zeros(total_layers+1,dtype=int)
    id_thresholds[1:] = np.cumsum(model.network_shape)

    sample_per_batch = total_test_samples//batch
    
    print("Calculating Geodesic...")
    w = np.ones((total_test_samples, total_polytopes), dtype=float)
    indx = [jj*sample_per_batch for jj in range(batch+1)]
    if indx[-1]<total_test_samples:
        indx.append(
            total_test_samples
        )
    for ii in tqdm(range(1,total_layers)):
        
        if save_temp:
            if os.path.exists('temp/temp'+str(ii)+'.pickle'):
                with open('temp/temp'+str(ii)+'.pickle','rb') as f:
                    w_ = pickle.load(f)
            else:
                w_ = 1-np.array(Parallel(n_jobs=batch, prefer="threads")(
                                delayed(dist)(
                                            polytope_id_test[indx[jj]:indx[jj+1],id_thresholds[ii]:id_thresholds[ii+1]],
                                            polytope_ids[:,id_thresholds[ii]:id_thresholds[ii+1]],
                                            'hamming'
                                        ) for jj in range(len(indx)-1)
                                )
                )
                get_reusable_executor().shutdown(wait=True)
                gc.collect()
                w_ = np.concatenate(w_, axis=0) 
                
                with open('temp/temp'+str(ii)+'.pickle','wb') as f:
                        pickle.dump(w_, f)
        else:
            w_ = 1-np.array(Parallel(n_jobs=batch, prefer="threads")(
                        delayed(dist)(
                                    polytope_id_test[indx[jj]:indx[jj+1],id_thresholds[ii]:id_thresholds[ii+1]],
                                    polytope_ids[:,id_thresholds[ii]:id_thresholds[ii+1]],
                                    'hamming'
                                ) for jj in range(len(indx)-1)
                        )
            )
            get_reusable_executor().shutdown(wait=True)
            gc.collect()
            w_ = np.concatenate(w_, axis=0) 
            

        w = w*w_
        del w_
        
        
    return 1 - w

def _compute_log_likelihood(model, distance, label, polytope_idx):
    polytope_cov = 1e-12#np.mean(model.polytope_cov[polytope_idx].reshape(-1))
    #print(polytope_cov, polytope_idx, distance)

    likelihood = (1-distance)**2/(2*polytope_cov) - .5*np.log(2*np.pi*polytope_cov)
    likelihood += np.log(model.polytope_cardinality[label][polytope_idx]) -\
            np.log(model.total_samples_this_label[label])
    return likelihood

def predict_proba(model, X, distance = 'Euclidean', return_likelihood=False, n_jobs=-1):
    r"""
        Calculate posteriors using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        #X = check_array(X)
        
    total_polytope = len(model.polytope_means)
    log_likelihoods = np.zeros(
        (np.size(X,0), len(model.labels)),
        dtype=float
    )
    
    print('Calculating distance')
    if distance == 'Euclidean':
        distance = model._compute_euclidean(X)
        polytope_idx = np.argmin(distance, axis=1)
    elif distance == 'Geodesic':
        total_polytope = len(model.polytope_means)
        batch = total_polytope//1000 + 1
        batchsize = total_polytope//batch
        polytope_ids = model._get_polytope_ids(
                np.array(model.polytope_means[:batchsize])
            ) 

        indx_X2 = np.inf
        for ii in range(1,batch):
            #print("doing batch ", ii)
            indx_X1 = ii*batchsize
            indx_X2 = (ii+1)*batchsize
            polytope_ids = np.concatenate(
                (polytope_ids,
                model._get_polytope_ids(
                np.array(model.polytope_means[indx_X1:indx_X2])
                )),
                axis=0
            )
        
        if indx_X2 < len(model.polytope_means):
            polytope_ids = np.concatenate(
                    (polytope_ids,
                    model._get_polytope_ids(
                np.array(model.polytope_means[indx_X2:]))),
                    axis=0
                )

        total_sample = X.shape[0]
        batch = total_sample//1000 + 1
        batchsize = total_sample//batch
        test_ids = model._get_polytope_ids(X[:batchsize]) 

        indx_X2 = np.inf
        for ii in range(1,batch):
            #print("doing batch ", ii)
            indx_X1 = ii*batchsize
            indx_X2 = (ii+1)*batchsize
            test_ids = np.concatenate(
                (test_ids,
                model._get_polytope_ids(X[indx_X1:indx_X2])),
                axis=0
            )
        
        if indx_X2 < X.shape[0]:
            test_ids = np.concatenate(
                    (test_ids,
                    model._get_polytope_ids(X[indx_X2:])),
                    axis=0
                )
            
        print('Polytope extracted!')
        ####################################
        batch = total_sample//50000 + 1
        batchsize = total_sample//batch
        polytope_idx = []
        distances = []

        indx = [jj*batchsize for jj in range(batch+1)]
        if indx[-1] < total_sample:
            indx.append(total_sample)

        print(indx, 'indx')
        for ii in range(len(indx)-1):
            distance = model._compute_geodesic(
                        test_ids[indx[ii]:indx[ii+1]],
                        polytope_ids,
                        batch=n_jobs
                    )
            print(indx[ii],distance.shape, polytope_ids.shape)
            idx_min = np.argmin(
                    distance, axis=1
                )
            print(idx_min)
            distances.extend(distance[:,idx_min]) 
            print(distances, 'distances')   
            polytope_idx.extend(
                list(idx_min)
            )
    else:
        raise ValueError("Unknown distance measure!")
    
    for ii,label in enumerate(model.labels):
        for jj in range(X.shape[0]):
            p = _compute_log_likelihood(model, distances[jj], label, polytope_idx[jj])
            print(p)
            log_likelihoods[jj, ii] = p
            max_pow = max(log_likelihoods[jj, ii], model.global_bias)
            log_likelihoods[jj, ii] = np.log(
                (np.exp(log_likelihoods[jj, ii] - max_pow)\
                    + np.exp(model.global_bias - max_pow))
                    *model.prior[label]
            ) + max_pow
            
    max_pow = np.nan_to_num(
        np.max(log_likelihoods, axis=1).reshape(-1,1)@np.ones((1,len(model.labels)))
    )

    if return_likelihood:
        likelihood = np.exp(log_likelihoods)

    log_likelihoods -= max_pow
    likelihoods = np.exp(log_likelihoods)

    total_likelihoods = np.sum(likelihoods, axis=1)

    proba = (likelihoods.T/total_likelihoods).T
    
    if return_likelihood:
        return proba, likelihood
    else:
        return proba

# %%
seeds = [100]

# %%
for seed in seeds: 
    print('doing seed ',seed)
    filename = '/Users/jayantadey/kdg/benchmarks/cifar10_experiments/resnet20_models/resnet_kdn_pretrained_50000_'+str(seed)+'.joblib'
    model_kdn = joblib.load(filename)
# %%
p = predict_proba(model_kdn, x_test[:10], distance='Geodesic')
# %%
X = im.reshape(1,32,32,3)#x_svhn[:100]

total_polytope = len(model_kdn.polytope_means)
log_likelihoods = np.zeros(
    (np.size(X,0), len(model_kdn.labels)),
    dtype=float
)

print('Calculating distance')

total_polytope = len(model_kdn.polytope_means)
batch = total_polytope//1000 + 1
batchsize = total_polytope//batch
polytope_ids = model_kdn._get_polytope_ids(
        np.array(model_kdn.polytope_means[:batchsize])
    ) 

indx_X2 = np.inf
for ii in range(1,batch):
    #print("doing batch ", ii)
    indx_X1 = ii*batchsize
    indx_X2 = (ii+1)*batchsize
    polytope_ids = np.concatenate(
        (polytope_ids,
        model_kdn._get_polytope_ids(
        np.array(model_kdn.polytope_means[indx_X1:indx_X2])
        )),
        axis=0
    )

if indx_X2 < len(model_kdn.polytope_means):
    polytope_ids = np.concatenate(
            (polytope_ids,
            model_kdn._get_polytope_ids(
        np.array(model_kdn.polytope_means[indx_X2:]))),
            axis=0
        )

total_sample = X.shape[0]
batch = total_sample//1000 + 1
batchsize = total_sample//batch
test_ids = model_kdn._get_polytope_ids(X[:batchsize]) 

indx_X2 = np.inf
for ii in range(1,batch):
    #print("doing batch ", ii)
    indx_X1 = ii*batchsize
    indx_X2 = (ii+1)*batchsize
    test_ids = np.concatenate(
        (test_ids,
        model_kdn._get_polytope_ids(X[indx_X1:indx_X2])),
        axis=0
    )

if indx_X2 < X.shape[0]:
    test_ids = np.concatenate(
            (test_ids,
            model_kdn._get_polytope_ids(X[indx_X2:])),
            axis=0
        )
    
print('Polytope extracted!')
####################################
batch = total_sample//50000 + 1
batchsize = total_sample//batch
polytope_idx = []
distances = []

indx = [jj*batchsize for jj in range(batch+1)]
if indx[-1] < total_sample:
    indx.append(total_sample)

#print(indx, 'indx')
for ii in range(len(indx)-1):
    distance = _compute_geodesic(
                model_kdn,
                test_ids[indx[ii]:indx[ii+1]],
                polytope_ids,
                batch=-1
            )
    #print(indx[ii],distance.shape, polytope_ids.shape)
    idx_min = np.argmin(
            distance, axis=1
        )
    
    for kk, id in enumerate(idx_min):
        distances.extend([distance[kk,id]]) 

    #print(distances, 'distances')   
    polytope_idx.extend(
        list(idx_min)
    )

for ii,label in enumerate(model_kdn.labels):
    for jj in range(X.shape[0]):
        p = _compute_log_likelihood(model_kdn, distances[jj], label, polytope_idx[jj])
        print(p)
        log_likelihoods[jj, ii] = p
        max_pow = max(log_likelihoods[jj, ii], model_kdn.global_bias)
        log_likelihoods[jj, ii] = np.log(
            (np.exp(log_likelihoods[jj, ii] - max_pow)\
                + np.exp(model_kdn.global_bias - max_pow))
                *model_kdn.prior[label]
        ) + max_pow
        
max_pow = np.nan_to_num(
    np.max(log_likelihoods, axis=1).reshape(-1,1)@np.ones((1,len(model_kdn.labels)))
)


log_likelihoods -= max_pow
likelihoods = np.exp(log_likelihoods)

total_likelihoods = np.sum(likelihoods, axis=1)

proba = (likelihoods.T/total_likelihoods).T

# %%
