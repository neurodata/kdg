# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle
from keras.models import Model
from kdg import kdcnn, kdf, kdn
import pickle
from tensorflow.keras.datasets import cifar10, cifar100
import timeit
from scipy.io import loadmat
import random
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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
x_svhn = loadmat('/Users/jayantadey/svhn/train_32x32.mat')['X']
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

    x_train[:,:,:,channel] -= x_train_mean
    x_train[:,:,:,channel] /= x_train_std

    x_test[:,:,:,channel] -= x_train_mean
    x_test[:,:,:,channel] /= x_train_std

    x_cifar100[:,:,:,channel] -= x_train_mean
    x_cifar100[:,:,:,channel] /= x_train_std

    x_svhn[:,:,:,channel] -= x_train_mean
    x_svhn[:,:,:,channel] /= x_train_std

#%% Load model file
seed = 0
filename = '/Users/jayantadey/kdg/benchmarks/cifar10_experiments/resnet20_models/resnet_kdn_50000_'+str(seed)+'.pickle'

with open(filename, 'rb') as f:
    model_kdn = pickle.load(f)

model_kdn.global_bias = -5e3

proba_in = model_kdn.predict_proba(x_test, distance='Geodesic')
proba_cifar100 = model_kdn.predict_proba(x_cifar100, distance='Geodesic')
proba_svhn = model_kdn.predict_proba(x_svhn, distance='Geodesic')

proba_in_dn = model_kdn.network.predict(x_test)
proba_cifar100_dn = model_kdn.network.predict(x_cifar100)
proba_svhn_dn = model_kdn.network.predict(x_svhn)

summary = (proba_in, proba_cifar100, proba_svhn, proba_in_dn, proba_cifar100_dn, proba_svhn_dn)
file_to_save = '/Users/jayantadey/kdg/benchmarks/cifar10_experiments/results/resnet20_'+str(seed)+'.pickle'

with open(file_to_save, 'wb') as f:
    pickle.dump(summary, f)
# %% funny tests
arg_in = np.argmin(distance_cifar10,axis=1)
arg_out = np.argmin(distance_cifar100,axis=1)
md_in = []
md_out = []

for ii in range(100):
    md_in.append(
        np.sum((model.polytope_means[arg_in[ii]].ravel()-x_test[ii].ravel())**2/np.mean(model.polytope_cov[arg_in[ii]].ravel()))
    )
    md_out.append(
        np.sum((model.polytope_means[arg_out[ii]].ravel()-x_cifar100[ii].ravel())**2/np.mean(model.polytope_cov[arg_out[ii]].ravel()))
    )

# %%
sns.histplot(md_in,color='r')
sns.histplot(md_out,color='b')
# %%
model = joblib.load('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/resnet_kdn_50000_0.joblib')
#%%
x_svhn = loadmat('/Users/jayantadey/svhn/train_32x32.mat')['X']
x_ = np.zeros((100,32,32,3),dtype=float)

for ii in range(100):
    x_[ii] = x_svhn[:,:,:,ii].astype('float32')/255

x_ -= x_train_mean
#%%
x_ = x_svhn[:100]
test_ids = model._get_polytope_ids(x_)
#%%
total_polytope = len(model.polytope_means)
batchsize = total_polytope//10
indx_X2 = np.inf
polytope_ids = model._get_polytope_ids(
                    np.array(model.polytope_means[:batchsize])
                ) 
for ii in range(1,10):
    print("doing batch ", ii)
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


# %%
distances = model._compute_geodesic(test_ids, polytope_ids)
#%%
distances_ = model._compute_geodesic(test_ids, polytope_ids)
# %%
arg_min = np.argmin(distances_,axis=1)

#%%
arg_sort = np.argsort(distances_,axis=1)
#%%
import matplotlib.pyplot as plt
id = 41
sorted_arg = arg_sort[id,:]
plt.imshow(model.polytope_means[arg_min[id]]+x_train_mean)
#%%
plt.imshow(x_[id]+x_train_mean)
#%%
polytope_mean = model.polytope_means[arg_min[id]]
polytope_cov = model.polytope_cov[arg_min[id]]**1
np.sum((x_[id]-polytope_mean).ravel()*polytope_cov.ravel())
#%%
#diff = np.abs(x_[id]-x_[id])
#diff = (1-distances[id,arg_min[id]])*(model.polytope_means[arg_min[id]]-x_[id])**2
#sum = 1-distances[id,arg_min[id]]
ex = 1
diff = (1-distances[id,sorted_arg[0]])**ex* (model.polytope_means[arg_sort[id,0]]-x_[id])**2
sum = (1-distances[id,sorted_arg[0]])**ex
for ii in range(1,30):

    if ii==id:
        continue

    diff += (1-distances[id,sorted_arg[ii]])**ex* (model.polytope_means[arg_sort[id,ii]]-x_[id])**2
    sum += (1-distances[id,sorted_arg[ii]])**ex
    #print(1-distances[id,sorted_arg[ii]])

diff /= sum
plt.imshow(diff/np.max(diff.ravel()))
plt.colorbar()
#%%
plt.hist(diff.ravel())
#%%
plt.imshow(np.abs(model.polytope_means[arg_min[id]]-x_[id]))
plt.colorbar()
# %%
np.sum((model.polytope_means[arg_min[id]].ravel()-x_[id].ravel())**2)
# %%
for ii in range(10):
    print(model._compute_log_likelihood(x_[id], ii, arg_min[id]))
# %%
polytope_mean = model.polytope_means[arg_min[id]]
polytope_cov = model.polytope_cov[arg_min[id]]**1
#polytope_cov /= np.max(polytope_cov.ravel())
X = x_[id]
sum = 0
img = np.zeros((32,32,3),dtype=float)
cov_arg = np.argsort(polytope_cov)[::-1]
count = 0
for ii in range(32):
    for jj in range(32):
        for kk in range(3):
            #if ii>.9*model.feature_dim:
            #    continue
            if polytope_cov[ii,jj,kk]<np.percentile(polytope_cov.ravel(),q=95):
                #print('found')
                count += 1
                continue
            img[ii,jj,kk] = model._compute_log_likelihood_1d(X[ii,jj,kk], polytope_mean[ii,jj,kk], polytope_cov[ii,jj,kk])
                
            sum += img[ii,jj,kk]

print(sum/count)

fig, ax = plt.subplots(1,3, figsize=(24,8))
sns.heatmap(img[:,:,0], cmap='autumn', ax=ax[0])
sns.heatmap(img[:,:,1], cmap='autumn', ax=ax[1])
sns.heatmap(img[:,:,2], cmap='autumn', ax=ax[2])

crd = []
for ii in range(10):
    crd.append(model.polytope_cardinality[ii][arg_min[id]])
print(np.argmax(crd), y_test[id])

#%%
img = abs(model.polytope_means[arg_min[id]] - x_[id])
fig, ax = plt.subplots(1,3, figsize=(24,8))
sns.heatmap(img[:,:,0], cmap='autumn', ax=ax[0])
sns.heatmap(img[:,:,1], cmap='autumn', ax=ax[1])
sns.heatmap(img[:,:,2], cmap='autumn', ax=ax[2])

#%%
img = model.polytope_cov[arg_min[id]]/np.max(model.polytope_cov[arg_min[id]].ravel())
 
fig, ax = plt.subplots(1,3, figsize=(24,8))
sns.heatmap(img[:,:,0], cmap='autumn', ax=ax[0])
sns.heatmap(img[:,:,1], cmap='autumn', ax=ax[1])
sns.heatmap(img[:,:,2], cmap='autumn', ax=ax[2])

# %%
cov = 1e-2
sum = 0

for ii in range(3072):
    sum += -0.5*(X[ii]-polytope_mean[ii])**2/cov -.5*np.log(cov)

sum -= 3072*np.log(2*np.pi)/2
print(sum)
# %%
for ii in range(total_polytope):
    model.polytope_cov[ii] = 1e-2*np.ones((32,32,3),dtype=float)
# %%
plt.imshow(model.polytope_cov[9]/np.max(model.polytope_cov[9].ravel()))
# %%
