# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle
from keras.models import Model
from kdg import kdcnn, kdf, kdn, get_ece
import pickle
from tensorflow.keras.datasets import cifar10, cifar100
import timeit
from scipy.io import loadmat
import random
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scikeras.wrappers import KerasClassifier
from sklearn.calibration import CalibratedClassifierCV as calcv
from sklearn.model_selection import train_test_split
from vit_keras import vit, utils
# %%
class nnwrapper(KerasClassifier):

    def predict_proba(self, X):
        return self.model.predict(X)
    
#%%
def fpr_at_95_tpr(conf_in, conf_out):
    TPR = 95
    PERC = np.percentile(conf_in, 100-TPR)
    #FP = np.sum(conf_out >=  PERC)
    FPR = np.sum(conf_out >=  PERC)/len(conf_out)
    return FPR, PERC

#%%
num_classes = 10
input_shape = (32,32,3)
image_size = 256 #size after resizing image
#%%
seeds = [0, 1, 2, 3, 2022]
# Load the CIFAR10 and CIFAR100 data.
(x_train_, y_train_), (x_test, y_test) = cifar10.load_data()
(_, _), (x_cifar100, y_cifar100) = cifar100.load_data()

x_train_, x_test, x_cifar100 = x_train_.astype('float')/255.0, x_test.astype('float')/255.0, x_cifar100.astype('float')/255.0
x_noise = np.random.random_integers(0,high=1,size=(1000,32,32,3)).astype('float')

x_svhn = loadmat('/Users/jayantadey/DF-CNN/data_five/SVHN/test_32x32.mat')['X']
y_svhn = loadmat('/Users/jayantadey/DF-CNN/data_five/SVHN/test_32x32.mat')['y']
#x_svhn = loadmat('/cis/home/jdey4/train_32x32.mat')['X']
#y_svhn = loadmat('/cis/home/jdey4/train_32x32.mat')['y']
#test_ids =  random.sample(range(0, x_svhn.shape[3]), 2000)
x_svhn = x_svhn.astype('float32')/255.0
x_tmp = np.zeros((len(x_svhn),32,32,3), dtype=float)

for ii in range(len(x_svhn)):
    x_tmp[ii,:,:,:] = x_svhn[:,:,:,ii]

x_svhn = x_tmp
del x_tmp

# Input image dimensions.
input_shape = x_train_.shape[1:]

for seed in seeds:
    print('Doing seed ',seed)
    
    x_train, x_cal, y_train, y_cal = train_test_split(x_train_, y_train_, random_state=seed, shuffle=True)
    
    model = keras.models.load_model('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/vit_model_'+str(seed)+'.keras')
    uncalibrated_model = KerasClassifier(model=model)
    uncalibrated_model.initialize(x_train, keras.utils.to_categorical(y_train))
    #uncalibrated_model.partial_fit(x_train, keras.utils.to_categorical(y_train))

    print('Training sigmoid')
    calibrated_nn_sigmoid = calcv(
                    uncalibrated_model, method='sigmoid', ensemble=False, cv='prefit')
    calibrated_nn_sigmoid.fit(x_cal, y_cal)

    print('Training isotonic')
    calibrated_nn_isotonic = calcv(
                    uncalibrated_model, method='isotonic', ensemble=False, cv='prefit')
    calibrated_nn_isotonic.fit(x_cal, y_cal)


    proba_in_sig = calibrated_nn_sigmoid.predict_proba(x_test)
    proba_cifar100_sig = calibrated_nn_sigmoid.predict_proba(x_cifar100)
    proba_svhn_sig = calibrated_nn_sigmoid.predict_proba(x_svhn)
    proba_noise_sig = calibrated_nn_sigmoid.predict_proba(x_noise)

    proba_in_iso = calibrated_nn_isotonic.predict_proba(x_test)
    proba_cifar100_iso = calibrated_nn_isotonic.predict_proba(x_cifar100)
    proba_svhn_iso = calibrated_nn_isotonic.predict_proba(x_svhn)
    proba_noise_iso = calibrated_nn_isotonic.predict_proba(x_noise)

    summary = (proba_in_sig, proba_cifar100_sig, proba_svhn_sig, proba_noise_sig,\
            proba_in_iso, proba_cifar100_iso, proba_svhn_iso, proba_noise_iso)

    print(summary)
    file_to_save = '/Users/jayantadey/kdg/benchmarks/cifar10_experiments/results/vit_baseline_'+str(seed)+'.pickle'

    with open(file_to_save, 'wb') as f:
        pickle.dump(summary, f)
# %%
'''p_in = proba_in_dn
p_out = proba_svhn_dn
from sklearn.metrics import roc_auc_score
true_labels = np.hstack((np.ones(len(p_in), ), np.zeros(len(p_out), )))

kdn_in_conf = np.max(p_in, axis=1)
kdn_out_conf = np.max(p_out, axis=1)

kdn_conf = np.hstack((kdn_in_conf, kdn_out_conf))

print(roc_auc_score(true_labels, kdn_conf))
print(fpr_at_95_tpr(kdn_in_conf, kdn_out_conf))
# %%
np.mean(np.argmax(p_in,axis=1)==y_test.ravel())
# %%
get_ece(p_in, y_test.ravel(), n_bins=15)
# %%
np.mean(np.abs(np.max(p_out,axis=1)-.1))'''
# %%
