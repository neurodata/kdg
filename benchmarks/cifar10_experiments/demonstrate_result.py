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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.metrics import roc_auc_score
#%%
def fpr_at_95_tpr(conf_in, conf_out):
    TPR = 95
    PERC = np.percentile(conf_in, 100-TPR)
    #FP = np.sum(conf_out >=  PERC)
    FPR = np.sum(conf_out >=  PERC)/len(conf_out)
    return FPR, PERC
#%%
# Load the CIFAR10 and CIFAR100 data.
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
(_, _), (x_cifar100, y_cifar100) = cifar100.load_data()
x_noise = np.random.random_integers(0,high=255,size=(1000,32,32,3)).astype('float')/255.0
#x_svhn = loadmat('/cis/home/jdey4/train_32x32.mat')['X']
#y_svhn = loadmat('/cis/home/jdey4/train_32x32.mat')['y']
#test_ids =  random.sample(range(0, x_svhn.shape[3]), 2000)
#x_svhn = x_svhn[:,:,:,test_ids].astype('float32')
#x_tmp = np.zeros((2000,32,32,3), dtype=float)

#for ii in range(2000):
#    x_tmp[ii,:,:,:] = x_svhn[:,:,:,ii]

#x_svhn = x_tmp
#del x_tmp
# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_cifar100 = x_cifar100.astype('float32') / 255
#x_svhn = x_svhn.astype('float32') / 255


for channel in range(3):
    x_train_mean = np.mean(x_train[:,:,:,channel])
    x_train_std = np.std(x_train[:,:,:,channel])

    x_train[:,:,:,channel] -= x_train_mean
    x_train[:,:,:,channel] /= x_train_std

    x_test[:,:,:,channel] -= x_train_mean
    x_test[:,:,:,channel] /= x_train_std

    x_cifar100[:,:,:,channel] -= x_train_mean
    x_cifar100[:,:,:,channel] /= x_train_std

    #x_svhn[:,:,:,channel] -= x_train_mean #+ 1
    #x_svhn[:,:,:,channel] /= x_train_std

    x_noise[:,:,:,channel] -= x_train_mean
    x_noise[:,:,:,channel] /= x_train_std

# %%
seeds = [100,200,300,400]
accuracy_kdn = []
accuracy_dn = []
accuracy_acet = []
accuracy_odin = []
accuracy_oe = []
mce_kdn = []
mce_dn = []
mce_acet = []
mce_odin = []
mce_oe = []

accuracy_iso = []
accuracy_sig = []
mce_iso = []
mce_sig = []

auroc_kdn_cifar100 = []
auroc_dn_cifar100 = []
auroc_acet_cifar100 = []
auroc_iso_cifar100 = []
auroc_sig_cifar100 = []
auroc_odin_cifar100 = []
auroc_oe_cifar100 = []
fpr_kdn_cifar100 = []
fpr_dn_cifar100 = []
fpr_acet_cifar100 = []
fpr_iso_cifar100 = []
fpr_sig_cifar100 = []
fpr_odin_cifar100 = []
fpr_oe_cifar100 = []
oce_kdn_cifar100 = []
oce_dn_cifar100 = []
oce_acet_cifar100 = []
oce_iso_cifar100 = []
oce_sig_cifar100 = []
oce_odin_cifar100 = []
oce_oe_cifar100 = []

auroc_kdn_svhn = []
auroc_dn_svhn = []
auroc_acet_svhn = []
auroc_iso_svhn = []
auroc_sig_svhn = []
auroc_odin_svhn = []
auroc_oe_svhn = []
fpr_kdn_svhn = []
fpr_dn_svhn = []
fpr_acet_svhn = []
fpr_iso_svhn = []
fpr_sig_svhn = []
fpr_odin_svhn = []
fpr_oe_svhn = []
oce_kdn_svhn = []
oce_dn_svhn = []
oce_acet_svhn = []
oce_iso_svhn = []
oce_sig_svhn = []
oce_odin_svhn = []
oce_oe_svhn = []

auroc_kdn_noise = []
auroc_dn_noise = []
auroc_acet_noise = []
auroc_iso_noise = []
auroc_sig_noise = []
auroc_odin_noise = []
auroc_oe_noise = []
fpr_kdn_noise = []
fpr_dn_noise = []
fpr_acet_noise = []
fpr_iso_noise = []
fpr_sig_noise = []
fpr_odin_noise = []
fpr_oe_noise = []
oce_kdn_noise = []
oce_dn_noise = []
oce_acet_noise = []
oce_iso_noise = []
oce_sig_noise = []
oce_odin_noise = []
oce_oe_noise = []

for seed in seeds:
    #with open('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/results/resnet20_new_'+str(seed)+'.pickle','rb') as f:
    with open('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/resnet20_cifar100_'+str(seed)+'.pickle','rb') as f:
        (proba_in, proba_cifar100, proba_svhn, proba_noise, proba_in_dn, proba_cifar100_dn, proba_svhn_dn, proba_noise_dn, proba_in_acet, proba_cifar100_acet, proba_svhn_acet, proba_noise_acet) = pickle.load(f)

    with open('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/results/resnet20_baseline_new_'+str(seed)+'.pickle', 'rb') as f:
        (proba_in_sig, proba_cifar100_sig, proba_svhn_sig, proba_noise_sig,\
            proba_in_iso, proba_cifar100_iso, proba_svhn_iso, proba_noise_iso) = pickle.load(f)

    with open('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/resnet20_cifar100_ODIN_'+str(seed)+'.pickle', 'rb') as f:
        (proba_in_odin, proba_cifar100_odin, proba_svhn_odin, proba_noise_odin) = pickle.load(f)

    with open('/Users/jayantadey/kdg/benchmarks/cifar10_experiments/resnet20_cifar10_OE_'+str(seed)+'.pickle', 'rb') as f:
        (proba_in_oe, proba_cifar100_oe, proba_svhn_oe, proba_noise_oe) = pickle.load(f)
              
    accuracy_kdn.append(
        np.mean(np.argmax(proba_in,axis=1)==y_test.ravel())
    )
    accuracy_dn.append(
        np.mean(np.argmax(proba_in_dn,axis=1)==y_test.ravel())
    )
    accuracy_acet.append(
        np.mean(np.argmax(proba_in_acet,axis=1)==y_test.ravel())
    )
    accuracy_iso.append(
        np.mean(np.argmax(proba_in_iso,axis=1)==y_test.ravel())
    )
    accuracy_sig.append(
        np.mean(np.argmax(proba_in_sig,axis=1)==y_test.ravel())
    )
    accuracy_odin.append(
        np.mean(np.argmax(proba_in_odin,axis=1)==y_test.ravel())
    )
    accuracy_oe.append(
        np.mean(np.argmax(proba_in_oe,axis=1)==y_test.ravel())
    )

    mce_dn.append(
        get_ece(proba_in_dn, y_test.ravel())
    )
    mce_kdn.append(
        get_ece(proba_in, y_test.ravel())
    )
    mce_acet.append(
        get_ece(proba_in_acet, y_test.ravel())
    )
    mce_iso.append(
        get_ece(proba_in_iso, y_test.ravel())
    )
    mce_sig.append(
        get_ece(proba_in_sig, y_test.ravel())
    )
    mce_odin.append(
        get_ece(proba_in_odin, y_test.ravel())
    )
    mce_oe.append(
        get_ece(proba_in_oe, y_test.ravel())
    )

    kdn_in_conf = np.max(proba_in, axis=1)
    kdn_out_conf = np.max(proba_cifar100, axis=1)
    kdn_conf_cifar100 = np.hstack((kdn_in_conf, kdn_out_conf))
    dn_in_conf = np.max(proba_in_dn, axis=1)
    dn_out_conf = np.max(proba_cifar100_dn, axis=1)
    dn_conf_cifar100 = np.hstack((dn_in_conf, dn_out_conf))
    acet_in_conf = np.max(proba_in_acet, axis=1)
    acet_out_conf = np.max(proba_cifar100_acet, axis=1)
    acet_conf_cifar100 = np.hstack((acet_in_conf, acet_out_conf))
    iso_in_conf = np.max(proba_in_iso, axis=1)
    iso_out_conf = np.max(proba_cifar100_iso, axis=1)
    iso_conf_cifar100 = np.hstack((iso_in_conf, iso_out_conf))
    sig_in_conf = np.max(proba_in_sig, axis=1)
    sig_out_conf = np.max(proba_cifar100_sig, axis=1)
    sig_conf_cifar100 = np.hstack((sig_in_conf, sig_out_conf))
    odin_in_conf = np.max(proba_in_odin, axis=1)
    odin_out_conf = np.max(proba_cifar100_odin, axis=1)
    odin_conf_cifar100 = np.hstack((sig_in_conf, sig_out_conf))
    oe_in_conf = np.max(proba_in_oe, axis=1)
    oe_out_conf = np.max(proba_cifar100_oe, axis=1)
    oe_conf_cifar100 = np.hstack((sig_in_conf, sig_out_conf))


    true_labels = np.hstack((np.ones(len(proba_in), ), np.zeros(len(proba_cifar100), )))

    auroc_kdn_cifar100.append(
        roc_auc_score(true_labels, kdn_conf_cifar100)
    )
    auroc_dn_cifar100.append(
        roc_auc_score(true_labels, dn_conf_cifar100)
    )
    auroc_acet_cifar100.append(
        roc_auc_score(true_labels, acet_conf_cifar100)
    )
    auroc_iso_cifar100.append(
        roc_auc_score(true_labels, iso_conf_cifar100)
    )
    auroc_sig_cifar100.append(
        roc_auc_score(true_labels, sig_conf_cifar100)
    )
    auroc_odin_cifar100.append(
        roc_auc_score(true_labels, odin_conf_cifar100)
    )
    auroc_oe_cifar100.append(
        roc_auc_score(true_labels, oe_conf_cifar100)
    )
    fpr_kdn_cifar100.append(
        fpr_at_95_tpr(kdn_in_conf, kdn_out_conf)
    )
    fpr_dn_cifar100.append(
        fpr_at_95_tpr(dn_in_conf, dn_out_conf)
    )
    fpr_acet_cifar100.append(
        fpr_at_95_tpr(acet_in_conf, acet_out_conf)
    )
    fpr_iso_cifar100.append(
        fpr_at_95_tpr(iso_in_conf, iso_out_conf)
    )
    fpr_sig_cifar100.append(
        fpr_at_95_tpr(sig_in_conf, sig_out_conf)
    )
    fpr_odin_cifar100.append(
        fpr_at_95_tpr(odin_in_conf, odin_out_conf)
    )
    fpr_oe_cifar100.append(
        fpr_at_95_tpr(oe_in_conf, oe_out_conf)
    )
    oce_kdn_cifar100.append(
        np.mean(np.abs(kdn_out_conf - 0.1))
    )
    oce_dn_cifar100.append(
        np.mean(np.abs(dn_out_conf - 0.1))
    )
    oce_acet_cifar100.append(
        np.mean(np.abs(acet_out_conf - 0.1))
    ) 
    oce_iso_cifar100.append(
        np.mean(np.abs(iso_out_conf - 0.1))
    ) 
    oce_sig_cifar100.append(
        np.mean(np.abs(sig_out_conf - 0.1))
    )  
    oce_odin_cifar100.append(
        np.mean(np.abs(odin_out_conf - 0.1))
    )       
    oce_oe_cifar100.append(
        np.mean(np.abs(oe_out_conf - 0.1))
    )

    kdn_in_conf = np.max(proba_in, axis=1)
    kdn_out_conf = np.max(proba_svhn, axis=1)
    kdn_conf_svhn= np.hstack((kdn_in_conf, kdn_out_conf))
    dn_in_conf = np.max(proba_in_dn, axis=1)
    dn_out_conf = np.max(proba_svhn_dn, axis=1)
    dn_conf_svhn = np.hstack((dn_in_conf, dn_out_conf))
    acet_in_conf = np.max(proba_in_acet, axis=1)
    acet_out_conf = np.max(proba_svhn_acet, axis=1)
    acet_conf_svhn = np.hstack((acet_in_conf, acet_out_conf))
    iso_in_conf = np.max(proba_in_iso, axis=1)
    iso_out_conf = np.max(proba_svhn_iso, axis=1)
    iso_conf_svhn = np.hstack((iso_in_conf, iso_out_conf))
    sig_in_conf = np.max(proba_in_sig, axis=1)
    sig_out_conf = np.max(proba_svhn_sig, axis=1)
    sig_conf_svhn = np.hstack((sig_in_conf, sig_out_conf))
    odin_in_conf = np.max(proba_in_odin, axis=1)
    odin_out_conf = np.max(proba_svhn_odin, axis=1)
    odin_conf_svhn = np.hstack((odin_in_conf, odin_out_conf))
    oe_in_conf = np.max(proba_in_oe, axis=1)
    oe_out_conf = np.max(proba_svhn_oe, axis=1)
    oe_conf_svhn = np.hstack((oe_in_conf, oe_out_conf))
    
    true_labels = np.hstack((np.ones(len(proba_in), ), np.zeros(len(proba_svhn), )))

    auroc_kdn_svhn.append(
        roc_auc_score(true_labels, kdn_conf_svhn)
    )
    auroc_dn_svhn.append(
        roc_auc_score(true_labels, dn_conf_svhn)
    )
    auroc_acet_svhn.append(
        roc_auc_score(true_labels, acet_conf_svhn)
    )
    auroc_iso_svhn.append(
        roc_auc_score(true_labels, iso_conf_svhn)
    )
    auroc_sig_svhn.append(
        roc_auc_score(true_labels, sig_conf_svhn)
    )
    auroc_odin_svhn.append(
        roc_auc_score(true_labels, odin_conf_svhn)
    )
    auroc_oe_svhn.append(
        roc_auc_score(true_labels, oe_conf_svhn)
    )
    fpr_kdn_svhn.append(
        fpr_at_95_tpr(kdn_in_conf, kdn_out_conf)
    )
    fpr_dn_svhn.append(
        fpr_at_95_tpr(dn_in_conf, dn_out_conf)
    )
    fpr_acet_svhn.append(
        fpr_at_95_tpr(acet_in_conf, acet_out_conf)
    )
    fpr_iso_svhn.append(
        fpr_at_95_tpr(iso_in_conf, iso_out_conf)
    )
    fpr_sig_svhn.append(
        fpr_at_95_tpr(sig_in_conf, sig_out_conf)
    )
    fpr_odin_svhn.append(
        fpr_at_95_tpr(odin_in_conf, odin_out_conf)
    )
    fpr_oe_svhn.append(
        fpr_at_95_tpr(oe_in_conf, oe_out_conf)
    )
    oce_kdn_svhn.append(
        np.mean(np.abs(kdn_out_conf - 0.1))
    )
    oce_dn_svhn.append(
        np.mean(np.abs(dn_out_conf - 0.1))
    )
    oce_acet_svhn.append(
        np.mean(np.abs(acet_out_conf - 0.1))
    )  
    oce_iso_svhn.append(
        np.mean(np.abs(iso_out_conf - 0.1))
    ) 
    oce_sig_svhn.append(
        np.mean(np.abs(sig_out_conf - 0.1))
    ) 
    oce_odin_svhn.append(
        np.mean(np.abs(odin_out_conf - 0.1))
    )
    oce_oe_svhn.append(
        np.mean(np.abs(oe_out_conf - 0.1))
    )

    kdn_in_conf = np.max(proba_in, axis=1)
    kdn_out_conf = np.max(proba_noise, axis=1)
    kdn_conf_noise = np.hstack((kdn_in_conf, kdn_out_conf))
    dn_in_conf = np.max(proba_in_dn, axis=1)
    dn_out_conf = np.max(proba_noise_dn, axis=1)
    dn_conf_noise = np.hstack((dn_in_conf, dn_out_conf))
    acet_in_conf = np.max(proba_in_acet, axis=1)
    acet_out_conf = np.max(proba_noise_acet, axis=1)
    acet_conf_noise = np.hstack((acet_in_conf, acet_out_conf))
    iso_in_conf = np.max(proba_in_iso, axis=1)
    iso_out_conf = np.max(proba_noise_iso, axis=1)
    iso_conf_noise = np.hstack((iso_in_conf, iso_out_conf))
    sig_in_conf = np.max(proba_in_sig, axis=1)
    sig_out_conf = np.max(proba_noise_sig, axis=1)
    sig_conf_noise = np.hstack((sig_in_conf, sig_out_conf))
    odin_in_conf = np.max(proba_in_odin, axis=1)
    odin_out_conf = np.max(proba_noise_odin, axis=1)
    odin_conf_noise = np.hstack((odin_in_conf, odin_out_conf))
    oe_in_conf = np.max(proba_in_oe, axis=1)
    oe_out_conf = np.max(proba_noise_oe, axis=1)
    oe_conf_noise = np.hstack((oe_in_conf, oe_out_conf))

    true_labels = np.hstack((np.ones(len(proba_in), ), np.zeros(len(proba_noise), )))

    auroc_kdn_noise.append(
        roc_auc_score(true_labels, kdn_conf_noise)
    )
    auroc_dn_noise.append(
        roc_auc_score(true_labels, dn_conf_noise)
    )
    auroc_acet_noise.append(
        roc_auc_score(true_labels, acet_conf_noise)
    )
    auroc_iso_noise.append(
        roc_auc_score(true_labels, iso_conf_noise)
    )
    auroc_sig_noise.append(
        roc_auc_score(true_labels, sig_conf_noise)
    )
    auroc_odin_noise.append(
        roc_auc_score(true_labels, odin_conf_noise)
    )
    auroc_oe_noise.append(
        roc_auc_score(true_labels, oe_conf_noise)
    )
    fpr_kdn_noise.append(
        fpr_at_95_tpr(kdn_in_conf, kdn_out_conf)
    )
    fpr_dn_noise.append(
        fpr_at_95_tpr(dn_in_conf, dn_out_conf)
    )
    fpr_acet_noise.append(
        fpr_at_95_tpr(acet_in_conf, acet_out_conf)
    )
    fpr_iso_noise.append(
        fpr_at_95_tpr(iso_in_conf, iso_out_conf)
    )
    fpr_sig_noise.append(
        fpr_at_95_tpr(sig_in_conf, sig_out_conf)
    )
    fpr_odin_noise.append(
        fpr_at_95_tpr(odin_in_conf, odin_out_conf)
    )
    fpr_oe_noise.append(
        fpr_at_95_tpr(oe_in_conf, oe_out_conf)
    )
    oce_kdn_noise.append(
        np.mean(np.abs(kdn_out_conf - 0.1))
    )
    oce_dn_noise.append(
        np.mean(np.abs(dn_out_conf - 0.1))
    )
    oce_acet_noise.append(
        np.mean(np.abs(acet_out_conf - 0.1))
    )
    oce_iso_noise.append(
        np.mean(np.abs(iso_out_conf - 0.1))
    )
    oce_sig_noise.append(
        np.mean(np.abs(sig_out_conf - 0.1))
    )
    oce_odin_noise.append(
        np.mean(np.abs(odin_out_conf - 0.1))
    )
    oce_oe_noise.append(
        np.mean(np.abs(oe_out_conf - 0.1))
    )

print('DN accuracy ', np.mean(accuracy_dn), '(+-',np.std(accuracy_dn),')')
print('KDN accuracy ', np.mean(accuracy_kdn), '(+-',np.std(accuracy_kdn),')')
print('ACET accuracy ', np.mean(accuracy_acet), '(+-',np.std(accuracy_acet),')')
print('Isotonic accuracy ', np.mean(accuracy_iso), '(+-',np.std(accuracy_iso),')')
print('Sigmoid accuracy ', np.mean(accuracy_sig), '(+-',np.std(accuracy_sig),')')
print('ODIN accuracy ', np.mean(accuracy_odin), '(+-',np.std(accuracy_odin),')')
print('OE accuracy ', np.mean(accuracy_oe), '(+-',np.std(accuracy_oe),')\n')

print('DN MCE ', np.mean(mce_dn), '(+-',np.std(mce_dn),')')
print('KDN MCE ', np.mean(mce_kdn), '(+-',np.std(mce_kdn),')')
print('ACET MCE ', np.mean(mce_acet), '(+-',np.std(mce_acet),')')
print('Isotonic MCE ', np.mean(mce_iso), '(+-',np.std(mce_iso),')')
print('Sigmoid MCE ', np.mean(mce_sig), '(+-',np.std(mce_sig),')')
print('ODIN MCE ', np.mean(mce_odin), '(+-',np.std(mce_odin),')')
print('OE MCE ', np.mean(mce_oe), '(+-',np.std(mce_oe),')\n')

print('DN AUROC cifar100', np.mean(auroc_dn_cifar100), '(+-',np.std(auroc_dn_cifar100),')')
print('KDN AUROC cifar100', np.mean(auroc_kdn_cifar100), '(+-',np.std(auroc_kdn_cifar100),')')
print('ACET AUROC cifar100', np.mean(auroc_acet_cifar100), '(+-',np.std(auroc_acet_cifar100),')')
print('Isotonic AUROC cifar100', np.mean(auroc_iso_cifar100), '(+-',np.std(auroc_iso_cifar100),')')
print('Sigmoid AUROC cifar100', np.mean(auroc_sig_cifar100), '(+-',np.std(auroc_sig_cifar100),')')
print('ODIN AUROC cifar100', np.mean(auroc_odin_cifar100), '(+-',np.std(auroc_odin_cifar100),')')
print('OE AUROC cifar100', np.mean(auroc_oe_cifar100), '(+-',np.std(auroc_oe_cifar100),')\n')

print('DN FPR@95 cifar100', np.mean(fpr_dn_cifar100), '(+-',np.std(fpr_dn_cifar100),')')
print('KDN FPR@95 cifar100', np.mean(fpr_kdn_cifar100), '(+-',np.std(fpr_kdn_cifar100),')')
print('ACET FPR@95 cifar100', np.mean(fpr_acet_cifar100), '(+-',np.std(fpr_acet_cifar100),')')
print('Isotonic FPR@95 cifar100', np.mean(fpr_iso_cifar100), '(+-',np.std(fpr_iso_cifar100),')')
print('Sigmoid FPR@95 cifar100', np.mean(fpr_sig_cifar100), '(+-',np.std(fpr_sig_cifar100),')')
print('ODIN FPR@95 cifar100', np.mean(fpr_odin_cifar100), '(+-',np.std(fpr_odin_cifar100),')')
print('OE FPR@95 cifar100', np.mean(fpr_oe_cifar100), '(+-',np.std(fpr_oe_cifar100),')\n')


print('DN OCE cifar100', np.mean(oce_dn_cifar100), '(+-',np.std(oce_dn_cifar100),')')
print('KDN OCE cifar100', np.mean(oce_kdn_cifar100), '(+-',np.std(oce_kdn_cifar100),')')
print('ACET OCE cifar100', np.mean(oce_acet_cifar100), '(+-',np.std(oce_acet_cifar100),')')
print('Isotonic OCE cifar100', np.mean(oce_iso_cifar100), '(+-',np.std(oce_iso_cifar100),')')
print('Sigmoid OCE cifar100', np.mean(oce_sig_cifar100), '(+-',np.std(oce_sig_cifar100),')')
print('ODIN OCE cifar100', np.mean(oce_odin_cifar100), '(+-',np.std(oce_odin_cifar100),')')
print('OE OCE cifar100', np.mean(oce_oe_cifar100), '(+-',np.std(oce_oe_cifar100),')\n')


print('DN AUROC svhn', np.mean(auroc_dn_svhn), '(+-',np.std(auroc_dn_svhn),')')
print('KDN AUROC svhn', np.mean(auroc_kdn_svhn), '(+-',np.std(auroc_kdn_svhn),')')
print('ACET AUROC svhn', np.mean(auroc_acet_svhn), '(+-',np.std(auroc_acet_svhn),')')
print('Isotonic AUROC svhn', np.mean(auroc_iso_svhn), '(+-',np.std(auroc_iso_svhn),')')
print('Sigmoid AUROC svhn', np.mean(auroc_sig_svhn), '(+-',np.std(auroc_sig_svhn),')')
print('ODIN AUROC svhn', np.mean(auroc_odin_svhn), '(+-',np.std(auroc_odin_svhn),')')
print('OE AUROC svhn', np.mean(auroc_oe_svhn), '(+-',np.std(auroc_oe_svhn),')\n')

print('DN FPR@95 svhn', np.mean(fpr_dn_svhn), '(+-',np.std(fpr_dn_svhn),')')
print('KDN FPR@95 svhn', np.mean(fpr_kdn_svhn), '(+-',np.std(fpr_kdn_svhn),')')
print('ACET FPR@95 svhn', np.mean(fpr_acet_svhn), '(+-',np.std(fpr_acet_svhn),')')
print('Isotonic FPR@95 svhn', np.mean(fpr_iso_svhn), '(+-',np.std(fpr_iso_svhn),')')
print('Sigmoid FPR@95 svhn', np.mean(fpr_sig_svhn), '(+-',np.std(fpr_sig_svhn),')')
print('ODIN FPR@95 svhn', np.mean(fpr_odin_svhn), '(+-',np.std(fpr_odin_svhn),')')
print('OE FPR@95 svhn', np.mean(fpr_oe_svhn), '(+-',np.std(fpr_oe_svhn),')\n')

print('DN OCE svhn', np.mean(oce_dn_svhn), '(+-',np.std(oce_dn_svhn),')')
print('KDN OCE svhn', np.mean(oce_kdn_svhn), '(+-',np.std(oce_kdn_svhn),')')
print('ACET OCE svhn', np.mean(oce_acet_svhn), '(+-',np.std(oce_acet_svhn),')')
print('Isotonic OCE svhn', np.mean(oce_iso_svhn), '(+-',np.std(oce_iso_svhn),')')
print('Sigmoid OCE svhn', np.mean(oce_sig_svhn), '(+-',np.std(oce_sig_svhn),')')
print('ODIN OCE svhn', np.mean(oce_odin_svhn), '(+-',np.std(oce_odin_svhn),')')
print('OE OCE svhn', np.mean(oce_oe_svhn), '(+-',np.std(oce_oe_svhn),')\n')

print('DN AUROC noise', np.mean(auroc_dn_noise), '(+-',np.std(auroc_dn_noise),')')
print('KDN AUROC noise', np.mean(auroc_kdn_noise), '(+-',np.std(auroc_kdn_noise),')')
print('ACET AUROC noise', np.mean(auroc_acet_noise), '(+-',np.std(auroc_acet_noise),')')
print('Isotonic AUROC noise', np.mean(auroc_iso_noise), '(+-',np.std(auroc_iso_noise),')')
print('Sigmoid AUROC noise', np.mean(auroc_sig_noise), '(+-',np.std(auroc_sig_noise),')')
print('ODIN AUROC noise', np.mean(auroc_odin_noise), '(+-',np.std(auroc_odin_noise),')')
print('OE AUROC noise', np.mean(auroc_oe_noise), '(+-',np.std(auroc_oe_noise),')\n')

print('DN FPR@95 noise', np.mean(fpr_dn_noise), '(+-',np.std(fpr_dn_noise),')')
print('KDN FPR@95 noise', np.mean(fpr_kdn_noise), '(+-',np.std(fpr_kdn_noise),')')
print('ACET FPR@95 noise', np.mean(fpr_acet_noise), '(+-',np.std(fpr_acet_noise),')')
print('Isotonic FPR@95 noise', np.mean(fpr_iso_noise), '(+-',np.std(fpr_iso_noise),')')
print('Sigmoid FPR@95 noise', np.mean(fpr_sig_noise), '(+-',np.std(fpr_sig_noise),')')
print('ODIN FPR@95 noise', np.mean(fpr_odin_noise), '(+-',np.std(fpr_odin_noise),')')
print('OE FPR@95 noise', np.mean(fpr_oe_noise), '(+-',np.std(fpr_oe_noise),')\n')

print('DN OCE noise', np.mean(oce_dn_noise), '(+-',np.std(oce_dn_noise),')')
print('KDN OCE noise', np.mean(oce_kdn_noise), '(+-',np.std(oce_kdn_noise),')')
print('ACET OCE noise', np.mean(oce_acet_noise), '(+-',np.std(oce_acet_noise),')')
print('Isotonic OCE noise', np.mean(oce_iso_noise), '(+-',np.std(oce_iso_noise),')')
print('Sigmoid OCE noise', np.mean(oce_sig_noise), '(+-',np.std(oce_sig_noise),')')
print('ODIN OCE noise', np.mean(oce_odin_noise), '(+-',np.std(oce_odin_noise),')')
print('OE OCE noise', np.mean(oce_oe_noise), '(+-',np.std(oce_oe_noise),')\n')
# %%
