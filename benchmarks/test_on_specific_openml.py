#%%
from kdg import kdf
from kdg.utils import get_ece
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import numpy as np
import openml
from sklearn.metrics import cohen_kappa_score
from kdg.utils import get_ece
#%%
dataset_id = 32
dataset = openml.datasets.get_dataset(dataset_id)
X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )

#features_to_remove = [0,1,6,7,8,9]
#features = [True]*X.shape[1]
#for ii in features_to_remove:
#    features[ii] = False

#X = X[:,features]

df = pd.DataFrame()

for ii in range(X.shape[1]):
    df['feature '+str(ii)] = X[:,ii]

df['class'] = y
#sns.pairplot(df,hue='class')
# %%
def experiment(X, y, folder, n_estimators=500, reps=30, feature=0):
    X = X[:,feature]
    X = X.reshape(-1,1)
    
    if np.mean(is_categorical) >0:
        return

    if np.isnan(np.sum(y)):
        return

    if np.isnan(np.sum(X)):
        return

    total_sample = X.shape[0]
    unique_classes, counts = np.unique(y, return_counts=True)

    test_sample = min(counts)//3

    indx = []
    for label in unique_classes:
        indx.append(
            np.where(
                y==label
            )[0]
        )

    max_sample = min(counts) - test_sample
    train_samples = np.logspace(
        np.log10(2),
        np.log10(max_sample),
        num=10,
        endpoint=True,
        dtype=int
        )
    
    err = []
    err_train = []
    err_rf = []
    ece = []
    ece_rf = []
    kappa = []
    kappa_rf = []
    mc_rep = []
    samples = []
    total_polytopes = []

    for train_sample in train_samples:
        
        for rep in range(reps):
            indx_to_take_train = []
            indx_to_take_test = []

            for ii, _ in enumerate(unique_classes):
                np.random.shuffle(indx[ii])
                indx_to_take_train.extend(
                    list(
                            indx[ii][:train_sample]
                    )
                )
                indx_to_take_test.extend(
                    list(
                            indx[ii][-test_sample:counts[ii]]
                    )
                )
            model_kdf = kdf(kwargs={'n_estimators':n_estimators})
            model_kdf.fit(X[indx_to_take_train], y[indx_to_take_train])
            proba_kdf = model_kdf.predict_proba(X[indx_to_take_test])
            proba_rf = model_kdf.rf_model.predict_proba(X[indx_to_take_test])
            predicted_label_kdf = np.argmax(proba_kdf, axis = 1)
            predicted_label_rf = np.argmax(proba_rf, axis = 1)

            polytope_count = 0
            for lbl in unique_classes:
                polytope_count += len(model_kdf.polytope_means[lbl])

            total_polytopes.append(
                polytope_count
            )
            err_train.append(
                1 - np.mean(
                        model_kdf.predict(X[indx_to_take_train])==y[indx_to_take_train]
                    )
            )
            err.append(
                1 - np.mean(
                        predicted_label_kdf==y[indx_to_take_test]
                    )
            )
            err_rf.append(
                1 - np.mean(
                    predicted_label_rf==y[indx_to_take_test]
                )
            )
            kappa.append(
                cohen_kappa_score(predicted_label_kdf, y[indx_to_take_test])
            )
            kappa_rf.append(
                cohen_kappa_score(predicted_label_rf, y[indx_to_take_test])
            )
            ece.append(
                get_ece(proba_kdf, predicted_label_kdf, y[indx_to_take_test])
            )
            ece_rf.append(
                get_ece(proba_rf, predicted_label_rf, y[indx_to_take_test])
            )
            samples.append(
                train_sample*len(unique_classes)
            )
            mc_rep.append(rep)

    df = pd.DataFrame() 
    df['err_kdf'] = err
    df['err_kdf_train'] = err_train
    df['err_rf'] = err_rf
    df['kappa_kdf'] = kappa
    df['kappa_rf'] = kappa_rf
    df['ece_kdf'] = ece
    df['ece_rf'] = ece_rf
    df['rep'] = mc_rep
    df['samples'] = samples
    df['total_polytopes'] = total_polytopes

    df.to_csv(folder+'/'+'openML_cc18_'+str(dataset_id)+'_'+str(feature)+'.csv')

#%%
folder = 'openml_res'

for ii in range(X.shape[1]):
    experiment(X, y, folder, feature=ii)
# %%
dataset_id = 182
dataset = openml.datasets.get_dataset(dataset_id)
X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )

#X = np.concatenate((X[:,:4],X[:,5:15]), axis=1)
total_sample = X.shape[0]
test_sample = total_sample//3
max_sample = total_sample - test_sample
train_samples = np.logspace(
        np.log10(2),
        np.log10(max_sample),
        num=10,
        endpoint=True,
        dtype=int
        )
indices = list(range(total_sample))
np.random.shuffle(indices)
indx_to_take_train = indices[:train_samples[-1]]
indx_to_take_test = indices[-test_sample:]
model_kdf = kdf(k=1, kwargs={'n_estimators':500})
model_kdf.fit(X[indx_to_take_train], y[indx_to_take_train])
proba_kdf = model_kdf.predict_proba(X[indx_to_take_test])
proba_rf = model_kdf.rf_model.predict_proba(X[indx_to_take_test])
predicted_label_kdf = np.argmax(proba_kdf, axis = 1)
predicted_label_rf = np.argmax(proba_rf, axis = 1)

print(1 - np.mean( predicted_label_kdf==y[indx_to_take_test]))
print(1 - np.mean( predicted_label_rf==y[indx_to_take_test]))


# %%
total_dim = X.shape[1]
idx = indx_to_take_train[21]
for ii in range(total_dim):
    print('doing ',ii)
    print(model_kdf._compute_log_likelihood_1d(X[idx,ii], model_kdf.polytope_means[1445][ii], model_kdf.polytope_cov[1445][ii]))
    print(model_kdf._compute_log_likelihood_1d(X[idx,ii], model_kdf.polytope_means[1460][ii], model_kdf.polytope_cov[1460][ii]))

# %%
for dim in range(X.shape[1]):
    cov = []

    for polytope, _ in enumerate(model_kdf.polytope_means):
        cov.append(
            model_kdf.polytope_cov[polytope][dim]
        )
    threshold = np.percentile(cov,50)
    print(threshold)
    for polytope, _ in enumerate(model_kdf.polytope_means):
        if model_kdf.polytope_cov[polytope][dim] < threshold:
            model_kdf.polytope_cov[polytope][dim] = threshold

# %%
idx = 65

for label in range(6):
    mx = -np.inf
    for polytope, _ in enumerate(model_kdf.polytope_means):
        a = model_kdf._compute_log_likelihood(X[indx_to_take_train[idx]:indx_to_take_train[idx]+1], label, polytope)
        if mx < a:
            mx = a
            pl = polytope

        if (np.sum(X[indx_to_take_train[idx]]-model_kdf.polytope_means[polytope])==0):
            indx = polytope
            lk_ = a

    print('label ', label, 'max likelihood ', mx, 'polytope ', pl, 'cardinality ', model_kdf.polytope_cardinality[label][pl], 'perfect match', indx, 'matched likelihood', lk_)


# %%
proba_kdf = model_kdf.predict_proba(X[indx_to_take_train])
proba_rf = model_kdf.rf_model.predict_proba(X[indx_to_take_train])
predicted_label_kdf = np.argmax(proba_kdf, axis = 1)
predicted_label_rf = np.argmax(proba_rf, axis = 1)

print(1 - np.mean( predicted_label_kdf==y[indx_to_take_train]))
print(1 - np.mean( predicted_label_rf==y[indx_to_take_train]))

match = predicted_label_kdf==predicted_label_rf
print(np.where(match==False)[0])
# %%
