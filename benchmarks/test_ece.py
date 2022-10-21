#%%
from statistics import mode, quantiles
import numpy as np
from torch import quantile
from kdg import kdf
import openml
from kdg.utils import get_ece, generate_gaussian_parity, generate_spirals, generate_ellipse
from sklearn.ensemble import RandomForestClassifier as rf
# %%
def generate_parity(n, d=2, invert_labels=False,acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
        
    X = np.random.uniform(-1, 1, size=(n, d))
    Y = (np.sum(X > 0, axis=1) % 2 == 0).astype(int)
    
    if invert_labels:
        Y = -1 * (Y - 1)
    
    return X, Y.astype(int)

#%%
def get_ece_(predicted_posterior, predicted_label, true_label, R=20):
    total_sample = len(true_label)
    K = predicted_posterior.shape[1]

    score = 0
    bin_size = total_sample//R
    for k in range(K):
        posteriors = predicted_posterior[:,k]
        sorted_indx = np.argsort(posteriors)
        #print(sorted_indx, len(sorted_indx))
        for r in range(R):        
            indx = sorted_indx[r*bin_size:(r+1)*bin_size]
            #print(indx)
            predicted_label_ = predicted_label[indx]
            true_label_ = true_label[indx]

            indx_k = np.where(true_label_ == k)[0]
            acc = (
                np.nan_to_num(np.mean(predicted_label_[indx_k] == k))
                if indx_k.size != 0
                else 0
            )
            conf = np.nan_to_num(np.mean(posteriors[indx_k])) if indx_k.size != 0 else 0

            #print(acc, conf)
            score += len(indx) * np.abs(acc - conf)

    score /= (K*total_sample)
    return score

#%%
train_samples = [10, 100, 1000, 10000]
test_sample = 1000
reps = 10
performance_indx = []
performance_indx_rf = []
kdf_25 = []
kdf_75 = []
rf_25 = []
rf_75 = []

for samples in train_samples:
    ECE_kdf = []
    ECE_rf = []

    for _ in range(reps):
        X, y = generate_ellipse(samples)
        X_test, y_test = generate_ellipse(test_sample)
        model_kdf = kdf(kwargs={'n_estimators':500})
        model_kdf.fit(X, y)
        proba_kdf = model_kdf.predict_proba(X_test)
        proba_rf = model_kdf.rf_model.predict_proba((X_test-model_kdf.min_val)/(model_kdf.max_val-model_kdf.min_val))

        ece_kdf = get_ece_(proba_kdf, np.argmax(proba_kdf,axis=1), y_test)
        ece_rf = get_ece_(proba_rf, np.argmax(proba_rf,axis=1), y_test)
        ECE_kdf.append(ece_kdf)
        ECE_rf.append(ece_rf)

    quantiles = np.quantile(ECE_kdf,[.25,.75],axis=0)
    kdf_25.append(quantiles[0])
    kdf_75.append(quantiles[1])

    quantiles = np.quantile(ECE_rf,[.25,.75],axis=0)
    rf_25.append(quantiles[0])
    rf_75.append(quantiles[1])

    performance_indx.append(
        np.mean(ECE_kdf)
    )
    performance_indx_rf.append(
        np.mean(ECE_rf)
    )

#%%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(train_samples, performance_indx, c='r', linewidth=2, label='kdf')
ax.fill_between(train_samples, kdf_25, kdf_75, facecolor='r', alpha=.3)

ax.plot(train_samples, performance_indx_rf, c='b', linewidth=2, label='rf')
ax.fill_between(train_samples, rf_25, rf_75, facecolor='b', alpha=.3)

ax.set_xlabel('Sample #')
ax.set_ylabel('ECE')
plt.legend()
plt.savefig('plots/test_ece_ellipse.pdf')
# %%
X, y = generate_gaussian_parity(1000)
X_test, y_test = generate_gaussian_parity(1000)

model_kdf = kdf(k=1e1, kwargs={'n_estimators':500})
model_kdf.fit(X, y)
print(np.mean(model_kdf.predict(X_test)==y_test))
print(np.mean(model_kdf.predict(X)==y))
# %%
dataset = openml.datasets.get_dataset(40499)
X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )

model_kdf = kdf(kwargs={'n_estimators':500})
total_sample = X.shape[0]
test_sample = total_sample//3
train_sample = total_sample - test_sample
indices = list(range(total_sample))
indx_to_take_train = indices[:train_sample]
indx_to_take_test = indices[-test_sample:]

model_kdf.fit(X[indx_to_take_train], y[indx_to_take_train])
model_kdf.global_bias = -10
proba_kdf = model_kdf.predict_proba(X[indx_to_take_test])
get_ece_(proba_kdf, np.argmax(proba_kdf,axis=1), y[indx_to_take_test])
# %%
proba_rf = model_kdf.rf_model.predict_proba(X[indx_to_take_test])
get_ece_(proba_rf, np.argmax(proba_rf,axis=1), y[indx_to_take_test])

# %%
