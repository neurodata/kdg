#%%
from statistics import mode
import numpy as np
from kdg import kdf
from kdg.utils import get_ece, generate_gaussian_parity
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
    bin_size = 1/R
    total_sample = len(true_label)
    K = predicted_posterior.shape[1]

    score = 0
    for k in range(K):
        posteriors = predicted_posterior[:,k]
        sorted_indx = np.argsort(posteriors)
        for r in range(R):        
            indx = sorted_indx[r*R:(r+1)*R]
            predicted_label_ = predicted_label[indx]
            true_label_ = true_label[indx]

            indx_k = np.where(true_label_ == k)[0]
            acc = (
                np.nan_to_num(np.mean(predicted_label_[indx_k] == k))
                if indx_k.size != 0
                else 0
            )
            conf = np.nan_to_num(np.mean(posteriors[indx_k])) if indx_k.size != 0 else 0

            print(acc, conf)
            score += len(indx) * np.abs(acc - conf)

    score /= (K*total_sample)
    return score

#%%
train_samples = [10, 100, 1000, 10000]
test_sample = 1000
reps = 10
performance_indx = []
performance_indx_rf = []

for samples in train_samples:
    ECE = []
    ECE_rf = []
    for _ in range(reps):
        X, y = generate_parity(samples)
        X_test, y_test = generate_parity(test_sample)
        model_kdf = kdf(kwargs={'n_estimators':500})
        model_kdf.fit(X, y)
        proba_kdf = model_kdf.predict_proba(X_test)
        proba_rf = model_kdf.rf_model.predict_proba(X_test)

        ECE.append(get_ece(proba_kdf, np.argmax(proba_kdf,axis=1), y_test))
        ECE_rf.append(get_ece(proba_rf, np.argmax(proba_rf,axis=1), y_test))

    performance_indx.append(
        np.mean(ECE)
    )
    performance_indx_rf.append(
        np.mean(ECE_rf)
    )

#%%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(train_samples, performance_indx, c='r', linewidth=3, label='kdf')
ax.plot(train_samples, performance_indx_rf, c='b', linewidth=3, label='rf')

ax.set_xlabel('Sample #')
ax.set_ylabel('ECE')
plt.legend()
plt.savefig('plots/test_ece.pdf')
# %%
X, y = generate_gaussian_parity(1000)
X_test, y_test = generate_gaussian_parity(1000)

model_kdf = kdf(k=1e1, kwargs={'n_estimators':500})
model_kdf.fit(X, y)
print(np.mean(model_kdf.predict(X_test)==y_test))
print(np.mean(model_kdf.predict(X)==y))
# %%
