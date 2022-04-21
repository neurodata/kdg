#%%
import numpy as np
from kdg import kdf
from kdg.utils import get_ece, generate_gaussian_parity
from sklearn.ensemble import RandomForestClassifier as rf
# %%
train_samples = [10,100,1000,10000]
test_sample = 1000
reps = 20
performance_indx = []

for samples in train_samples:
    ECE = []
    for _ in range(reps):
        X, y = generate_gaussian_parity(samples)
        X_test, y_test = generate_gaussian_parity(test_sample)
        model_rf = rf(n_estimators=500)
        model_rf.fit(X, y)
        proba_rf = model_rf.predict_proba(X_test)

        ECE.append(get_ece(proba_rf, np.argmax(proba_rf,axis=1), y_test))

    performance_indx.append(
        np.mean(ECE)
    )

#%%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(train_samples, performance_indx)
ax.set_xlabel('Sample #')
ax.set_ylabel('ECE')
plt.savefig('plots/test_ece.pdf')
# %%
