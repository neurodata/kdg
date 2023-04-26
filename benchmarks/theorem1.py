#%%
from sklearn.ensemble import RandomForestClassifier as rf 
from kdg.utils import generate_sinewave, generate_spirals, sample_unifrom_circle, get_ece, hellinger
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# %%
estimators = list(range(1,100,1))
n_samples = 1000
calibration = []
OOD_cal = []
reps = 10

tp_df = pd.read_csv("true_posterior/spiral_pdf.csv")
true_posterior = tp_df['posterior']
true_posterior = np.vstack((true_posterior.ravel(), 1-true_posterior.ravel())).T

p = np.arange(-1, 1, step=0.01)
q = np.arange(-1, 1, step=0.01)
xx, yy = np.meshgrid(p, q)
grid_samples = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)

for estimator in tqdm(estimators):
    hellinger_dis = []
    OOD_dis = []
    for _ in range(reps):

        X_ood = sample_unifrom_circle(n=1000, r=1)
        for r in np.arange(1.5,20,1):
            X_ood = np.concatenate((X_ood,sample_unifrom_circle(n=1000, r=r)),axis=0)

        X, y = generate_spirals(n_samples)
        norms = np.linalg.norm(X,axis=0)
        X /= np.max(norms)

        rf_model = rf(n_estimators=estimator)
        rf_model.fit(X, y)
        proba = rf_model.predict_proba(grid_samples)

        hellinger_dis.append(
            hellinger(true_posterior, proba)
        )
        OOD_proba = np.mean(
            np.max(
                rf_model.predict_proba(X_ood),axis=1
            )
        )
        OOD_dis.append(
            np.abs(
                OOD_proba-0.5
            )
        )
    calibration.append(np.mean(hellinger_dis))
    OOD_cal.append(np.mean(OOD_dis))


# %%
sns.set_context('talk')
fig, ax = plt.subplots(1, 1, figsize=(14,14), constrained_layout=True)
ax.scatter(calibration, OOD_cal)
sns.regplot(x=calibration, y=OOD_cal, ci=False, line_kws={'color':'red'}, ax=ax)

ax.set_xlabel('Helinger Distance', fontsize=65)
ax.set_ylabel('OOD Calibration', fontsize=65)
ax.set_xticks([.35,.4,.45,.5,.55])
#ax.set_yticks([.45,.48,.5])
ax.tick_params(labelsize=50)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('plots/theorem1_spiral_corrected.pdf')
# %%
