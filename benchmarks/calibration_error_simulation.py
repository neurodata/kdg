#%%
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier as rf
from kdg.utils import generate_gaussian_parity, generate_ellipse, generate_sinewave, generate_polynomial, generate_spirals, hellinger, get_ace, get_ece
# %%

# %%
samples = [10,100,500,1000, 10000]
n_test = 1000
n_estimators = 100
bins = 30
p = np.arange(-1, 1, step=0.01)
q = np.arange(-1, 1, step=0.01)
xx, yy = np.meshgrid(p, q)
grid_samples = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)
hellinger_dis_rf = [] 
ECE = []
ACE = []

tp_df = pd.read_csv("../true_posterior/Gaussian_xor_pdf.csv")
true_posterior = tp_df['posterior']
true_posterior = np.vstack((true_posterior.ravel(), 1-true_posterior.ravel())).T
for _ in range(10):
    for sample in samples:
        X, y = generate_gaussian_parity(sample)
        X_test, y_test = generate_gaussian_parity(n_test)

        rf_model = rf(n_estimators=n_estimators)
        rf_model.fit(X, y)

        proba_rf = rf_model.predict_proba(grid_samples)
        proba_rf_test = rf_model.predict_proba(X_test)
        hellinger_dis_rf.append(hellinger(true_posterior, proba_rf))
        ECE.append(get_ece(proba_rf_test, y_test, n_bins=bins))
        ACE.append(get_ace(proba_rf_test, y_test, R=bins))


tp_df = pd.read_csv("../true_posterior/ellipse_pdf.csv")
true_posterior = tp_df['posterior']
true_posterior = np.vstack((true_posterior.ravel(), 1-true_posterior.ravel())).T
for _ in range(10):
    for sample in samples:
        X, y = generate_ellipse(sample)
        X_test, y_test = generate_ellipse(n_test)

        rf_model = rf(n_estimators=n_estimators)
        rf_model.fit(X, y)

        proba_rf = rf_model.predict_proba(grid_samples)
        proba_rf_test = rf_model.predict_proba(X_test)
        hellinger_dis_rf.append(hellinger(true_posterior, proba_rf))
        ECE.append(get_ece(proba_rf_test, y_test, n_bins=bins))
        ACE.append(get_ace(proba_rf_test, y_test, R=bins))


tp_df = pd.read_csv("../true_posterior/sinewave_pdf.csv")
true_posterior = tp_df['posterior']
true_posterior = np.vstack((true_posterior.ravel(), 1-true_posterior.ravel())).T

for _ in range(10):
    for sample in samples:
        X, y = generate_sinewave(sample)
        X_test, y_test = generate_sinewave(n_test)

        rf_model = rf(n_estimators=n_estimators)
        rf_model.fit(X, y)

        proba_rf = rf_model.predict_proba(grid_samples)
        proba_rf_test = rf_model.predict_proba(X_test)
        hellinger_dis_rf.append(hellinger(true_posterior, proba_rf))
        ECE.append(get_ece(proba_rf_test, y_test, n_bins=bins))
        ACE.append(get_ace(proba_rf_test, y_test, R=bins))


tp_df = pd.read_csv("../true_posterior/spiral_pdf.csv")
true_posterior = tp_df['posterior']
true_posterior = np.vstack((true_posterior.ravel(), 1-true_posterior.ravel())).T

for _ in range(10):
    for sample in samples:
        X, y = generate_spirals(sample)
        X_test, y_test = generate_spirals(n_test)

        rf_model = rf(n_estimators=n_estimators)
        rf_model.fit(X, y)

        proba_rf = rf_model.predict_proba(grid_samples)
        proba_rf_test = rf_model.predict_proba(X_test)
        hellinger_dis_rf.append(hellinger(true_posterior, proba_rf))
        ECE.append(get_ece(proba_rf_test, y_test, n_bins=bins))
        ACE.append(get_ace(proba_rf_test, y_test, R=bins))


tp_df = pd.read_csv("../true_posterior/polynomial_pdf.csv")
true_posterior = tp_df['posterior']
true_posterior = np.vstack((true_posterior.ravel(), 1-true_posterior.ravel())).T

for _ in range(10):
    for sample in samples:
        X, y = generate_polynomial(sample, a=(1,3))
        X_test, y_test = generate_polynomial(n_test, a=(1,3))

        rf_model = rf(n_estimators=n_estimators)
        rf_model.fit(X, y)

        proba_rf = rf_model.predict_proba(grid_samples)
        proba_rf_test = rf_model.predict_proba(X_test)
        hellinger_dis_rf.append(hellinger(true_posterior, proba_rf))
        ECE.append(get_ece(proba_rf_test, y_test, n_bins=bins))
        ACE.append(get_ace(proba_rf_test, y_test, R=bins))
# %%
plt.scatter(hellinger_dis_rf, ECE)
# %%
plt.scatter(hellinger_dis_rf, ACE)
# %%
