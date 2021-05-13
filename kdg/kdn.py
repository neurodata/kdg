from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
import keras
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
import warnings

class kdn(KernelDensityGraph):

    def __init__(self,
        network,
        covariance_types = 'full', 
        criterion=None,
        complie_kwargs = {
            "loss": "categorical_crossentropy",
            "optimizer": keras.optimizers.Adam(3e-4)
            },
        fit_kwargs = {
            "epochs": 100,
            "batch_size": 32,
            "verbose": False
            }
        ):
        super().__init__()
        self.polytope_means = {}
        self.polytope_cov = {}
        self.network = network
        self.compile_kwargs = complie_kwargs
        self.fit_kwargs = fit_kwargs
        self.covariance_types = covariance_types
        self.criterion = criterion

    def _get_polytopes(self, X):
        polytope_memberships = []
        last_activations = X
        total_layers = len(self.network.layers)

        for layer_id in range(total_layers):
            weights, bias = self.network.layers[layer_id].get_weights()
            preactivation = np.matmul(last_activations, weights) + bias
            if layer_id == total_layers - 1:
                binary_preactivation = (preactivation > 0.5).astype('int')
            else:
                binary_preactivation = (preactivation > 0).astype('int')
            polytope_memberships.append(binary_preactivation)
            last_activations = preactivation * binary_preactivation
        polytope_memberships = [np.tensordot(np.concatenate(polytope_memberships, axis = 1), 2 ** np.arange(0, np.shape(np.concatenate(polytope_memberships, axis = 1))[1]), axes = 1)]

        return polytope_memberships

    def fit(self, X, y):
        r"""
        Fits the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response) data matrix.
        """
        X, y = check_X_y(X, y)
        self.labels = np.unique(y)

        self.network.compile(**self.compile_kwargs)
        self.network.fit(X, keras.utils.to_categorical(y), **self.fit_kwargs)

        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_cov[label] = []

            X_ = X[np.where(y==label)[0]]
            polytopes = self._get_polytopes(X_)[0]
            
            for polytope in polytopes:
                idx = np.where(polytopes==polytope)[0]
                
                if len(idx) == 1:
                        continue
                    
                if self.criterion == None:
                    gm = GaussianMixture(n_components=1, covariance_type=self.covariance_types, reg_covar=1e-4).fit(X_[idx])
                    self.polytope_means[label].append(
                            gm.means_[0]
                    )
                    self.polytope_cov[label].append(
                            gm.covariances_[0]
                    )
                else:
                    min_val = 1e20
                    tmp_means = np.mean(
                        X_[idx],
                        axis=0
                    )
                    tmp_cov = np.var(
                        X_[idx],
                        axis=0
                    )
                        
                    for cov_type in self.covariance_types:
                        try:
                            gm = GaussianMixture(n_components=1, covariance_type=cov_type, reg_covar=1e-3).fit(X_[idx])
                        except:
                            warnings.warn("Could not fit for cov_type "+cov_type)
                        else:
                            if self.criterion == 'aic':
                                constraint = gm.aic(X_[idx])
                            elif self.criterion == 'bic':
                                constraint = gm.bic(X_[idx])

                            if min_val > constraint:
                                min_val = constraint
                                tmp_cov = gm.covariances_[0]
                                tmp_means = gm.means_[0]
                            
                    self.polytope_means[label].append(
                        tmp_means
                    )
                    self.polytope_cov[label].append(
                        tmp_cov
                    )

    def _compute_pdf(self, X, label, polytope_idx):
        polytope_mean = self.polytope_means[label][polytope_idx]
        polytope_cov = self.polytope_cov[label][polytope_idx]

        var = multivariate_normal(
            mean=polytope_mean, 
            cov=polytope_cov, 
            allow_singular=True
            )

        likelihood = var.pdf(X)
        return likelihood

    def predict_proba(self, X):
        r"""
        Calculate posteriors using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        X = check_array(X)

        likelihoods = np.zeros(
            (np.size(X,0), len(self.labels)),
            dtype=float
        )
        
        for ii,label in enumerate(self.labels):
            for polytope_idx,_ in enumerate(self.polytope_means[label]):
                likelihoods[:,ii] += np.nan_to_num(self._compute_pdf(X, label, polytope_idx))

        proba = (likelihoods.T/(np.sum(likelihoods,axis=1)+1e-100)).T
        return proba

    def predict(self, X):
        r"""
        Perform inference using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        return np.argmax(self.predict_proba(X), axis = 1)

