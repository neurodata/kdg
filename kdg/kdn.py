from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
import keras
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
import warnings
from sklearn.covariance import LedoitWolf

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

        # compute the total number of FC neurons
        self.num_fc_neurons = 0
        for i in range(len(self.network.layers)):
            self.num_fc_neurons += self.network.layers[i].output_shape[1]
        
    def _get_polytope_memberships(self, X):
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
        feature_dim = X.shape[1]
        
        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_cov[label] = []

            X_ = X[np.where(y==label)[0]]
            polytope_memberships = self._get_polytope_memberships(X_)[0]
            unique_polytopes = np.unique(polytope_memberships) # get the unique polytopes
            
            for polytope in unique_polytopes: # fit Gaussians for each unique non-singleton polytope
                idx = np.where(polytope_memberships==polytope)[0] # collect the samples that belong to the current polytope
                
                if len(idx) == 1: # don't fit a gaussian to singleton polytopes
                    continue

                # get the activation pattern of the current polytope
                current_polytope_activation = np.binary_repr(polytope, width=self.num_fc_neurons) 

                # compute the weights
                weights = []
                for member in polytope_memberships:
                    member_activation = np.binary_repr(member, width=self.num_fc_neurons)
                    weight = np.sum(np.array(list(current_polytope_activation))==np.array(list(member_activation)))/num_fc_neurons
                    weights.append(weight)
                weights = np.array(weights)

                X_tmp = X_.copy()
                polytope_mean_ = np.average(X_tmp, axis=0, weights=weights) # compute the weighted average of the samples 
                X_tmp -= polytope_mean_ # center the data

                sqrt_weights = np.sqrt(weights).reshape(-1,1) @ np.ones(feature_dim).reshape(1,-1)
                X_tmp *= sqrt_weights # scale the centered data with the square root of the weights

                # compute the paramters of the Gaussian underlying the polytope
                 
                ## Gaussian Mixture Model
                # gm = GaussianMixture(n_components=1, covariance_type=self.covariance_types, reg_covar=1e-4).fit(X_tmp)
                # polytope_mean_ = gm.means_[0]
                # polytope_cov_ = gm.covariances_[0]
                
                # LedoitWolf Estimator
                covariance_model = LedoitWolf(assume_centered=True)
                covariance_model.fit(X_tmp)
                polytope_cov_ = covariance_model.covariance_

                # store the mean and covariances
                self.polytope_means[label].append(
                        polytope_mean_
                )
                self.polytope_cov[label].append(
                        polytope_cov_
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

