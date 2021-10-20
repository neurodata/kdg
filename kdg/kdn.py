from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from tensorflow import keras
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
import warnings
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

class kdn(KernelDensityGraph):

    def __init__(self,
        network,
        covariance_types = 'full', 
        criterion=None
        ):
        super().__init__()
        self.polytope_means = {}
        self.polytope_cov = {}
        self.network = network
        self.covariance_types = covariance_types
        self.criterion = criterion

    def _get_polytope_memberships(self, X):
        polytope_memberships = []
        last_activations = X
        total_layers = len(self.network.layers)

        # Iterate through neural network manually, getting node activations at each step
        for layer_id in range(total_layers):
            weights, bias = self.network.layers[layer_id].get_weights()

            # Calculate new activations based on input to this layer
            preactivation = np.matmul(last_activations, weights) + bias

             # get list of activated nodes in this layer
            if layer_id == total_layers - 1:
                binary_preactivation = (preactivation > 0.5).astype('int')
            else:
                binary_preactivation = (preactivation > 0).astype('int')
            
            # # determine the polytope memberships only based on the penultimate layer (uncomment )
            # if layer_id == total_layers - 2:
            #   polytope_memberships.append(binary_preactivation)

            # determine the polytope memberships only based on all the FC layers (uncomment)
            polytope_memberships.append(binary_preactivation)
            
            # remove all nodes that were not activated
            last_activations = preactivation * binary_preactivation

        #Concatenate all activations for given observation
        polytope_obs = np.concatenate(polytope_memberships, axis = 1)
        polytope_memberships = [np.tensordot(polytope_obs, 2 ** np.arange(0, np.shape(polytope_obs)[1]), axes = 1)]
        
        self.num_fc_neurons = polytope_obs.shape[1] # get the number of total FC neurons under consideration

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

        feature_dim = X.shape[1]
        
        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_cov[label] = []

            X_ = X[np.where(y==label)[0]]
            polytope_memberships = self._get_polytope_memberships(X_)[0]
            unique_polytopes = np.unique(polytope_memberships) # get the unique polytopes
            # print("Number of Polytopes : ", len(polytope_memberships))
            # print("Number of Unique Polytopes : ", len(unique_polytopes))
            
            polytope_member_count = [] # store the polytope member counts
            for polytope in unique_polytopes: # fit Gaussians for each unique non-singleton polytope
                idx = np.where(polytope_memberships==polytope)[0] # collect the samples that belong to the current polytope
                polytope_member_count.append(len(idx))

                if len(idx) == 1: # don't fit a gaussian to polytopes that has less members than the specified threshold
                    continue

                # get the activation pattern of the current polytope
                native_polytope_activation = np.binary_repr(polytope, width=self.num_fc_neurons)[::-1] 
                a_native = np.array(list(native_polytope_activation)).astype('int')

                # compute the weights
                weights = []
                for member in polytope_memberships:
                    member_activation = np.binary_repr(member, width=self.num_fc_neurons)[::-1]
                    a_member = np.array(list(member_activation)).astype('int')
                    
                    match_status = a_member == a_native
                    match_status = match_status.astype('int')

                    # # weight based on the total number of matches (uncomment)
                    # weight = np.sum(match_status)/self.num_fc_neurons

                    # weight based on the first mistmatch (uncomment)
                    if len(np.where(match_status==0)[0]) == 0:
                        weight = 1.0
                    else:
                        first_mismatch_idx = np.where(match_status==0)[0][0]
                        weight = first_mismatch_idx / self.num_fc_neurons

                    # # layer-by-layer weights
                    # total_layers = len(self.network.layers)
                    # weight = 0
                    # start = 0
                    # for layer_id in range(total_layers):
                    #     num_neurons = self.network.layers[layer_id].output_shape[-1]
                    #     end = start + num_neurons
                    #     weight += np.sum(match_status[start:end])/num_neurons
                    #     start = end
                    # weight /= total_layers

                    weights.append(weight)
                weights = np.array(weights)
                weights[weights < 1] = 0 # only use the data from the native polytope

                X_tmp = X_.copy()
                polytope_mean_ = np.average(X_tmp, axis=0, weights=weights) # compute the weighted average of the samples 
                X_tmp -= polytope_mean_ # center the data

                sqrt_weights = np.sqrt(weights)
                sqrt_weights = np.expand_dims(sqrt_weights, axis=-1)
                X_tmp *= sqrt_weights # scale the centered data with the square root of the weights

                # compute the paramters of the Gaussian underlying the polytope
                 
                # # Gaussian Mixture Model (uncomment)
                # gm = GaussianMixture(n_components=1, covariance_type=self.covariance_types, reg_covar=1e-4).fit(X_[idx])
                # polytope_mean_ = gm.means_[0]
                # polytope_cov_ = gm.covariances_[0]
                
                # LedoitWolf Estimator (uncomment)
                covariance_model = LedoitWolf(assume_centered=True)
                covariance_model.fit(X_tmp)
                polytope_cov_ = covariance_model.covariance_ * len(weights) / sum(weights)

                # store the mean and covariances
                self.polytope_means[label].append(
                        polytope_mean_
                )
                self.polytope_cov[label].append(
                        polytope_cov_
                )
            plt.hist(polytope_member_count, bins=30)
            plt.xlabel("Number of Members")
            plt.ylabel("Number of Polytopes")
            plt.show()

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

