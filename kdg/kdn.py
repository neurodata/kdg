from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from tensorflow import keras
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
import warnings
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import itertools

class kdn(KernelDensityGraph):

    def __init__(self,
        network,
        polytope_compute_method = 'all', # 'all': all the FC layers, 'pl': only the penultimate layer
        T=2, 
        weighting_method = None, # 'TM', 'FM', 'LL', 'AP'
        verbose=True
        ):
        super().__init__()
        self.polytope_means = {}
        self.polytope_cov = {}
        self.network = network
        self.polytope_compute_method = polytope_compute_method
        self.T = T
        self.weighting_method = weighting_method
        self.verbose = verbose

        self.total_layers = len(self.network.layers)

    def _get_polytope_memberships(self, X):
        polytope_memberships = []
        last_activations = X

        # Iterate through neural network manually, getting node activations at each step
        for layer_id in range(self.total_layers):
            weights, bias = self.network.layers[layer_id].get_weights()

            # Calculate new activations based on input to this layer
            preactivation = np.matmul(last_activations, weights) + bias

             # get list of activated nodes in this layer
            if layer_id == self.total_layers - 1:
                binary_preactivation = (preactivation > 0.5).astype('int')
            else:
                binary_preactivation = (preactivation > 0).astype('int')
            
            if self.polytope_compute_method == 'pl':
                # determine the polytope memberships only based on the penultimate layer (uncomment )
                if layer_id == self.total_layers - 2:
                    polytope_memberships.append(binary_preactivation)

            if self.polytope_compute_method == 'all':
                # determine the polytope memberships only based on all the FC layers (uncomment)
                polytope_memberships.append(binary_preactivation)
            
            # remove all nodes that were not activated
            last_activations = preactivation * binary_preactivation

        #Concatenate all activations for given observation
        polytope_obs = np.concatenate(polytope_memberships, axis = 1)
        polytope_memberships = [np.tensordot(polytope_obs, 2 ** np.arange(0, np.shape(polytope_obs)[1]), axes = 1)]
        
        self.num_fc_neurons = polytope_obs.shape[1] # get the number of total FC neurons under consideration

        return polytope_memberships

    def _get_activation_pattern(self, polytope_id):
        binary_string = np.binary_repr(polytope_id, width=self.num_fc_neurons)[::-1] 
        return np.array(list(binary_string)).astype('int')

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
            unique_polytope_ids = np.unique(polytope_memberships) # get the unique polytopes
            
            if self.verbose:
                print("Number of Polytopes : ", len(polytope_memberships))
                print("Number of Unique Polytopes : ", len(unique_polytope_ids))
            
            polytope_member_count = [] # store the polytope member counts
            for polytope_id in unique_polytope_ids: # fit Gaussians for each unique non-singleton polytope
                idx = np.where(polytope_memberships==polytope_id)[0] # collect the samples that belong to the current polytope
                polytope_member_count.append(len(idx))

                if len(idx) < self.T: # don't fit a gaussian to polytopes that has less members than the specified threshold
                    continue

                # get the activation pattern of the current polytope
                a_native = self._get_activation_pattern(polytope_id)
                
                if self.weighting_method == 'AP':
                    disregard_polytope = False
                    start = 0
                    A_native = []
                    for layer_id in range(self.total_layers):
                        layer_num_neurons = self.network.layers[layer_id].output_shape[-1]
                        end = start + layer_num_neurons
                        layer_a_ref = a_native[start:end]*np.arange(1, layer_num_neurons+1, 1)
                        start = end
                        if len(layer_a_ref[layer_a_ref!=0]) == 0:
                            disregard_polytope = True
                            break
                        A_native.append(layer_a_ref[layer_a_ref!=0])
                        
                    if disregard_polytope:
                        continue

                    P_native = np.array([k for k in itertools.product(A_native[0], A_native[1], A_native[2])])
                    P_native = np.array([str(P_native[l]) for l in range(len(P_native))])

                # compute the weights
                weights = []
                for member_polytope_id in polytope_memberships:
                    a_foreign = self._get_activation_pattern(member_polytope_id)
                    
                    match_status = a_foreign == a_native
                    match_status = match_status.astype('int')

                    if self.weighting_method == 'TM' or self.weighting_method == None:
                        # weight based on the total number of matches (uncomment)
                        weight = np.sum(match_status)/self.num_fc_neurons

                    if self.weighting_method == 'FM':
                        # weight based on the first mistmatch (uncomment)
                        if len(np.where(match_status==0)[0]) == 0:
                            weight = 1.0
                        else:
                            first_mismatch_idx = np.where(match_status==0)[0][0]
                            weight = first_mismatch_idx / self.num_fc_neurons

                    if self.weighting_method == 'LL':
                        # layer-by-layer weights
                        weight = 0
                        start = 0
                        for layer_id in range(self.total_layers):
                            num_neurons = self.network.layers[layer_id].output_shape[-1]
                            end = start + num_neurons
                            weight += np.sum(match_status[start:end])/num_neurons
                            start = end
                        weight /= self.total_layers

                    # activation path-based weights
                    if self.weighting_method == 'AP':
                        A_foreign = []
                        start = 0
                        for layer_id in range(self.total_layers):
                            layer_num_neurons = self.network.layers[layer_id].output_shape[-1]
                            end = start + layer_num_neurons
                            layer_a_bar = a_foreign[start:end]*np.arange(1, layer_num_neurons+1, 1)
                            A_foreign.append(layer_a_bar[layer_a_bar!=0])
                            start = end
                        P_foreign = np.array([k for k in itertools.product(A_foreign[0], A_foreign[1], A_foreign[2])])
                        P_foreign = np.array([str(P_foreign[l]) for l in range(len(P_foreign))])
                        weight = len(np.intersect1d(P_native, P_foreign))/len(np.union1d(P_native, P_foreign))
                    
                    weights.append(weight)
                weights = np.array(weights)
                if self.weighting_method == None:
                    weights[weights < 1] = 0 # only use the data from the native polytopes

                # apply weights to the data
                X_tmp = X_.copy()
                polytope_mean_ = np.average(X_tmp, axis=0, weights=weights) # compute the weighted average of the samples 
                X_tmp -= polytope_mean_ # center the data

                sqrt_weights = np.sqrt(weights)
                sqrt_weights = np.expand_dims(sqrt_weights, axis=-1)
                X_tmp *= sqrt_weights # scale the centered data with the square root of the weights

                # compute the paramters of the Gaussian underlying the polytope
                
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

            if self.verbose:
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

