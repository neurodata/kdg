from functools import total_ordering

from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import comb
import warnings
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import itertools
import math


class kdn(KernelDensityGraph):
    def __init__(
        self,
        network,
        k=1,
        polytope_compute_method="all",  # 'all': all the FC layers, 'pl': only the penultimate layer
        weighted=True,
        verbose=True,
    ):
        super().__init__()
        self.polytope_means = {}
        self.polytope_cov = {}
        self.network = network  # fitted neural network used as input
        self.k = k  # bias scaling parameter
        self.polytope_compute_method = polytope_compute_method
        self.weighted = weighted
        self.bias = {}
        self.verbose = verbose

        self.total_layers = len(self.network.layers)

        self.network_shape = []
        for layer in network.layers:
            self.network_shape.append(layer.output_shape[-1])

        self.num_fc_neurons = sum(self.network_shape)

    def _get_polytope_memberships(self, X):
        r"""
        Get the polytopes (neural network activation paths) for a given set of observations.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        Returns
        -------
        polytopes            : numerical id corresponding with individual polytopes
        polytope_memberships : binary list-of-lists
                               Each list represents activations of nodes in the neural network for a given observation
                               0 = not activated; 1 = activated
        """
        polytope_memberships = []
        last_activations = X

        # Iterate through neural network manually, getting node activations at each step
        for layer_id in range(self.total_layers):
            weights, bias = self.network.layers[layer_id].get_weights()

            # Calculate new activations based on input to this layer
            preactivation = np.matmul(last_activations, weights) + bias

            # get list of activated nodes in this layer
            if layer_id == self.total_layers - 1:
                binary_preactivation = (preactivation > 0.5).astype("int")
            else:
                binary_preactivation = (preactivation > 0).astype("int")

            if self.polytope_compute_method == "pl":
                # determine the polytope memberships only based on the penultimate layer
                if layer_id == self.total_layers - 2:
                    polytope_memberships.append(binary_preactivation)

            elif self.polytope_compute_method == "all":
                # determine the polytope memberships only based on all the FC layers
                polytope_memberships.append(binary_preactivation)

            # remove all nodes that were not activated
            last_activations = preactivation * binary_preactivation

        # Concatenate all activations for given observation
        polytope_obs = np.concatenate(polytope_memberships, axis=1)
        # get the number of total FC neurons under consideration
        self.num_fc_neurons = polytope_obs.shape[1]

        polytopes = [
            np.tensordot(
                polytope_obs, 2 ** np.arange(0, np.shape(polytope_obs)[1]), axes=1
            )
        ]
        # return layer-by-layer raw polytope information
        return polytopes[0], polytope_memberships

    def fit(self, X, y):
        r"""
        Fits the kernel density network.
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

            X_ = X[np.where(y == label)[0]]
            polytopes, polytope_memberships = self._get_polytope_memberships(X_)
            _, unique_idx = np.unique(
                polytopes, return_index=True
            )  # get the unique polytopes

            if self.verbose:
                print("Number of Polytopes : ", len(polytopes))
                print("Number of Unique Polytopes : ", len(unique_idx))

            polytope_member_count = []  # store the polytope member counts

            for (
                polytope_idx
            ) in unique_idx:  # fit Gaussians for each unique non-singleton polytope
                polytope_id = polytopes[polytope_idx]
                idx = np.where(polytopes == polytope_id)[
                    0
                ]  # collect the samples that belong to the current polytope
                # polytope_member_count.append(len(idx))

                # compute the weights
                weights = []

                # iterate through all the polytopes
                for n in range(len(polytopes)):
                    # calculate match
                    match_status = []
                    n_nodes = 0
                    for layer_id in range(self.total_layers):
                        layer_match = (
                            polytope_memberships[layer_id][n, :]
                            == polytope_memberships[layer_id][polytope_idx, :]
                        )
                        match_status.append(layer_match.astype("int"))
                        n_nodes += layer_match.shape[0]

                    if self.weighted:
                        # pseudo-ensembled first mismatch
                        weight = 0
                        for layer in match_status:
                            n = layer.shape[0]  # length of layer
                            m = np.sum(layer)  # matches
                            if m == n:  # perfect match
                                weight += n / n_nodes
                            elif m <= math.floor(n / 2):  # break if too few nodes match
                                break
                            else:  # imperfect match, add scaled layer weight and break
                                layer_weight = (2 * m - n) / (n_nodes * (n - m + 1))
                                weight += layer_weight
                                break

                    else:
                        # only use the data from the native polytopes
                        total_matches = sum([np.sum(layer) for layer in match_status])
                        weight = 1 if total_matches == n_nodes else 0

                    weights.append(weight)
                weights = np.array(weights)

                polytope_size = len(weights[weights > 0])
                polytope_member_count.append(polytope_size)

                if (
                    polytope_size == 1
                ):  # don't fit a gaussian to polytopes that have only 1 member
                    continue

                # apply weights to the data
                X_tmp = X_.copy()
                polytope_mean_ = np.average(
                    X_tmp, axis=0, weights=weights
                )  # compute the weighted average of the samples
                X_tmp -= polytope_mean_  # center the data

                sqrt_weights = np.sqrt(weights)
                sqrt_weights = np.expand_dims(sqrt_weights, axis=-1)
                X_tmp *= sqrt_weights  # scale the centered data with the square root of the weights

                # compute the parameters of the Gaussian underlying the polytope

                # LedoitWolf Estimator
                covariance_model = LedoitWolf(assume_centered=True)
                covariance_model.fit(X_tmp)
                polytope_cov_ = (
                    covariance_model.covariance_ * len(weights) / sum(weights)
                )

                # store the mean and covariances
                self.polytope_means[label].append(polytope_mean_)
                self.polytope_cov[label].append(polytope_cov_)

            ## calculate bias for each label
            likelihoods = np.zeros((np.size(X_, 0)), dtype=float)

            for polytope_idx, _ in enumerate(self.polytope_means[label]):
                likelihoods += np.nan_to_num(self._compute_pdf(X_, label, polytope_idx))

            likelihoods /= X_.shape[0]
            self.bias[label] = np.min(likelihoods) / (self.k * X_.shape[0])

            if self.verbose:
                plt.hist(polytope_member_count, bins=30)
                plt.xlabel("Number of Members")
                plt.ylabel("Number of Polytopes")
                plt.show()

    def _compute_pdf(self, X, label, polytope_idx):
        r"""
        Calculate probability density function based on samples in polytope
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        label : string
                A single group we want the PDF for
        polytope_idx : index of a polytope, within label
        """
        polytope_mean = self.polytope_means[label][polytope_idx]
        polytope_cov = self.polytope_cov[label][polytope_idx]

        var = multivariate_normal(
            mean=polytope_mean, cov=polytope_cov, allow_singular=True
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

        likelihoods = np.zeros((np.size(X, 0), len(self.labels)), dtype=float)

        for ii, label in enumerate(self.labels):
            total_polytopes = len(self.polytope_means[label])
            for polytope_idx, _ in enumerate(self.polytope_means[label]):
                likelihoods[:, ii] += np.nan_to_num(
                    self._compute_pdf(X, label, polytope_idx)
                )

            likelihoods[:, ii] = likelihoods[:, ii] / total_polytopes
            likelihoods[:, ii] += self.bias[label]

        proba = (likelihoods.T / (np.sum(likelihoods, axis=1) + 1e-100)).T
        return proba

    def predict_proba_nn(self, X):
        r"""
        Calculate posteriors using the vanilla NN
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        X = check_array(X)

        proba = self.network.predict(X)
        return proba

    def predict(self, X):
        r"""
        Perform inference using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        return np.argmax(self.predict_proba(X), axis=1)
