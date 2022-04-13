#
# Created on Mon Jan 31 2022 10:02:26 AM
# Author: Ashwin De Silva (ldesilv2@jhu.edu)
# Objective: Kernel Density Network
#

# import standard libraries
from .base import KernelDensityGraph
from sklearn.utils.validation import check_array, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf


class kdn(KernelDensityGraph):
    def __init__(
        self,
        network,
        weighting=True,
        k=1.0,
        T=1e-3,
        h=0.33,
        verbose=True,
    ):
        r"""[summary]

        Parameters
        ----------
        network : tf.keras.Model()
            trained neural network model
        weighting : bool, optional
            use weighting if true, by default True
        k : float, optional
            bias control parameter, by default 1
        T : float, optional
            neighborhood size control parameter, by default 1e-3
        h : float, optional
            variational parameters of the weighting, by default 0.33
        verbose : bool, optional
            print internal data, by default True
        """
        super().__init__()
        self.polytope_means = {}
        self.polytope_covs = {}
        self.polytope_samples = {}
        self.class_priors = {}
        self.network = network
        self.weighting = weighting
        self.k = k
        self.h = h
        self.T = T
        self.bias = {}
        self.verbose = verbose

        # total number of layers in the NN
        self.total_layers = len(self.network.layers)

        # get the sizes of each layer
        self.network_shape = []
        for layer in network.layers:
            self.network_shape.append(layer.output_shape[-1])

        # total number of units in the network (up to the penultimate layer)
        self.num_neurons = sum(self.network_shape) - self.network_shape[-1]

        # get the weights and biases of the trained MLP
        self.weights = {}
        self.biases = {}
        for i in range(len(self.network.layers)):
            weight, bias = self.network.layers[i].get_weights()
            self.weights[i], self.biases[i] = weight, bias.reshape(1, -1)

    def _get_polytope_ids(self, X):
        r"""
        Obtain the polytope ID of each input sample
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        polytope_ids_tmp = []
        last_activations = X

        # Iterate through neural network manually, getting node activations at each step
        for l in range(self.total_layers):
            weights, bias = self.weights[l], self.biases[l]
            preactivation = np.matmul(last_activations, weights) + bias

            if l == self.total_layers - 1:
                binary_preactivation = (preactivation > 0.5).astype("int")
            else:
                binary_preactivation = (preactivation > 0).astype("int")

            if (
                l < self.total_layers - 1
            ):  # record the activation patterns only upto the penultimate layer
                polytope_ids_tmp.append(binary_preactivation)

            last_activations = preactivation * binary_preactivation

        # Concatenate all activations for given observation
        polytope_ids_tmp = np.concatenate(polytope_ids_tmp, axis=1)
        polytope_ids = [
            np.tensordot(
                polytope_ids_tmp,
                2 ** np.arange(0, np.shape(polytope_ids_tmp)[1]),
                axes=1,
            )
        ]

        self.num_neurons = polytope_ids_tmp.shape[
            1
        ]  # get the number of total FC neurons under consideration

        return polytope_ids[0]

    def _get_activation_pattern(self, polytope_id):
        r"""get the ReLU activation pattern given the polytope ID

        Parameters
        ----------
        polytope_id : int
            polytope identifier

        Returns
        -------
        ndarray
            ReLU activation pattern (binary) corresponding to the given polytope ID
        """
        binary_string = np.binary_repr(polytope_id, width=self.num_neurons)[::-1]
        return np.array(list(binary_string)).astype("int")

    def compute_weights(self, X_, polytope_id):
        """compute weights based on the global network linearity measure
        Parameters
        ----------
        X_ : ndarray
            Input data matrix
        polytope_id : int
            refernce polytope identifier
        Returns
        -------
        ndarray
            weights of each input sample in the input data matrix
        """

        M_ref = self._get_activation_pattern(polytope_id)

        start = 0
        A = X_
        A_ref = X_
        d = 0
        for l in range(len(self.network_shape) - 1):
            end = start + self.network_shape[l]
            M_l = M_ref[start:end]
            start = end
            W, B = self.weights[l], self.biases[l]
            pre_A = A @ W + B
            A = np.maximum(0, pre_A)
            pre_A_ref = A_ref @ W + B
            A_ref = pre_A_ref @ np.diag(M_l)
            d += np.linalg.norm(A - A_ref, axis=1, ord=2)

        return np.exp(-d / self.h)

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
            self.polytope_covs[label] = []
            self.polytope_samples[label] = []

            X_ = X[np.where(y == label)[0]]  # data having the current label

            # get class prior probability
            self.class_priors[label] = len(X_) / len(X)

            # get polytope ids and unique polytope ids
            polytope_ids = self._get_polytope_ids(X_)
            unique_polytope_ids = np.unique(polytope_ids)

            for polytope in unique_polytope_ids:
                weights = self.compute_weights(X_, polytope)
                if not self.weighting:
                    weights[weights < 1] = 0
                weights[weights < self.T] = 0  # set very small weights to zero

                points_with_nonzero_weights = len(np.where(weights > 0)[0])
                if points_with_nonzero_weights < 2:
                    continue

                # apply weights to the data
                X_tmp = X_.copy()
                polytope_mean_ = np.average(
                    X_tmp, axis=0, weights=weights
                )  # compute the weighted average of the samples
                X_tmp -= polytope_mean_  # center the data

                polytope_cov_ = np.average(X_tmp**2, axis=0, weights=weights)

                polytope_samples_ = len(
                    np.where(polytope_ids == polytope)[0]
                )  # count the number of points in the polytope

                # store the mean, covariances, and polytope sample size
                self.polytope_means[label].append(polytope_mean_)
                self.polytope_covs[label].append(polytope_cov_)
                self.polytope_samples[label].append(polytope_samples_)

            ## calculate bias for each label
            likelihoods = np.zeros((np.size(X_, 0)), dtype=float)

            for polytope, _ in enumerate(self.polytope_means[label]):
                likelihoods += np.nan_to_num(
                    self.polytope_samples[label][polytope]
                    * self._compute_pdf(X_, label, polytope)
                )

            likelihoods /= sum(self.polytope_samples[label])
            self.bias[label] = np.min(likelihoods) / (
                self.k * sum(self.polytope_samples[label])
            )

    def _compute_pdf(self, X, label, polytope_id):
        r"""compute the likelihood for the given data

        Parameters
        ----------
        X : ndarray
            Input data matrix
        label : int
            class label
        polytope_idx : int
            polytope identifier

        Returns
        -------
        ndarray
            likelihoods
        """
        polytope_mean = self.polytope_means[label][polytope_id]
        polytope_cov = self.polytope_covs[label][polytope_id]

        var = multivariate_normal(
            mean=polytope_mean, 
            cov=np.eye(len(polytope_cov))*polytope_cov,
            allow_singular=True
        )

        likelihood = var.pdf(X)
        return likelihood

    def predict_proba(self, X, return_likelihoods=False):
        r"""
        Calculate posteriors using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        X = check_array(X)

        likelihoods = np.zeros((np.size(X, 0), len(self.labels)), dtype=float)
        priors = np.zeros((len(self.labels), 1))
        for ii, label in enumerate(self.labels):
            priors[ii] = self.class_priors[label]
            for polytope, _ in enumerate(self.polytope_means[label]):
                likelihoods[:, ii] += np.nan_to_num(
                    self.polytope_samples[label][polytope]
                    * self._compute_pdf(X, label, polytope)
                )

            likelihoods[:, ii] = likelihoods[:, ii] / sum(self.polytope_samples[label])
            likelihoods[:, ii] += min(self.bias.values())

        proba = (
            likelihoods.T * priors / (np.sum(likelihoods * priors.T, axis=1) + 1e-100)
        ).T
        if return_likelihoods:
            return proba, likelihoods
        else:
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
