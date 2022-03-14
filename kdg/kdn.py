#
# Created on Mon Mar 7 2022 11:18:00 PM
# Author: Ashwin De Silva (ldesilv2@jhu.edu), Tiffany Chu (tchu13@jhu.edu)
# Objective: Kernel Density Network with Forward Transfer
#

# import standard libraries
from .base import KernelDensityGraph
from sklearn.utils.validation import check_array, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf
from tensorflow import keras

class kdn(KernelDensityGraph):
    def __init__(
        self,
        network,
        weighting=True,
        k=1.0,
        T=1e-3,
        h=0.33,
        optimizer="rmsprop",
        loss=None,
        verbose=True
    ):
        r"""[summary]

        Parameters
        ----------
        network : tf.keras.Model()
            Neural network model, will be cloned for each task. Can be compiled or not.
        weighting : bool, optional
            use weighting if true, by default True
        k : float, optional
            bias control parameter, by default 1
        T : float, optional
            neighborhood size control parameter, by default 1e-3
        h : float, optional
            variational parameters of the weighting, by default 0.33
        optimizer : keras.optimizer object, optional
            Optimizer used to compile network. If None, will use optimizer from network
        loss : string, optional
            Loss function used to compile network. If None, will use loss function from network
        verbose : bool, optional
            print internal data, by default True
        """
        super().__init__()
        self.polytope_means = []
        self.polytope_covs = []
        self.polytope_sizes = {}
        
        self.task_list = []

        self.task_labels = {}
        self.class_priors = {}
        self.task_bias = {}
        
        #Need to fit the network within fit()
        self.network = keras.models.clone_model(network)
        self.weighting = weighting
        self.k = k
        self.h = h
        self.T = T
        self.verbose = verbose
        
        self.compile_kwargs = {}
        if loss is None: self.compile_kwargs["loss"] = network.loss
        else: self.compile_kwargs["loss"] = loss
        if optimizer is None: self.compile_kwargs["optimizer"] = network.optimizer
        else: self.compile_kwargs["optimizer"] = optimizer
                
        # total number of layers in the NN
        self.total_layers = len(self.network.layers)

        # get the sizes of each layer - the last one is not used
        self.network_shape = []
        for layer in network.layers:
            self.network_shape.append(layer.output_shape[-1])

        # total number of units in the network (up to the penultimate layer)
        self.num_neurons = sum(self.network_shape) - self.network_shape[-1]

        # get the weights and biases of the trained MLP
        #self.weights = {}
        #self.biases = {}
        #for i in range(len(self.network.layers)):
        #    weight, bias = self.network.layers[i].get_weights()
        #    self.weights[i], self.biases[i] = weight, bias.reshape(1, -1)

    def _get_polytope_ids(self, X, X_network):
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
            weight, bias = X_network.layers[l].get_weights()
            preactivation = np.matmul(last_activations, weight) + bias

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

    def compute_weights(self, X_, X_network, polytope_id):
        """compute weights based on the global network linearity measure
        Parameters
        ----------
        X_ : ndarray
            Input data matrix
        X_network: keras.Sequential network
            Network fit to X dataset
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
        for l in range(self.total_layers - 1):
            weight, bias = X_network.layers[l].get_weights()
            end = start + self.network_shape[l]
            M_l = M_ref[start:end]
            start = end
            pre_A = A @ weight + bias
            A_ref = pre_A @ np.diag(M_l)
            A = np.maximum(0, pre_A)
            d += np.linalg.norm(A - A_ref, axis=1, ord=2)

        return np.exp(-d / self.h)

    def fit(self, X, y, task_id = None, **kwargs):
        r"""
        Add a task to the multi-task kernel density network.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response) data matrix.
        task_id : string, optional
            Name used to identify task
        kwargs : dict, optional
            Additional arguments to pass to keras fit
        """
        X, y = check_X_y(X, y)
        labels = np.unique(y)
        feature_dim = X.shape[1]
        
        if task_id is None: task_id = f"task{len(self.task_list)}" 
        self.task_list.append(task_id)
        self.task_labels[task_id] = labels
        
        #create and fit neural network
        X_network = keras.models.clone_model(self.network)
        #replace labeling layer
        X_network.pop()
        X_network.add(keras.layers.Dense(units=len(labels), activation = 'softmax'))
        X_network.compile(**self.compile_kwargs)
        X_network.fit(X, keras.utils.to_categorical(y), **kwargs)
        
        polytope_means = []
        polytope_covs = []
        polytope_sizes = []
        priors = []
        for label in labels:
            X_ = X[np.where(y == label)[0]]  # data having the current label
            one_hot = np.zeros(len(labels))
            one_hot[label] = 1

            # get class prior probability
            priors.append(len(X_) / len(X))

            # get polytope ids and unique polytope ids
            polytope_ids = self._get_polytope_ids(X_, X_network)
            unique_polytope_ids = np.unique(polytope_ids)

            for polytope in unique_polytope_ids:
                weights = self.compute_weights(X_, X_network, polytope)
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

                sqrt_weights = np.sqrt(weights).reshape(-1, 1) @ np.ones(
                    feature_dim
                ).reshape(1, -1)
                X_tmp *= sqrt_weights  # scale the centered data with the square root of the weights

                covariance_model = LedoitWolf(assume_centered=True)
                covariance_model.fit(X_tmp)
                polytope_cov_ = (
                    covariance_model.covariance_ * len(weights) / sum(weights)
                )

                polytope_size_ = len(
                    np.where(polytope_ids == polytope)[0]
                )  # count the number of points in the polytope

                # store the mean, covariances, and polytope sample size
                polytope_means.append(polytope_mean_)
                polytope_covs.append(polytope_cov_)
                polytope_sizes.append(polytope_size_ * one_hot)
                        
        #save calculations for all polytopes
        start_idx = len(self.polytope_means)
        stop_idx = len(polytope_means) + start_idx
        if start_idx == 0:
            self.polytope_means = np.array(polytope_means)
            self.polytope_covs = np.array(polytope_covs)
            self.polytope_sizes[task_id] = np.array(polytope_sizes)
        else:
            self.polytope_means = np.concatenate([self.polytope_means, np.array(polytope_means)])
            self.polytope_covs = np.concatenate([self.polytope_covs, np.array(polytope_covs)])
            self.polytope_sizes[task_id] = np.concatenate([np.full([start_idx, len(labels)], fill_value=np.nan),
                                                           polytope_sizes])
            #pad polytope sizes of previous tasks
            for prev_task in self.task_list[:-1]:
                self.polytope_sizes[prev_task] = np.concatenate([self.polytope_sizes[prev_task],
                                                                 np.full([stop_idx - start_idx,
                                                                          len(self.task_labels[prev_task])],
                                                                         fill_value=np.nan)])
        
        #Calculate bias
        likelihood = []
        for polytope in range(start_idx, stop_idx):
            likelihood.append(self._compute_pdf(X, polytope))
        likelihood = np.array(likelihood)
        
        bias = np.sum(np.min(likelihood, axis=1) * np.sum(polytope_sizes, axis=1)) / self.k / np.sum(polytope_sizes)
        self.task_bias[task_id] = bias
        self.class_priors[task_id] = np.array(priors)

        
    def forward_transfer(self, X, y, task_id):
        r"""
        Forward transfer all previously unused polytopes to the target task based on current data

        Parameters:
        -----------
        X: ndarray
            Input data matrix; training data for current task
        y : ndarray
            Output (i.e. response) data matrix for current task
        task_id : int or string
            Task that data is an instance of. If task_id is an integer, then use as index. Otherwise use as task id directly.
        """

        X = check_array(X)
        if isinstance(task_id, int):
            task_id = self.task_list[task_id]
        labels = self.task_labels[task_id]

        likelihood = []
        for polytope_idx in range(self.polytope_means.shape[0]):
            likelihood.append(self._compute_pdf(X, polytope_idx))
        likelihood = np.array(likelihood)
        
        transfer_idx = np.isnan(self.polytope_sizes[task_id])[:,0].nonzero()[0]
            
        transfer_polytopes = np.argmax(likelihood[transfer_idx,:], axis=0)
        polytope_by_label = [transfer_polytopes[y == label] for label in labels]

        new_sizes = np.zeros([len(transfer_idx), len(labels)])
        for L, _ in enumerate(labels):
            polytope_idxs = np.unique(polytope_by_label[L])
            for idx in polytope_idxs:
                new_sizes[idx, L] = np.sum(polytope_by_label[L] == idx)

        self.polytope_sizes[task_id][transfer_idx, :] = new_sizes

        bias = np.sum(np.min(likelihood, axis=1) * np.sum(self.polytope_sizes[task_id], axis=1)) / self.k / np.sum(self.polytope_sizes[task_id])

        self.task_bias[task_id] = bias
            
    def _compute_pdf(self, X, polytope_idx):
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
        polytope_mean = self.polytope_means[polytope_idx]
        polytope_cov = self.polytope_covs[polytope_idx]

        var = multivariate_normal(
            mean=polytope_mean, cov=polytope_cov, allow_singular=True
        )

        likelihood = var.pdf(X)
        return likelihood
    
    def predict_proba(self, X, task_id, return_likelihoods=False):
        r"""
        Calculate posteriors using the kernel density network.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        task_id : int or string
            Task that data is an instance of. If task_id is an integer, then use as index. Otherwise use as task id directly.
        return_likelihoods : bool
            Whether to return likelihoods as well as array
        
        Returns
        -------
        ndarray
            probability of X belonging to each label
            likelihoods matrix for all polytope
        """
        X = check_array(X)
        if isinstance(task_id, int):
            task_id = self.task_list[task_id]
            
        labels = self.task_labels[task_id]

        likelihood = np.zeros((np.size(X, 0), len(labels)), dtype=float)
        priors = self.class_priors[task_id]
        priors = np.reshape(priors, (len(priors), 1))
        
        for polytope, sizes in enumerate(self.polytope_sizes[task_id]):
            likelihood += np.nan_to_num(
                    np.outer(self._compute_pdf(X, polytope), sizes)
                )
    
        likelihood += self.task_bias[task_id]
        proba = (
            likelihood.T * priors / (np.sum(likelihood.T * priors, axis=0) + 1e-100)
        ).T
        if return_likelihoods:
            return proba, likelihoods
        else:
            return proba
        
    def predict(self, X, task_id):
        r"""
        Perform inference using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        
        Returns
        -------
        ndarray
            predicted labels for each item in X
        """
        if isinstance(task_id, int):
            task_id = self.task_list[task_id]
            
        predictions = np.argmax(self.predict_proba(X, task_id), axis=1)
        return np.array([self.task_labels[task_id][pred] for pred in predictions])
