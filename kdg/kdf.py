# Created on Wed Mar 30 11:01:00 PM
# Author: Aishwarya Seth (aseth5@jhu.edu)
# Objective: Kernel Density Forest with Forward Transfer


from enum import unique
from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.stats import multivariate_normal
import warnings
from sklearn.covariance import MinCovDet, fast_mcd, GraphicalLassoCV, LedoitWolf, EmpiricalCovariance, OAS, EllipticEnvelope

class kdf(KernelDensityGraph):
    # Need to create a template forest at some point? 
    def __init__(self, k = 1, kwargs={}):
        print("In the updated KDF!")
        super().__init__()

        #self.polytope_means = {}
        #self.polytope_cov = {}
        self.polytope_means = []
        self.polytope_cov = []
        self.polytope_sizes = {}
        
        self.task_list = []
        
        self.task_labels = {}
        self.class_priors = {}
        self.task_bias = {}

        self.polytope_cardinality = {}
        self.polytope_mean_cov = {}
        self.prior = {}
        self.bias = {}
        self.global_bias = 0
        self.kwargs = kwargs
        self.k = k
        
        self.is_fitted = False

        self.rf_model =  rf(**self.kwargs)

        # #self.compile_kwargs lines
        # self.compile_kwargs = {}
        # if n_estimators is None: self.compile_kwargs["n_estimators"] = rf_model.n_estimators
        # else: self.compile_kwargs["n_estimators"] = n_estimators
        

        #kwargs={'n_estimators':n_estimators}

    def fit(self, X, y, task_id = None, **kwargs):
        r"""
        Fits the kernel density forest.
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
        if self.is_fitted:
            raise ValueError(
                "Model already fitted!"
            )
            return

        X, y = check_X_y(X, y)
        labels = np.unique(y)        
        feature_dim = X.shape[1]

        if task_id is None: task_id = f"task{len(self.task_list)}" 
        self.task_list.append(task_id)
        self.task_labels[task_id] = labels

        #model.compile(**self.compile_kwargs)
        model = self.rf_model.fit(X, y)

        polytope_means = []
        polytope_covs = []
        polytope_sizes = []
        priors = []

        for label in labels:
            X_ = X[np.where(y==label)[0]]
            # print(type(label))
            # print(label)
            # print(len(labels))
            one_hot = np.zeros(len(labels))
            one_hot[label] = 1
            # print(len(one_hot))
            # print(one_hot)

            #self.polytope_cardinality[label] = []

            priors.append(len(X_) / len(X))

            predicted_leaf_ids_across_trees = np.array(
                [tree.apply(X_) for tree in model.estimators_]
                ).T
            polytope_ids = predicted_leaf_ids_across_trees
            polytopes, polytope_count = np.unique(
                predicted_leaf_ids_across_trees, return_inverse=True, axis=0
            )
            unique_polytope_ids = np.unique(polytope_ids)
            # self.polytope_cardinality[label].extend(
            #     polytope_count
            # )
            total_polytopes_this_label = len(polytopes)
            #total_polytopes_this_label = len(unique_polytope_ids)
            total_samples_this_label = X_.shape[0]
            self.prior[label] = total_samples_this_label/X.shape[0]

            for polytope in range(total_polytopes_this_label):
                # print(polytopes)
                # print(polytope)
                # print(unique_polytope_ids)
                matched_samples = np.sum(
                    predicted_leaf_ids_across_trees == polytopes[polytope],
                    axis=1
                )
                idx = np.where(
                    matched_samples>0
                )[0]
                
                if len(idx) == 1:
                    continue
                
                scales = matched_samples[idx]/np.max(matched_samples[idx])
                X_tmp = X_[idx].copy()
                polytope_mean_ = np.average(X_tmp, axis=0, weights=scales)
                X_tmp -= polytope_mean_
                
                sqrt_scales = np.sqrt(scales).reshape(-1,1) @ np.ones(feature_dim).reshape(1,-1)
                X_tmp *= sqrt_scales

                covariance_model = LedoitWolf(assume_centered=True)
                covariance_model.fit(X_tmp)

                # # Why not the sum of scales?
                # polytope_cov_ = (
                #     covariance_model.covariance_*len(idx)/sum(scales)
                # )
                # Why not the sum of scales?
                polytope_cov_ = (
                    covariance_model.covariance_*len(scales)/sum(scales)
                )

                polytope_size_ = len(
                    np.where(polytope_ids == polytope)[0]
                )  # count the number of points in the polytope

                #print(polytope_size_)
                polytope_size_ = polytope_count[polytope]
                #print(polytope_size_)


                #polytope_size_ = polytope_count[label]

                # store the mean, covariances, and polytope sample size
                polytope_means.append(polytope_mean_)
                polytope_covs.append(polytope_cov_)
                polytope_sizes.append(polytope_size_ * one_hot)
                #print(polytope_size_ * one_hot)
                #print(len(polytope_sizes))
                #print(polytope_sizes)

                # self.polytope_means[label].append(
                #     location_
                # )
                # self.polytope_cov[label].append(
                #     covariance_model.covariance_*len(idx)/sum(scales)
                # )
        # print(polytope_sizes[0:5])
        # START OF NEW 
        # append the data we have generated + also pad previously generated polytope sizes with np.nan to
        # maintain n_polytopes x n_labels 
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
        # END OF NEW 

        #Calculate bias
        #print(polytope_sizes.shape)
        likelihood = []
        for polytope in range(start_idx, stop_idx):
            #for label in labels: 
            likelihood.append(self._compute_pdf(X, polytope))
        likelihood = np.array(likelihood)
        #print(polytope_sizes.shape)
        # bias over all of the Gaussians 
        # print("Likelihood Shape: ", likelihood.shape)
        # print("Polytope_Size Length: ", len(polytope_sizes))
        bias = np.sum(np.min(likelihood, axis = 1) * np.sum(polytope_sizes, axis = 1)) / self.k / np.sum(polytope_sizes)
        # bias = np.sum(np.min(likelihood, axis=1) * np.sum(self.polytope_sizes[task_id], axis=1)) / self.k / np.sum(self.polytope_sizes[task_id])
        self.task_bias[task_id] = bias
        self.class_priors[task_id] = np.array(priors)
            # ## calculate bias for each label
            # likelihoods = np.zeros(
            #     (np.size(X_,0)),
            #     dtype=float
            # )
            # for polytope_idx,_ in enumerate(self.polytope_means[label]):
            #     likelihoods += np.nan_to_num(self._compute_pdf(X_, label, polytope_idx))

            # likelihoods /= total_samples_this_label
            # self.bias[label] = np.min(likelihoods)/(self.k*total_samples_this_label)

        self.global_bias = 0 #min(self.bias.values())
        # self.is_fitted = True

    def generate_data(self, n_data, task_id, force_equal_priors = True):
        r"""
        Generate new data using existing polytopes
        Parameters:
        -----------
        n_data: int
            total size of data to return
        task_id : int or string
            Task that data will be an instance of. If task_id is an integer, then use as index. Otherwise use as task id directly.
        force_equal_priors : bool
            If True, generated data will be equally distributed between all labels.
            If False, generated data will be distributed between labels in proportion to the existing priors.
        
        Returns:
        ndarray
            Input data matrix.
            Output (i.e. response) data matrix.
        """
        if isinstance(task_id, int):
            task_id = self.task_list[task_id]
        labels = self.task_labels[task_id]
        n_labels = len(labels)
        n_data = int(n_data)

        X = []
        y = []
        
        if force_equal_priors:
            X_label = np.full(n_labels, n_data/n_labels)
        else: 
            X_label = n_data * self.class_priors[task_id]
        X_label = X_label.astype(int)
        if np.sum(X_label) < n_data :
            X_label[-1] = X_label[-1] + 1
        
        for i in range(n_labels):
            index = np.cumsum(np.nan_to_num(self.polytope_sizes[task_id][:,i]))
            polytopes = np.random.randint(0, index[-1], X_label[i])
            polytope_size = [np.count_nonzero(j > polytopes) for j in index]
            polytope_size = polytope_size - np.concatenate(([0], polytope_size[0:-1]))
            for polytope, size in enumerate(polytope_size):
                if size > 0:
                    xi = np.random.multivariate_normal(self.polytope_means[polytope],
                                                       self.polytope_covs[polytope],
                                                       size)
                    yi = np.full(size, i)
                    X.append(xi)
                    y.append(yi)
                    
        return np.concatenate(X), np.concatenate(y)

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
        # find np.nan parts & use the new data from generate_data 
        # once you've done one forward transfer, don't do it again 
        # fit it once only 
        # nans are used to find polytopes for which we're doing forward transfer to 
        # relies only on polytopes -- should not need to change this for KDF 

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
        # print(polytope_idx)
        # print(self.polytope_cov)
        # polytope_mean = self.polytope_means[label][polytope_idx]
        # polytope_cov = self.polytope_cov[label][polytope_idx]
        polytope_mean = self.polytope_means[polytope_idx]
        polytope_cov = self.polytope_covs[polytope_idx]
        var = multivariate_normal(
            mean=polytope_mean, 
            cov=polytope_cov, 
            allow_singular=True
            )

        #likelihood = self.polytope_cardinality[label][polytope_idx]*var.pdf(X)
        #likelihood = self.polytope_sizes[polytope_idx] * var.pdf(X)
        likelihood = var.pdf(X)
        return likelihood

    

    def predict_proba(self, X, task_id, return_likelihood=False):
        r"""
        Calculate posteriors using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        
        X = check_array(X)

        if isinstance(task_id, int):
            task_id = self.task_list[task_id]
            
        labels = self.task_labels[task_id]

        likelihoods = np.zeros(
            (np.size(X,0), len(labels)),
            dtype=float
        )
        
        for polytope, sizes in enumerate(self.polytope_sizes[task_id]):
            likelihoods += np.nan_to_num(
                    np.outer(self._compute_pdf(X, polytope), sizes)
                )
        priors = self.class_priors[task_id]
        priors = np.reshape(priors, (len(priors), 1))
        
        # for ii,label in enumerate(self.labels):
        #     total_polytopes = len(self.polytope_means[label])
        #     for polytope_idx,_ in enumerate(self.polytope_means[label]):
        #         likelihoods[:,ii] += self.prior[label] * np.nan_to_num(self._compute_pdf(X, label, polytope_idx))

        #     likelihoods[:,ii] = likelihoods[:,ii]/total_polytopes
        #     #likelihoods[:,ii] += self.global_bias
        #     likelihoods[:, ii] += self.task_bias[task_id]

        likelihoods += self.task_bias[task_id]
        proba = (
            likelihoods.T * priors / (np.sum(likelihoods.T * priors, axis=0) + 1e-100)
        ).T        
        #proba = (likelihoods.T/np.sum(likelihoods,axis=1)).T
        
        if return_likelihood:
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
        #return np.argmax(self.predict_proba(X), axis = 1)
