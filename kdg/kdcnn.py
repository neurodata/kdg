from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from tensorflow import keras
from keras import layers
from keras import Model
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
import warnings

class kdcnn(KernelDensityGraph):

    def __init__(self,
        network,
        num_fc_layers, # number of fully connected layers
        covariance_types = 'full', 
        criterion = None,
        compile_kwargs = {
            "loss": "categorical_crossentropy",
            "optimizer": keras.optimizers.Adam(3e-4)
            },
        fit_kwargs = {
            "epochs": 10,
            "batch_size": 32,
            "verbose": True
            }
        ):
        super().__init__()
        self.polytope_means = {}
        self.polytope_cov = {}
        self.network = network
        self.num_fc_layers = num_fc_layers
        self.encoder = None
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs
        self.covariance_types = covariance_types
        self.criterion = criterion

    def _get_polytopes(self, X):
        r"""
        Get the polytopes (neural network activation paths) for a given set of observations.
        
        Parameters
        ----------
        X : ndarray
            Input data matrix.

        num_fc_layers: int
            Number of fully-connected layers in the CNN
            
        Returns
        -------
        polytope_memberships : binary list-of-lists
                               Each list represents activations of nodes in the neural network for a given observation
                               0 = not activated; 1 = activated
        """
        polytope_memberships = []
        last_activations = X
        total_layers = len(self.network.layers)
        fully_connected_layers = np.arange(total_layers-self.num_fc_layers, total_layers, 1) # get the layer IDs of the fully-connected layers

        # Iterate through the fully connected layers, getting node activations at each step
        for layer_id in fully_connected_layers:
            weights, bias = self.network.layers[layer_id].get_weights()
            # Calculate new activations based on input to this layer
            preactivation = np.matmul(last_activations, weights) + bias
            # get list of activated nodes in this layer
            if layer_id == total_layers - 1:
                binary_preactivation = (preactivation > 0.5).astype('int')
            else:
                binary_preactivation = (preactivation > 0).astype('int')
            # append activation results to list
            polytope_memberships.append(binary_preactivation)
            # remove all nodes that were not activated
            last_activations = preactivation * binary_preactivation
        #Concatenate all activations for given observation
        polytope_obs = np.concatenate(polytope_memberships, axis = 1)
        polytope_memberships = [np.tensordot(polytope_obs, 2 ** np.arange(0, np.shape(polytope_obs)[1]), axes = 1)]

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
        
        self.labels = np.unique(y)

        self.network.compile(**self.compile_kwargs)
        self.network.fit(X, keras.utils.to_categorical(y), **self.fit_kwargs)
        feature_dim = X.shape[1]

        # get the encoder outputs
        self.encoder = Model(self.network.input, self.network.layers[-(self.num_fc_layers + 1)].output)
        X = self.encoder.predict(X)
        
        for label in self.labels:
            print(label)
            self.polytope_means[label] = []
            self.polytope_cov[label] = []
            
            # Get all training items that match our given label
            X_ = X[np.where(y==label)[0]]
            
            # Calculate polytope memberships for each observation in X_
            polytopes = self._get_polytopes(X_)[0]
            
            for polytope in polytopes:
                # find all other data with same polytopes
                idx = np.where(polytopes==polytope)[0]
                
                if len(idx) == 1:
                    continue  #skip all calculations if there are no other matching polytopes
                
                if self.criterion == None:
                    # Calculate single Gaussian over data in group
                    # Note: Will this break if we list 2+ covariance types?
                    gm = GaussianMixture(n_components=1, covariance_type=self.covariance_types, reg_covar=1e-4).fit(X_[idx])
                    self.polytope_means[label].append(
                            gm.means_[0]
                    )
                    tmp_cov = gm.covariances_[0]
                    if self.covariance_types == 'spherical':
                        tmp_cov = np.eye(feature_dim) * tmp_cov
                    elif self.covariance_types == 'diag':
                        tmp_cov = np.eye(len(tmp_cov)) * tmp_cov

                    self.polytope_cov[label].append(
                            tmp_cov
                    )
                else:
                    min_val = np.inf
                    # Get means and covariance directly from observations in this group
                    tmp_means = np.mean(
                        X_[idx],
                        axis=0
                    )
                    tmp_cov = np.var(
                        X_[idx],
                        axis=0
                    )
                    tmp_cov = np.eye(len(tmp_cov)) * tmp_cov

                    for cov_type in self.covariance_types:
                        # Note: Why are we using different reg_covar for aic/bic vs None?
                        try:
                            gm = GaussianMixture(n_components=1, covariance_type=cov_type, reg_covar=1e-3).fit(X_[idx])
                        except:
                            warnings.warn("Could not fit for cov_type "+cov_type)
                        else:
                            if self.criterion == 'aic':
                                constraint = gm.aic(X_[idx])
                            elif self.criterion == 'bic':
                                constraint = gm.bic(X_[idx])
                            
                            # If our current min_val > constraint for this cov_type, replace values
                            if min_val > constraint:
                                min_val = constraint
                                tmp_cov = gm.covariances_[0]
                                
                                if cov_type == 'spherical':
                                    tmp_cov = np.eye(feature_dim)*tmp_cov
                                elif cov_type == 'diag':
                                    tmp_cov = np.eye(len(tmp_cov)) * tmp_cov

                                tmp_means = gm.means_[0]
                    
                    # For criterion = None, this fits the first? covariance type in self.covariance_types
                    # For constraint = aic/bic, this will fot output the most constraining cov_type in self.covariance_types
                    self.polytope_means[label].append(
                        tmp_means
                    )
                    self.polytope_cov[label].append(
                        tmp_cov
                    )

    def _compute_pdf(self, X, label, polytope_idx):
        r"""
        Calculate probability density function using the kernel density network for a given group.
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
            mean=polytope_mean, 
            cov=polytope_cov, 
            allow_singular=True
            )

        likelihood = var.pdf(X)
        return likelihood

    def predict_proba(self, X):
        r"""
        Calculate posteriors using the kernel density network.
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
        # get the encoder outputs
        X = self.encoder.predict(X)
        return np.argmax(self.predict_proba(X), axis = 1)