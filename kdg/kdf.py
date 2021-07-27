from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.stats import multivariate_normal
import warnings

class kdf(KernelDensityGraph):

    def __init__(self, covariance_types = 'full', criterion=None, kwargs={}):
        super().__init__()
        
        if isinstance(covariance_types, str)==False and criterion == None:
            raise ValueError(
                    "The criterion cannot be None when there are more than 1 covariance_types."
                )
            return

        self.polytope_means = {}
        self.polytope_cov = {}
        self.polytope_cardinality = {}
        self.polytope_mean_cov = {}
        self.kwargs = kwargs
        self.covariance_types = covariance_types
        self.criterion = criterion

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
        self.rf_model = rf(**self.kwargs).fit(X, y)
        feature_dim = X.shape[1]
        
        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_cov[label] = []

            X_ = X[np.where(y==label)[0]]
            predicted_leaf_ids_across_trees = np.array(
                [tree.apply(X_) for tree in self.rf_model.estimators_]
                ).T
            total_polytopes_this_label = len(X_)

            for polytope in range(total_polytopes_this_label):
                matched_samples = np.sum(
                    predicted_leaf_ids_across_trees == predicted_leaf_ids_across_trees[polytope],
                    axis=1
                )
                idx = np.where(
                    matched_samples>0
                )[0]

                if len(idx) == 1:
                    continue
                

                if self.criterion == None:
                    gm = GaussianMixture(n_components=1, covariance_type=self.covariance_types, reg_covar=1e-4).fit(X_[idx])
                    self.polytope_means[label].append(
                            gm.means_[0]
                    )
                    tmp_cov = gm.covariances_[0]
                    if self.covariance_types == 'spherical':
                        tmp_cov = np.eye(feature_dim)*tmp_cov
                    elif self.covariance_types == 'diag':
                        tmp_cov = np.eye(len(tmp_cov)) * tmp_cov

                    self.polytope_cov[label].append(
                            tmp_cov
                    )
                else:
                    min_val = np.inf
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

                                if cov_type == 'spherical':
                                    tmp_cov = np.eye(feature_dim)*tmp_cov
                                elif cov_type == 'diag':
                                    tmp_cov = np.eye(len(tmp_cov)) * tmp_cov

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