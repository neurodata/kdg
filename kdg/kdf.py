from .base import KernelDensityGraph
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import RandomForestClassifier as rf 
import numpy as np
from scipy.stats import multivariate_normal
import multiprocessing
from joblib import Parallel, delayed

class kdf(KernelDensityGraph):

    def __init__(self, kwargs={}):
        super().__init__()
        self.polytope_means = {}
        self.polytope_vars = {}
        self.polytope_cardinality = {}
        self.polytope_mean_cov = {}
        self.gaussian_dist = {}
        self.kwargs = kwargs

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

        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_vars[label] = []
            self.polytope_cardinality[label] = []
            self.polytope_mean_cov[label] = []

        predicted_leaf_ids_across_trees = [tree.apply(X) for tree in self.rf_model.estimators_]

        for polytopes_in_a_tree in predicted_leaf_ids_across_trees:
            for polytope in np.unique(polytopes_in_a_tree):
                for label in self.labels:
                    polytope_label_idx = np.where((y==label) & (polytopes_in_a_tree==polytope))[0]
                    
                    if polytope_label_idx.size == 0:
                        continue
                    
                    self.polytope_means[label].append(
                        np.mean(
                            X[polytope_label_idx],
                            axis=0
                        )
                    )
                    self.polytope_vars[label].append(
                        np.var(
                            X[polytope_label_idx],
                            axis=0
                        )
                    )
                    self.polytope_cardinality[label].append(
                        len(polytope_label_idx)
                    )

        for label in self.labels:
            self.polytope_mean_cov = np.average(
                self.polytope_vars[label],
                weights = self.polytope_cardinality[label],
                axis = 0
                )
        
            self.gaussian_dist[label] = multivariate_normal(
                                mean=np.zeros(X.shape[1], dtype=float), 
                                cov=self.polytope_mean_cov, 
                                allow_singular=True
                                )

    def _compute_pdf(self, X, label, worker_id, total_polytopes):
        last_idx = len(self.polytope_cardinality[label])
        last_polytope_to_operate = int((worker_id+1)*total_polytopes)
        polytope_ids = range(int(worker_id*total_polytopes), last_polytope_to_operate) if last_idx>last_polytope_to_operate else range(int(worker_id*total_polytopes), last_idx) 
        polytope_cardinality = self.polytope_cardinality[label]

        likelihood = 0.0
        for idx in polytope_ids:
            X_ = X - self.polytope_means[label][idx]
            likelihood += self.gaussian_dist[label].pdf(X_)*polytope_cardinality[idx]/np.sum(polytope_cardinality)
        
        return likelihood

    def predict_proba(self, X, n_jobs=-1):
        r"""
        Calculate posteriors using the kernel density forest.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        n_jobs : int, default=-1 (all cores)
            The number of jobs to run in parallel.
        """
        X = check_array(X)

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        likelihoods = np.zeros(
            (np.size(X,0), len(self.labels)),
            dtype=float
        )
        
        for ii,label in enumerate(self.labels):
            total_polytopes = len(self.polytope_cardinality[label])
            polytopes_per_worker = np.ceil(total_polytopes/n_jobs)
            worker_in_action = int(total_polytopes/polytopes_per_worker)
            
            likelihood_ = np.array(
                Parallel(n_jobs=worker_in_action)(
                    delayed(self._compute_pdf)(
                        X,
                        label,
                        worker_id,
                        polytopes_per_worker
                        ) for worker_id in range(worker_in_action)
                    )
                )
            
            likelihoods[:,ii] += np.mean(likelihood_)
            
        proba = (likelihoods.T/np.sum(likelihoods,axis=1)).T
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
