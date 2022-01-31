from sklearn.covariance import log_likelihood
from .kdf import kdf
import numpy as np

def update_kdf(X_target, y_target, source_model, target_model):
    
    for label in target_model.labels:
        X_ = X_target[np.where(y_target==label)[0]]

        for label_source in source_model.labels:
            total_polytopes = len(source_model.polytope_means[label_source])
            for ii in range(total_polytopes):
                likelihoods = source_model._compute_pdf(X_,label_source,ii)
                log_likelihoods = np.sum(np.log(likelihoods))/X_.shape[0]

                if log_likelihoods >= target_model.min_likelihood and log_likelihoods <= target_model.max_likelihood:
                    target_model.polytope_means[label].append(
                        source_model.polytope_means[label_source][ii]
                    )
                    target_model.polytope_cov[label].append(
                        source_model.polytope_cov[label_source][ii]
                    )

    return target_model