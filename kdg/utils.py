import numpy as np 
from sklearn.datasets import make_blobs
from numpy.random import uniform, normal

def get_ece(predicted_posterior, predicted_label, true_label, num_bins=40):
    poba_hist = []
    accuracy_hist = []
    bin_size = 1/num_bins
    total_sample = len(true_label)
    posteriors = predicted_posterior.max(axis=1)

    score = 0
    for bin in range(num_bins):
        indx = np.where(
            (posteriors>bin*bin_size) & (posteriors<=(bin+1)*bin_size)
        )[0]
        
        acc = np.nan_to_num(
            np.mean(
            predicted_label[indx] == true_label[indx]
        )
        ) if indx.size!=0 else 0
        conf = np.nan_to_num(
            np.mean(
            posteriors[indx]
        )
        ) if indx.size!=0 else 0
        score += len(indx)*np.abs(
            acc - conf
        )
    
    score /= total_sample
    return score

def hellinger(p, q):
    """Hellinger distance between two discrete distributions.
       Same as original version but without list comprehension
    """
    return np.mean(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis = 1)) / np.sqrt(2))

def _generate_2d_rotation(theta=0):
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    return R


def generate_gaussian_parity(
    n_samples,
    centers=None,
    class_label=None,
    cluster_std=0.25,
    center_box=(-1.0,1.0),
    angle_params=None,
    random_state=None,
):
    """
    Generate 2-dimensional Gaussian XOR distribution.
    (Classic XOR problem but each point is the
    center of a Gaussian blob distribution)
    Parameters
    ----------
    n_samples : int
        Total number of points divided among the four
        clusters with equal probability.
    centers : array of shape [n_centers,2], optional (default=None)
        The coordinates of the ceneter of total n_centers blobs.
    class_label : array of shape [n_centers], optional (default=None)
        class label for each blob.
    cluster_std : float, optional (default=1)
        The standard deviation of the blobs.
    center_box : tuple of float (min, max), default=(-1.0, 1.0)
        The bounding box for each cluster center when centers are generated at random.
    angle_params: float, optional (default=None)
        Number of radians to rotate the distribution by.
    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """

    if random_state != None:
        np.random.seed(random_state)

    if centers == None:
        centers = np.array([(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)])

    if class_label == None:
        class_label = [0, 1, 1, 0]

    blob_num = len(class_label)

    # get the number of samples in each blob with equal probability
    samples_per_blob = np.random.multinomial(
        n_samples, 1 / blob_num * np.ones(blob_num)
    )

    X, y = make_blobs(
        n_samples=samples_per_blob,
        n_features=2,
        centers=centers,
        cluster_std=cluster_std,
        center_box=center_box
    )

    for blob in range(blob_num):
        y[np.where(y == blob)] = class_label[blob]

    if angle_params != None:
        R = _generate_2d_rotation(angle_params)
        X = X @ R

    return X, y.astype(int)

def pdf(x, cov_scale=0.25):
    mu01 = np.array([-0.5,0.5])
    mu02 = np.array([0.5,-0.5])
    mu11 = np.array([0.5,0.5])
    mu12 = np.array([-0.5,-0.5])
    cov = cov_scale* np.eye(2)
    inv_cov = np.linalg.inv(cov) 

    p0 = (
        np.exp(-(x - mu01)@inv_cov@(x-mu01).T) 
        + np.exp(-(x - mu02)@inv_cov@(x-mu02).T)
    )/(2*np.pi*np.sqrt(np.linalg.det(cov)))

    p1 = (
        np.exp(-(x - mu11)@inv_cov@(x-mu11).T) 
        + np.exp(-(x - mu12)@inv_cov@(x-mu12).T)
    )/(2*np.pi*np.sqrt(np.linalg.det(cov)))

    return p0/(p0+p1)

def sparse_parity(
    n_samples,
    p_star = 3,
    p = 20
):
    X = np.random.uniform(low=-1,high=1,size=(n_samples,p))
    y = np.sum(X[:,:p_star]>0, axis=1)%2

    return X, y.astype(int)

def gaussian_sparse_parity(
    n_samples,
    centers=None,
    class_label=None,
    p_star = 3,
    p = 20,
    cluster_std = 0.25,
    center_box=(-1.0,1.0),
    random_state = None
):
    if random_state != None:
        np.random.seed(random_state)

    if centers == None:
        if p_star == 2:
            centers = np.array([(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)])
        else:
            centers = np.array(
                [(0.5, 0.5, 0.5), (-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, -0.5)]
                )

    if class_label == None:
        class_label = 1 - np.sum(centers[:,:p_star]>0, axis=1)%2
    
    blob_num = len(class_label)

    # get the number of samples in each blob with equal probability
    samples_per_blob = np.random.multinomial(
        n_samples, 1 / blob_num * np.ones(blob_num)
    )
    
    X, y = make_blobs(
        n_samples=samples_per_blob,
        n_features=p_star,
        centers=centers,
        cluster_std=cluster_std,
        center_box=center_box
    )

    for blob in range(blob_num):
        y[np.where(y == blob)] = class_label[blob]

    if p > p_star:
        X_noise = np.random.uniform(low=center_box[0],high=center_box[1],size=(n_samples,p-p_star))
        X = np.concatenate((X, X_noise), axis=1)

    return X, y.astype(int)

def trunk_sim(
    n_samples,
    dim = 1
):
    samples_per_class = np.random.multinomial(
        n_samples, 1 / 2 * np.ones(2)
    )

    mean = 1./np.sqrt(np.arange(1,dim+1,1))

    X = np.concatenate(
        (
            np.random.multivariate_normal(mean, np.eye(dim), size=samples_per_class[0]),
            np.random.multivariate_normal(-mean, np.eye(dim), size=samples_per_class[1])
        ),
        axis=0
    )
    y = np.concatenate(
        (
            np.zeros(samples_per_class[0]),
            np.ones(samples_per_class[1])
        ),
        axis=0
    )

    return X, y.astype(int)