import numpy as np 

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
