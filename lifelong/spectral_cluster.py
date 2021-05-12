#%%
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, MiniBatchKMeans
import os

def Cosine_Affinity(embds):
    """
        from Speaker Diarization with LSTM paper
    """
    cosine_similarities = np.matmul(embds, np.transpose(embds))
    affinity = (cosine_similarities + 1.0) / 2.0
    return affinity

def sim_enhancement(S):
    """
        from Speaker Diarization with LSTM paper
    """
    def Cropdiag(S):
        X = np.copy(S)
        np.fill_diagonal(X, 0.0)
        di = np.diag_indices(X.shape[0])
        X[di] = X.max(axis=1)
        return X
    def GaussianFilter(S):
        return gaussian_filter(S, sigma=1.0)
    def row_wise_thresholding(S, percentile = 0.95, multiplier = 0.01):
        row_percentile = np.percentile(S, percentile * 100, axis=1)
        row_percentile = np.expand_dims(row_percentile, axis=1)
        is_smaller = S < row_percentile
        S = (S * np.invert(is_smaller)) + (S * multiplier * is_smaller)
        return S    
    def symertisize(S):
        return np.maximum(S, S.T)
    def diffusion(S):
        return np.dot(S, S.T)
    def row_max_norm(S):
        Y = np.copy(S)
        row_max = Y.max(axis=1)
        Y /= np.expand_dims(row_max, axis=1)
        return Y

    S = Cropdiag(S)
    S = GaussianFilter(S)
    S = row_wise_thresholding(S)
    S = symertisize(S)
    S = diffusion(S)
    S = row_max_norm(S)

    return S

def MySpectralClustering(embds, mask_t = 0.01, Kmiter = 100):
    """
        from Speaker Diarization with LSTM paper
        embds: matrix of shape TxD
        returns: a list of labels
    """
    S = Cosine_Affinity(embds)
    S = sim_enhancement(S) # their replacement for laplacian
    eigvals, eigvecs = np.linalg.eig(S)
    eigvals = eigvals.real
    eigvecs = eigvecs.real

    # to sort in descending order as required by paper
    sorted_indice = np.argsort(-eigvals)
    eigvals = eigvals[sorted_indice]
    eigvecs = eigvecs[:, sorted_indice]

    # eigen values smaller than mask_t are ignored
    max_delta = 0
    max_delta_index = 0
    for i in range(1, len(eigvals)):
        if eigvals[i - 1] < mask_t:
            break
        delta = eigvals[i - 1] / eigvals[i]
        if delta > max_delta:
            max_delta = delta
            max_delta_index = i
    
    print(f"number of speakers detected: {max_delta_index}")
    km = MiniBatchKMeans(n_clusters=max_delta_index, max_iter= Kmiter)
    P = eigvecs[:, :max_delta_index]
    return km.fit_predict(P)    
