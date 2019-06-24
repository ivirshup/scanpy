import numpy as np
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs
from sklearn.preprocessing import StandardScaler


def _get_mean_var(X):
    X = X.astype(np.float64, copy=False)  # Easy to get inaccuracies otherwise
    if issparse(X):
        m, v = sparsefuncs.mean_variance_axis(X, axis=0)
        v *= (X.shape[0]/(X.shape[0]-1))
    else:
        scaler = StandardScaler(with_mean=False).partial_fit(X)
        m = scaler.mean_
        v = scaler.var_ * (X.shape[0]/(X.shape[0]-1))
    return m, v