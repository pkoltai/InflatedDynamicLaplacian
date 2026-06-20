"""diffusion_maps.py
Kernel Computation and Temporal Laplacian """

import numpy as np
from scipy import sparse
from scipy.spatial import KDTree

# Nearest Neighbour Distance Computing Function
def nndist(A: np.ndarray) -> np.ndarray:
    """ Return the nearest neighbour distance for each row of A """
    tree  = KDTree(A)
    dists, _ = tree.query(A, k=2)           # k=2: first hit is self (dist=0)
    return dists[:, 1]
# Mean Squared Nearest Neighbour Distance
def mean_nndist_sq(pts_slice: np.ndarray) -> float:
    """ Mean squared nearest-neighbour distance for a (2, N) slice."""
    nn = nndist(pts_slice.T)        # transpose to (N, 2)
    return float(np.mean(nn) ** 2)


# Core Diffusion Maps Block 
def diffusion_maps_matrix(pts: np.ndarray, epsilon: float):
    """ Build the N*N row-stochastic diffusion maps matrix """
    m      = pts.shape[1]
    data_T = pts.T                    # (N, d)
    # Cutoff Radius
    r    = np.sqrt(5.0 * epsilon)
    tree = KDTree(data_T)
    idx_list = tree.query_ball_point(data_T, r)
    # Pre allocate COO arrays
    lv   = sum(len(idx_list[i]) for i in range(m))
    rows = np.empty(lv, np.int32)
    cols = np.empty(lv, np.int32)
    vals = np.empty(lv, np.float64)
    # Kernel K_ij = exp(−‖xi−xj‖^2 / eps)
    icurr = 0
    for i in range(m):
        nbrs = idx_list[i]
        li   = len(nbrs)
        for jj, j in enumerate(nbrs):
            diff = data_T[i] - data_T[j]
            d2   = float(np.dot(diff, diff))
            rows[icurr + jj] = i
            cols[icurr + jj] = j
            vals[icurr + jj] = np.exp(-d2 / epsilon)
        icurr = (icurr + li)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(m, m), dtype=float)
    diag_A = np.array(A.diagonal())
    A      = A - sparse.diags(diag_A, format='csr') + sparse.eye(m, format='csr')
    # Density Normalisation 
    row_means = np.asarray(A.mean(axis=1)).ravel()
    row_means = np.where(row_means == 0, 1e-14, row_means)
    q         = 1.0 / row_means      # alpha = 1
    Adensnorm = sparse.diags(q, format='csr') @ A @ sparse.diags(q, format='csr')
    row_sums = np.asarray(Adensnorm.sum(axis=1)).ravel()
    row_sums = np.where(row_sums == 0, 1e-14, row_sums)
    DMM      = sparse.diags(1.0 / row_sums, format='csr') @ Adensnorm
    return DMM.tocsr(), A.tocsr()


# Temporal Laplacian
def temp_laplace(Tspan: np.ndarray) -> np.ndarray:
    """ Temporal Laplacian  """
    ts = np.asarray(Tspan).ravel().astype(np.float64)
    n  = len(ts)
    hs = np.diff(ts)         # step sizes 
    L = np.zeros((n, n), dtype=np.float64)
    # First row 
    L[0, 0] = -1.0 / hs[0]**2
    L[0, 1] =  1.0 / hs[0]**2
    # Last row
    L[n-1, n-2] =  1.0 / hs[-1]**2
    L[n-1, n-1] = -1.0 / hs[-1]**2
    # Interior rows
    for i in range(1, n - 1):
        hm = hs[i - 1]
        hp = hs[i]
        L[i, i-1] =  2.0 / (hm * (hm + hp))
        L[i, i  ] = -2.0 / (hm * hp)
        L[i, i+1] =  2.0 / (hp * (hm + hp))
    return L