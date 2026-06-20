"""epsilon_selection.py
Optimal bandwidth ε via the log-log slope heuristic """

import time
import warnings
import numpy as np
from scipy.spatial.distance import pdist

# Core S(epsilon) computation 
def _compute_S(X: np.ndarray, arr_eps: np.ndarray, eps_max: float) -> np.ndarray:
    N     = X.shape[0]
    r_cut = np.sqrt(5.0 * eps_max)
    d_all = pdist(X)                    # N(N-1)/2 pairwise distances
    d_cut = d_all[d_all <= r_cut]
    if len(d_cut) == 0:
        return np.full(len(arr_eps), 1.0 / N)
    # Off-diagonal kernel sum for each epsilon
    K_off = np.exp(-d_cut[None, :]**2 / arr_eps[:, None]).sum(axis=1)
    # Diagonal contributes N ones
    return (N + 2.0 * K_off) / N**2

# Heuristic selection 
def compute_epsilon(X: np.ndarray, eps_min: float = 1e-4, eps_max: float = 1.0,
                    n_eps:int = 40, verbose: bool  = True, ) -> tuple:
    """ Selecting epsilon """
    arr_eps = np.exp(np.linspace(np.log(eps_min), np.log(eps_max), n_eps))
    S       = _compute_S(X, arr_eps, eps_max)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        slope = (np.log(S[1:] / S[:-1]) / np.log(arr_eps[1:] / arr_eps[:-1]))
    imdS    = int(np.argmax(slope))
    eps_opt = float(arr_eps[imdS])
    dim_est = 2.0 * float(slope[imdS])
    if verbose:
        print(f'  epsilon = {eps_opt:.6f}   dim_est = {dim_est:.2f}')
    return eps_opt, dim_est, arr_eps, S, slope


# Main
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    t0 = time.time()
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from generate_pts_CMC import pts
    paper_val = 0.032
    X = pts[:, :, 0].T.astype(np.float64)
    print(f'epsilon_selection.py ')
    eps_opt, dim_est, arr_eps, S, slope = compute_epsilon(X, eps_min=1e-4, eps_max=1.0, 
                                                          n_eps=40, verbose=True,)
    print(f'  Computed       : {eps_opt:.6f}')
    np.save('epsilon_opt.npy', np.float64(eps_opt))