"""spacetime.py
Strang Splitting Spacetime Diffusion Operator """

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from diffusion_maps import diffusion_maps_matrix, temp_laplace

# Building Spacetime Operator factors
def make_spacetime_diffusion_mat_productform(SpacePointsarray: np.ndarray, TimePoints: np.ndarray,
                                              epsilonx: float, epsilont: float, 
                                              t_factor: float,) -> tuple:
    """ Build the factors of the Strang-split spacetime operator """
    _, N, T = SpacePointsarray.shape
    # Spatial diffusion blocks 
    Px_blocks = []
    for k in range(T):
        Px_slice, _ = diffusion_maps_matrix(SpacePointsarray[:, :, k], epsilonx)
        Px_blocks.append(Px_slice)
    Px = sparse.block_diag(Px_blocks, format='csr')    # (N·T, N·T)
    # Temporal Laplacian 
    Lt_interval = temp_laplace(TimePoints)             # (T, T)

    # Temporal half step matrix 
    exponent        = t_factor * epsilonx / 8.0        
    Pthalf_interval = expm(Lt_interval * exponent)     # (T, T)  dense
    # Kronecker product
    Pthalf = sparse.kron(sparse.csr_matrix(Pthalf_interval), sparse.eye(N, format='csr'),
                         format='csr',)
    return Pthalf, Px, Pthalf_interval, Lt_interval

# Strang Splitting Computation
def matvec_Pa(x: np.ndarray, Pthalf_interval: np.ndarray, Px_dir: sparse.csr_matrix,
              N: int, T: int,) -> np.ndarray:
    def _apply_Pthalf(v: np.ndarray) -> np.ndarray:
        # v : (N·T,)  to  reshape (T, N)  to  Pthalf @ X  to ravel
        V = v.reshape(T, N)
        return (Pthalf_interval @ V).ravel()
    # Strang Splitting
    y = _apply_Pthalf(x)    # first temporal half-step
    y = Px_dir @ y           # spatial step
    y = _apply_Pthalf(y)    # second temporal half-step
    return np.real(y)