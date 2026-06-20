"""seba.py
Sparse Eigenbasis Approximation """

import numpy as np
from scipy.linalg import svd, qr


# SEBA Vectors Computation
def seba(V: np.ndarray, Rinit: np.ndarray = None, tol: float = 1e-12) -> tuple:
    """ Compute SEBA vectors """
    # Orthonormalise columns of V 
    V, _ = qr(V, mode='economic')
    p, r = V.shape
    # Soft-threshold parameter  μ = 0.99 / sqrt(p)
    mu = 0.99 / np.sqrt(p)
    S = np.zeros((p, r), dtype=np.float64)
    # Initialise rotation 
    if Rinit is None:
        Rnew = np.eye(r, dtype=np.float64)
    else:
        U2, _, Vt2 = svd(Rinit, full_matrices=False)
        Rnew = U2 @ Vt2
        if np.linalg.det(Rnew) < 0:
            Rnew[:, 0] *= -1
    R = np.zeros_like(Rnew)
    iters = 0
    while np.linalg.norm(Rnew - R, 2) > tol:
        R = Rnew.copy()
        # Z = V*R^T   (p, r)
        Z = V @ R.T
        # Soft-thresholding + normalise to unit Euclidean norm
        for i in range(r):
            s   = np.sign(Z[:, i]) * np.maximum(np.abs(Z[:, i]) - mu, 0.0)
            nrm = np.linalg.norm(s)
            S[:, i] = s / nrm if nrm > 0 else s
        # Polar decomposition of  M = S^T*V 
        M          = S.T @ V           # (r, r)
        U2, _, Vt2 = svd(M, full_matrices=False)
        Rnew       = U2 @ Vt2
        if np.linalg.det(Rnew) < 0:
            Rnew[:, 0] *= -1
        iters += 1
        if iters > 100_000:
            import warnings
            warnings.warn('SEBA: reached 100000 iterations without convergence.')
            break

    # Post-processing of sign correction and normalise by column maximum
    for i in range(r):
        S[:, i] *= np.sign(np.sum(S[:, i]))
        mx = np.max(S[:, i])
        if mx > 0:
            S[:, i] /= mx
    # Sort columns by min(S) descending 
    I = np.argsort(np.min(S, axis=0))[::-1]
    S = S[:, I]
    return S, R, I