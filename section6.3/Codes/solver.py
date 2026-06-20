"""solver.py
Inflated Dynamic Laplacian for CMC flow """

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs, LinearOperator
from diffusion_maps import mean_nndist_sq, temp_laplace
from spacetime import make_spacetime_diffusion_mat_productform, matvec_Pa


class InflatedDynamicLaplacian:
    """ Inflated Dynamic Laplacian solver """

    def __init__(self, epsilonx:  float = None, a_factor:  float = 4.0, dirichlet: bool  = False,
                 num_evals: int   = 40, verbose:   bool  = True, ):
        self.epsilonx  = epsilonx
        self.a_factor  = a_factor
        self.dirichlet = dirichlet
        self.num_evals = num_evals
        self.verbose   = verbose
        self.SpacePointsarray = None
        self.TimePoints       = None
        self.Pthalf_interval  = None
        self.Px_dir           = None
        self.Px               = None
        self.Lt_interval      = None
        self.evals_s          = None
        self.evecs            = None
        self.evecs_3d         = None
        self.dynmodes         = None
        self.tempmodes        = None
        self.slicemeans       = None
        self.var_slicemeans   = None
        self.L2norms          = None
        self.a                = None
        self.a_min            = None
        self.epsilonx_        = None
        self.epsilont_        = None

    # IDL fitting
    def fit(self, pts: np.ndarray, TimePoints: np.ndarray, Ix: tuple = (0, 2),Iy: tuple = (0, 1), ):
        """Fit the inflated dynamic Laplacian to trajectory data """
        TimePoints = np.asarray(TimePoints).ravel()
        _, N, T    = pts.shape
        SpacePointsarray      = pts.copy()
        self.SpacePointsarray = SpacePointsarray
        self.TimePoints       = TimePoints
        if self.verbose:
            print(f'  SpacePointsarray: (2, {N}, {T})')
        # epsilon selection 
        if self.epsilonx is None:
            self.epsilonx_ = 0.032   
            if self.verbose:
                print(f'  Using epsilon_x = {self.epsilonx_}')
        else:
            self.epsilonx_ = float(self.epsilonx)
        epsilonx = self.epsilonx_
        # temporal epsilon
        dt = float(np.max(np.diff(TimePoints)))
        self.epsilont_ = 2.0 * dt**2
        tau = float(TimePoints[-1] - TimePoints[0])
        # a_min computed
        xlength0 = float(np.max(SpacePointsarray[0, :, 0]) - np.min(SpacePointsarray[0, :, 0]))
        ylength0 = float(np.max(SpacePointsarray[1, :, 0]) - np.min(SpacePointsarray[1, :, 0]))
        if self.dirichlet:
            a_min = tau * np.sqrt(1.0 / xlength0**2 + 1.0 / ylength0**2)
        else:
            a_min = tau / max(xlength0, ylength0)
        self.a_min = a_min
        self.a     = self.a_factor * a_min
        if self.verbose:
            print(f'  a_min = {a_min:.4f}   a = {self.a:.4f}   ' )
        # Build P(x) and P(t)
        if self.verbose:
            print(' Building spatial diffusion maps P^(x) ')
        (Pthalf, Px, Pthalf_interval, Lt_interval) = make_spacetime_diffusion_mat_productform(
             SpacePointsarray, TimePoints, epsilonx, self.epsilont_, self.a ** 2,)
        self.Pthalf_interval = Pthalf_interval
        self.Px              = Px
        self.Lt_interval     = Lt_interval
        Px_dir = Px.copy()
        # Optional Dirichlet BC
        if self.dirichlet:
            if self.verbose:
                print('  Applying Dirichlet BC ')
            b_globind = self._compute_dirichlet_indices(SpacePointsarray, N, T)
            Px_dir    = Px_dir.tolil()
            Px_dir[b_globind, :] = 0
            Px_dir[:, b_globind] = 0
            Px_dir = Px_dir.tocsr()
        self.Px_dir = Px_dir
        # Eigenpairs via power iteration 
        if self.verbose:
            print(f'  Computing top {self.num_evals} eigenpairs ')
        def _matvec(x):
            return matvec_Pa(x, Pthalf_interval, Px_dir, N, T)
        linop = LinearOperator(shape=(N * T, N * T), matvec=_matvec, dtype=float,)
        raw_evals, raw_evecs = eigs(linop, k=self.num_evals, which='LM', tol=0, maxiter=10_000,)
        # Sort descending by real part 
        order     = np.argsort(np.real(raw_evals))[::-1]
        raw_evals = raw_evals[order]
        raw_evecs = raw_evecs[:, order]
        # Handle complex conjugate pairs i.e., replace (u+iv, u-iv) with (u, v)
        for k in range(self.num_evals):
            if not np.isreal(raw_evals[k]):
                u = np.real(raw_evecs[:, k])
                v = np.imag(raw_evecs[:, k])
                for q in range(k, self.num_evals):
                    if (np.abs(raw_evals[k] - np.conj(raw_evals[q])) < 1e-14
                            and not np.isreal(raw_evals[q])):
                        raw_evecs[:, k] = u
                        raw_evecs[:, q] = v
                        break
        self.evals_s = np.real(raw_evals)
        evecs_real   = np.real(raw_evecs)
        # Normalise: multiply by sqrt(N*T)  
        evecs_real *= np.sqrt(N * T)
        self.evecs  = evecs_real
        # Reshape to (N, T, K) 
        self.evecs_3d = evecs_real.reshape(N, T, self.num_evals, order='F')
        # Mode classification
        self._classify_modes(T)
        if self.verbose:
            print(f'  dynmodes = {len(self.dynmodes)}  ' f'tempmodes = {len(self.tempmodes)}')
            lam = self.laplacian_eigenvalues()
            print(f'  Top-10 nu_k: {lam[:10]}')
        return self

    # Mode classification 
    def _classify_modes(self, T: int):
        """ Classify eigenvectors as temporal or spatial """
        if self.dirichlet:
            K = self.evecs_3d.shape[2]
            self.dynmodes  = np.arange(K)
            self.tempmodes = np.array([], dtype=int)
        else:
            # slicemeans: mean over N spatial points  shape (T, K)
            slicemeans     = np.mean(self.evecs_3d, axis=0)   # (T, K)
            var_slicemeans = np.var(slicemeans, axis=0)        # (K,)
            self.slicemeans     = slicemeans
            self.var_slicemeans = var_slicemeans
            self.tempmodes = np.where(var_slicemeans >= 0.1)[0]
            self.dynmodes  = np.where(var_slicemeans <  0.1)[0]
        # L2 norms on each time slice shape (T, K)
        self.L2norms = np.linalg.norm(self.evecs_3d, axis=0)

    # Dirichlet boundary indices 
    def _compute_dirichlet_indices(self, SpacePointsarray, N, T):
        """Optional Dirichlet BC """
        from scipy.spatial import ConvexHull
        boundary_spatial = set()
        for k in range(T):
            pts_k = SpacePointsarray[:, :, k].T    # (N, 2)
            try:
                hull = ConvexHull(pts_k)
                boundary_spatial.update(hull.vertices.tolist())
            except Exception:
                pass
        b_globind = []
        for k in range(T):
            for j in boundary_spatial:
                b_globind.append(k * N + j)
        return np.array(b_globind, dtype=int)

    # Laplacian eigenvalues
    def laplacian_eigenvalues(self) -> np.ndarray:
        nu = np.clip(np.real(self.evals_s), 1e-14, 1.0)
        return np.log(nu) / (self.epsilonx_ / 4.0)

    # Time Averaged spatial operator
    def px_average(self):
        """ Time average of the block diagonal spatial diffusion matrix """
        _, N, T = self.SpacePointsarray.shape
        acc = sparse.csr_matrix((N, N), dtype=float)
        for k in range(T):
            sl  = slice(k * N, (k + 1) * N)
            acc = acc + self.Px_dir[sl, sl]
        return acc / T

    def px_avg_eigs(self, num_evals: int = 40):
        """ Leading eigenpairs of the time averaged spatial operator """
        Px_avg = self.px_average()
        ev, evec = eigs(Px_avg, k=num_evals, which='LM')
        order = np.argsort(np.real(ev))[::-1]
        ev_r  = np.real(ev[order])
        ev_v  = np.real(evec[:, order])
        # Normalise to unit max-abs per column
        for i in range(ev_v.shape[1]):
            mx = np.max(np.abs(ev_v[:, i]))
            if mx > 0:
                ev_v[:, i] /= mx
        return ev_r, ev_v