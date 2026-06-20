"""generate_pts_CMC.py 
Generate particle trajectories for CMC double gyre """

import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import savemat, loadmat

# Parameters
A_CMC     = 0.25
OMEGA_CMC = np.pi
Ix = [0.0, 2.0]
Iy = [0.0, 1.0]
# Grid parameters
mx, my = 40, 20
T0, TF = 0.0, 15.0
dt     = 0.2
Tspan  = np.arange(T0, TF + 0.5 * dt, dt)     # 76 pts
T      = len(Tspan)                           # 76
N      = mx * my                              # 800
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'Pts_CMC_{mx}x{my}x{T}.mat')

def _delta(t: float) -> float:
    """ delta=0.25 if 5 <= t <= 10, else 0."""
    return 0.25 if (5.0 <= t <= 10.0) else 0.0

def cmc_rhs(t: float, x) -> np.ndarray:
    """ RHS of the CMC double gyre """
    x1, x2 = float(x[0]), float(x[1])
    d   = _delta(t)
    sw  = np.sin(OMEGA_CMC * t)
    f   = d * sw * x1**2 + (1.0 - 2.0 * d * sw) * x1
    fx1 = 2.0 * d * sw * x1 + (1.0 - 2.0 * d * sw)
    u = -np.pi * A_CMC * np.sin(np.pi * f)  * np.cos(np.pi * x2)
    v =  np.pi * A_CMC * np.cos(np.pi * x1) * np.sin(np.pi * x2) * fx1   
    return np.array([u, v], dtype=np.float64)

# Generate or Load Trajectories
if os.path.exists(CACHE):
    _d    = loadmat(CACHE)
    pts   = _d['pts']                         # (2, 800, 76)
    Tspan = _d['Tspan'].ravel()               # (76,)
else:
    print(f'Generating {N} trajectories * {T} time steps ...')
    t_start = time.time()
    # Initial grid 
    _e  = 1e-3
    xg  = np.linspace(Ix[0] + _e * (Ix[1] - Ix[0]), Ix[1] - _e * (Ix[1] - Ix[0]), mx)
    yg  = np.linspace(Iy[0] + _e * (Iy[1] - Iy[0]), Iy[1] - _e * (Iy[1] - Iy[0]), my)
    Xg, Yg = np.meshgrid(xg, yg)                    # each (20, 40)
    x0 = Xg.ravel()                                 # (800,)
    y0 = Yg.ravel()                                 # (800,)
    pts = np.empty((2, N, T), dtype=np.float64)
    for i in range(N):
        sol = solve_ivp(cmc_rhs, [T0, TF], [x0[i], y0[i]], t_eval=Tspan, method='RK45', 
                        rtol=1e-9, atol=1e-12, )
        pts[:, i, :] = sol.y   # (2, T)
    savemat(CACHE, {'pts': pts, 'Tspan': Tspan, 'mx': mx, 'my': my})
    print(f'Saved {CACHE}  ({time.time()-t_start:.1f}s)')