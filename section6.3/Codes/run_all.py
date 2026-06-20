"""run_all.py
Main Execution of the package and outputs generated and saved in /results/ """

import os, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS, exist_ok=True)
def _rp(name: str) -> str:
    return os.path.join(RESULTS, name)


# Trajectories Generation for CMC 
print('\n Generating Trajectories')
from generate_pts_CMC import pts, Tspan, Ix, Iy, N, T
print(f'  pts shape: {pts.shape}   Tspan: [{Tspan[0]}, ..., {Tspan[-1]}]  T={T}  N={N}')

# Building Inflated Dynamic Laplacian
print('\n Applying Inflated Dynamic Laplacian')
from solver import InflatedDynamicLaplacian
idl = InflatedDynamicLaplacian(
    epsilonx  = 0.032,   # epsilon value
    a_factor  = 4.0,     # a = 4 × a_min  
    dirichlet = False,   # Neumann BCs 
    num_evals = 40,      # num_evals = 40
    verbose   = True, )

t0 = time.time()
idl.fit(pts, Tspan, Ix=Ix, Iy=Iy)
print(f'  IDL fit completed in {time.time()-t0:.1f}s')
print(f'  a = {idl.a:.4f}   eps = {idl.epsilonx_:.6f}')
# Save key data 
np.save(_rp('evals_s.npy'),  idl.evals_s)
np.save(_rp('L2norms.npy'),  idl.L2norms)

# Eigenvalues spectrum
print('\n Eigenvalues spectrum')
lam = idl.laplacian_eigenvalues()
dyn  = np.asarray(idl.dynmodes)
temp = np.asarray(idl.tempmodes)
print(f'  Top 10 Spatial  eigenvalues: {lam[dyn[:10]]}')
if len(temp):
    print(f' Top 5 Temporal eigenvalues:  {lam[temp[:5]]}')

# Dynamic Laplacian Plotting
print('\n Dynamic Laplacian Plots')
from plotting import fig18_dynlap, fig19_dynlap_seba
from seba import seba
fig18_dynlap(idl, outpath=_rp('fig18_dynlap.png'))
fig19_dynlap_seba(idl, seba, outpath=_rp('fig19_dynlap_seba.png'))

# Plotting IDL eigenvectors in 3D
print('\n IDL spatial eigenvectors in 3D Plot')
from plotting import fig20_idl_3d
fig20_idl_3d(idl, abs_cutoff=0.25, outpath=_rp('fig20_idl_3d.png'))

# Plotting Time Slices
print('\n Time Slices Plot ')
from plotting import fig21_timeslices
fig21_timeslices(idl, abs_cutoff=0.25, outpath=_rp('fig21_timeslices.png'))

# SEBA Augmentation and  Application
print('\n SEBA')
_, N_p, T_p = idl.SpacePointsarray.shape
dyn = np.asarray(idl.dynmodes)

# Take The 3 Nontrivial Spatial Modes
spatial_idx = dyn[1:4] if len(dyn) >= 4 else dyn[:3]             #  Skip trivial mode (dyn[0])
n_spatial   = len(spatial_idx)
print(f'  Using {n_spatial} spatial eigenvectors: indices {spatial_idx}')

# Column matrix of spatial eigenvectors  (N·T, n_spatial)
V_spatial = idl.evecs[:, spatial_idx]

# Augmented vectors
V_aug = np.zeros_like(V_spatial)
for ci in range(n_spatial):
    col = spatial_idx[ci]
    for ti in range(T_p):
        sl  = slice(ti * N_p, (ti + 1) * N_p)   # timeslice ti
        nrm = np.linalg.norm(idl.evecs[sl, col])
        V_aug[sl, ci] = nrm

# Input to SEBA
V_full = np.hstack([V_spatial, V_aug])
t0 = time.time()
S_seba, R_seba, I_seba = seba(V_full)
print(f'  SEBA completed in {time.time()-t0:.1f}s')
np.save(_rp('seba_vectors.npy'), S_seba)

# Step 9 — Plotting SEBA and l2 Norm
print('\n SEBA and l2 Norm Plots')
from plotting import fig22_l2norms, fig23_seba
fig22_l2norms(idl, S_seba, outpath=_rp('fig22_l2norms.png'))
fig23_seba(idl, S_seba, abs_cutoff=0.25, outpath=_rp('fig23_seba.png'))

# Plotting Animations
print('\n Animations Generation')
from plotting import (anim_eigenvector, anim_seba_sets, anim_regime_sweep,)
print(' Spatial Mode 1 Animation ')
anim_eigenvector(idl, mode_offset=1, save=_rp('SpatMode_1_CMC.gif'))
print(' Spatial Mode 2 Animation')
anim_eigenvector(idl, mode_offset=2, save=_rp('SpatMode_2_CMC.gif'))
print(' SEBA Sets Animation')
anim_seba_sets(idl, S_seba, save=_rp('SEBA_sets_CMC.gif'))
print(' Animating Regime Sweep ')
anim_regime_sweep(idl, save=_rp('RegimeSweep_CMC.gif'))
# After Completion without any errors
print('\n All Work Done! ')