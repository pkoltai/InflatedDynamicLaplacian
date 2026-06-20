"""plotting.py
All figures and Animations for Coherent Mixing Coherent Flow """

import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
warnings.filterwarnings('ignore')

# Colormap 
def _bluewhitered(m=256):
    anchors = np.array([
        [0.0, 0.0, 0.5],   # dark blue
        [0.0, 0.5, 1.0],   # sky blue
        [1.0, 1.0, 1.0],   # white
        [1.0, 0.0, 0.0],   # red
        [0.5, 0.0, 0.0],   # dark red
    ])
    steps = np.linspace(0, 1, 5)
    out   = np.zeros((m, 3))
    t     = np.linspace(0, 1, m)
    for i in range(3):
        out[:, i] = np.clip(np.interp(t, steps, anchors[:, i]), 0, 1)
    return LinearSegmentedColormap.from_list('bluewhitered', out, N=m)
BWR  = _bluewhitered(256)
SRED = LinearSegmentedColormap.from_list(
    'sred', [(1., 1., 1.), (1., 0., 0.), (0.5, 0., 0.)], N=256)
def _bwr_norm():
    return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

# Sign convention helpers 
def _norm_to_pm1(v):
    mx = np.max(np.abs(v))
    return v / mx if mx > 1e-15 else v.copy()

def _prep_idl_evec(evecs_3d, col, SPA):
    """ Preparing an IDL eigenvector for plotting  """
    v = -evecs_3d[:, :, col].astype(float)
    v = _norm_to_pm1(v)
    left = SPA[0, :, 0] < 1.0
    if left.any() and np.mean(v[left, 0]) < 0:
        v = -v
    return v

def _prep_avg_evec_lr(v_raw, SPA):
    """Preparing Px_avg eigenvector that separates gyres  """
    v = _norm_to_pm1(np.real(v_raw).copy())
    left = SPA[0, :, 0] < 1.0
    if left.any() and np.mean(v[left]) < 0:
        v = -v
    return v

def _prep_avg_evec_cores(v_raw, SPA):
    """ Prepare Px_avg eigenvector that shows gyre cores """
    v = _norm_to_pm1(np.real(v_raw).copy())
    x1, x2 = SPA[0, :, 0], SPA[1, :, 0]
    core_l = (x1 > 0.25) & (x1 < 0.75) & (x2 > 0.25) & (x2 < 0.75)
    core_r = (x1 > 1.25) & (x1 < 1.75) & (x2 > 0.25) & (x2 < 0.75)
    mean_cores = np.mean(v[core_l | core_r]) if (core_l | core_r).any() else np.mean(v)
    if mean_cores < 0:
        v = -v
    return v

# Axis and layout helpers
def _set_3d_axes(ax, tmax=15):
    ax.set_xlabel(r'$t$',   fontsize=9, labelpad=2)
    ax.set_ylabel(r'$x_1$', fontsize=9, labelpad=2)
    ax.set_zlabel(r'$x_2$', fontsize=9, labelpad=2)
    ax.set_xlim(0, tmax); ax.set_ylim(0, 2); ax.set_zlim(0, 1)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=25, azim=225)

# Spatial Modes or Non trivial Spatial Modes
def _spatial_modes(idl):
    """ dyn[1], dyn[2], dyn[3] which are the 3 nontrivial spatial IDL modes """
    dyn = np.asarray(idl.dynmodes)
    return dyn[1:4] if len(dyn) >= 4 else dyn[:3]

# Saving Animations
def _save_ani(ani, path, fps=10):
    try:
        from matplotlib.animation import FFMpegWriter
        ani.save(str(path), writer=FFMpegWriter(fps=fps), dpi=100)
    except Exception:
        gp = str(path).replace('.mp4', '.gif')
        ani.save(gp, writer=PillowWriter(fps=fps), dpi=100)
        path = gp
    print(f'  Saved {path}')
    return path


# ── Dynamic Laplacian
def fig18_dynlap(idl, outpath=None):
    Tspan   = idl.TimePoints
    SPA     = idl.SpacePointsarray
    _, N, T = SPA.shape
    _, avg_ev = idl.px_avg_eigs(num_evals=40)
    # Prepare eigenvectors with correct signs
    fld1 = _prep_avg_evec_lr(avg_ev[:, 1], SPA)     # mode 1: left-right
    fld2 = _prep_avg_evec_cores(avg_ev[:, 2], SPA)  # mode 2: cores
    sm = plt.cm.ScalarMappable(cmap=BWR, norm=_bwr_norm())
    sm.set_array([])
    fig = plt.figure(figsize=(11, 5))
    for ci, (fld, title) in enumerate([(fld1, 'DynLap eigenfunction 2'),
                                       (fld2, 'DynLap eigenfunction 3')]):
        ax = fig.add_subplot(1, 2, ci + 1, projection='3d')
        for ti in range(T):
            mask = np.abs(fld) > 0.25
            if not mask.any():
                continue
            ax.scatter(Tspan[ti] * np.ones(mask.sum()), SPA[0, mask, ti], SPA[1, mask, ti],
                c=sm.to_rgba(fld[mask]), s=4, depthshade=False, rasterized=True, )
        _set_3d_axes(ax)
        ax.set_title(title, fontsize=10)
    fig.colorbar(sm, ax=fig.axes, shrink=0.5, pad=0.03,
                 label='eigenfunction value  (|·|<0.25 cut)')
    plt.suptitle(' Dynamic Laplacian eigenfunctions (M₁)', fontsize=11)
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f'  Saved {outpath}')
    return fig

def fig19_dynlap_seba(idl, seba_fn, outpath=None):
    Tspan   = idl.TimePoints
    SPA     = idl.SpacePointsarray
    _, N, T = SPA.shape
    _, avg_ev = idl.px_avg_eigs(num_evals=40)
    V3 = avg_ev[:, 0:3]
    S, _, _ = seba_fn(V3)
    sm = plt.cm.ScalarMappable(cmap=SRED, norm=Normalize(0, 1))
    sm.set_array([])
    fig = plt.figure(figsize=(14, 5))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        for ti in range(T):
            mask = S[:, i] >= 0.25
            if not mask.any():
                continue
            ax.scatter(Tspan[ti] * np.ones(mask.sum()), SPA[0, mask, ti], SPA[1, mask, ti],
                c=sm.to_rgba(S[mask, i]), s=5, depthshade=False, rasterized=True,)
        _set_3d_axes(ax)
        ax.set_title(f'DynLap SEBA {i + 1}', fontsize=10)
    fig.colorbar(sm, ax=fig.axes, shrink=0.5, pad=0.03, label='SEBA value')
    plt.suptitle('SEBA vectors from DynLap (M₁)', fontsize=11)
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f'  Saved {outpath}')
    return fig

# IDL Spatial Eigenvectors 3D
def fig20_idl_3d(idl, abs_cutoff=0.25, outpath=None):
    """ Leading 3 spatial IDL eigenvectors in M₁ (3D) """
    Tspan   = idl.TimePoints
    SPA     = idl.SpacePointsarray
    _, N, T = SPA.shape
    nt      = _spatial_modes(idl)
    sm = plt.cm.ScalarMappable(cmap=BWR, norm=_bwr_norm())
    sm.set_array([])
    fig = plt.figure(figsize=(5 * len(nt) + 1.5, 5))
    titles = ['Spatial eigvec 1 ', 'Spatial eigvec 2 ', 'Spatial eigvec 3 ']
    for ci, col in enumerate(nt):
        ax = fig.add_subplot(1, len(nt), ci + 1, projection='3d')
        V  = _prep_idl_evec(idl.evecs_3d, col, SPA)
        for ti in range(T):
            fld  = V[:, ti]
            mask = np.abs(fld) > abs_cutoff
            if not mask.any():
                continue
            ax.scatter( Tspan[ti] * np.ones(mask.sum()), SPA[0, mask, ti], SPA[1, mask, ti],
                c=sm.to_rgba(fld[mask]), s=4, depthshade=False, rasterized=True, )
        _set_3d_axes(ax)
        ax.set_title(titles[ci] if ci < len(titles) else f'Spatial eigvec {ci+1}', fontsize=9)
    fig.colorbar(sm, ax=fig.axes, shrink=0.45, pad=0.03,
                 label='eigenfunction value  (|·|<0.25 cut)')
    plt.suptitle(r' Leading 3 IDL spatial eigenfunctions ($\mathbb{M}_1$)', fontsize=11)
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f'  Saved {outpath}')
    return fig

# Time Slices
def fig21_timeslices(idl, abs_cutoff=0.25, outpath=None):
    Tspan   = idl.TimePoints
    SPA     = idl.SpacePointsarray
    _, N, T = SPA.shape
    nt      = _spatial_modes(idl)
    t_targets = [0.4, 4.4, 5.4, 6.6, 7.6, 8.6, 9.6, 10.6, 14.6]
    t_idx     = [int(np.argmin(np.abs(Tspan - t))) for t in t_targets]
    nr, nc = len(t_idx), len(nt)
    fig, axes = plt.subplots(nr, nc, figsize=(3.8 * nc, 2.2 * nr))
    if nr == 1: axes = axes[np.newaxis, :]
    if nc == 1: axes = axes[:, np.newaxis]
    col_titles = ['Spatial eigenfunction 1', 'Spatial eigenfunction 2', 'Spatial eigenfunction 3']
    for ci, col in enumerate(nt):
        V = _prep_idl_evec(idl.evecs_3d, col, SPA)
        for ri, ti in enumerate(t_idx):
            ax  = axes[ri, ci]
            fld = V[:, ti].copy()
            fld[np.abs(fld) < abs_cutoff] = np.nan
            sc = ax.scatter(SPA[0, :, ti], SPA[1, :, ti], c=fld, cmap=BWR, norm=_bwr_norm(),
                s=6, rasterized=True, linewidths=0,)
            ax.set_facecolor('white')
            ax.set_aspect('equal')
            ax.set_xlim(0, 2); ax.set_ylim(0, 1)
            ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=7)
            if ri == 0:
                ax.set_title(col_titles[ci] if ci < len(col_titles) else f'Eigvec {ci+1}',
                             fontsize=10)
            if ci == 0:
                ax.set_ylabel(f't = {Tspan[ti]:.1f}', fontsize=9)
    plt.suptitle(' Time slices of IDL spatial eigenfunctions (M₁)', fontsize=11)
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f'  Saved {outpath}')
    return fig

# L2 Norm
def fig22_l2norms(idl, seba_vectors, outpath=None):
    Tspan   = idl.TimePoints
    _, N, T = idl.SpacePointsarray.shape
    nt      = _spatial_modes(idl)
    n_seba  = seba_vectors.shape[1]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    styles = ['-', '--', ':']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # Left panel — eigenvector L2 norms
    ax = axes[0]
    for ci, col in enumerate(nt):
        norms = np.linalg.norm(idl.evecs_3d[:, :, col], axis=0)
        ax.plot(Tspan, norms, styles[ci], marker='.', ms=5,
                color=colors[ci], lw=1.8,
                label=f'spatial eigenvector {ci + 1}')
    ax.set_xlabel('t', fontsize=13)
    ax.set_ylabel('norm of time-fibre', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 15); ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=10)
    # Right panel — SEBA L² norms
    ax = axes[1]
    S3 = seba_vectors.reshape(N, T, n_seba, order='F')
    for i in range(n_seba):
        norms = np.linalg.norm(S3[:, :, i], axis=0)
        ax.plot(Tspan, norms, marker='.', ms=5,
                color=colors[i % len(colors)],
                lw=1.8, label=f'SEBA vector {i + 1}')
    ax.set_xlabel('t', fontsize=13)
    ax.set_ylabel('norm of time-fibre', fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(0, 15); ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=10)
    plt.suptitle(r' $L^2$ norms ', fontsize=11)
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f'  Saved {outpath}')
    return fig

# SEBA Plot
def fig23_seba(idl, seba_vectors, abs_cutoff=0.25, outpath=None):
    Tspan   = idl.TimePoints
    SPA     = idl.SpacePointsarray
    _, N, T = SPA.shape
    n_seba  = seba_vectors.shape[1]
    S3      = seba_vectors.reshape(N, T, n_seba, order='F')
    sm = plt.cm.ScalarMappable(cmap=SRED, norm=Normalize(0, 1))
    sm.set_array([])
    nc = 3
    nr = int(np.ceil(n_seba / nc))
    fig = plt.figure(figsize=(5.5 * nc, 4.5 * nr))
    for k in range(n_seba):
        ax = fig.add_subplot(nr, nc, k + 1, projection='3d')
        for ti in range(T):
            fld  = S3[:, ti, k]
            mask = fld >= abs_cutoff
            if not mask.any():
                continue
            ax.scatter( Tspan[ti] * np.ones(mask.sum()), SPA[0, mask, ti], SPA[1, mask, ti],
                c=sm.to_rgba(fld[mask]), s=5, depthshade=False, rasterized=True, )
        _set_3d_axes(ax)
        ax.set_title(f'SEBA {k + 1}', fontsize=10)
    for k in range(n_seba, nr * nc):
        fig.add_subplot(nr, nc, k + 1).axis('off')
    fig.colorbar(sm, ax=fig.axes, shrink=0.35, pad=0.03, label='SEBA value')
    plt.suptitle(' SEBA vectors ', fontsize=11)
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f'  Saved {outpath}')
    return fig

# Animations
def anim_eigenvector(idl, mode_offset=1, abs_cutoff=0.25, fps=10, save=None):
    """ Animation of Spatial eigenvectors """
    Tspan   = idl.TimePoints
    SPA     = idl.SpacePointsarray
    _, N, T = SPA.shape
    dyn     = np.asarray(idl.dynmodes)
    col     = dyn[mode_offset] if len(dyn) > mode_offset else dyn[-1]
    V       = _prep_idl_evec(idl.evecs_3d, col, SPA)
    sm = plt.cm.ScalarMappable(cmap=BWR, norm=_bwr_norm())
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    cbar = plt.colorbar(sm, ax=ax, label='eigenfunction value')
    def update(fr):
        ax.cla()
        ax.set_aspect('equal'); ax.set_xlim(0, 2); ax.set_ylim(0, 1)
        ax.set_xlabel(r'$x_1$', fontsize=11)
        ax.set_ylabel(r'$x_2$', fontsize=11)
        t_val  = float(Tspan[fr])
        regime = 'coherent' if (t_val < 5 or t_val > 10) else 'mixing'
        ax.set_title(f'Spatial mode {mode_offset}   t = {t_val:.2f}   [{regime}]',
                     fontsize=11)
        fld = V[:, fr].copy()
        fld[np.abs(fld) < abs_cutoff] = np.nan
        ax.scatter(SPA[0, :, fr], SPA[1, :, fr],
                   c=fld, cmap=BWR, norm=_bwr_norm(), s=8, rasterized=True)
        return (ax,)
    ani  = FuncAnimation(fig, update, frames=T, interval=100, blit=False)
    path = save or f'anim_eigvec_mode{mode_offset}.gif'
    _save_ani(ani, path, fps)
    plt.close()
    return path

def anim_seba_sets(idl, seba_vectors, abs_cutoff=0.25, fps=10, save=None):
    """ Animation of SEBA sets  """
    Tspan   = idl.TimePoints
    SPA     = idl.SpacePointsarray
    _, N, T = SPA.shape
    n_seba  = seba_vectors.shape[1]
    S3      = seba_vectors.reshape(N, T, n_seba, order='F')
    sm = plt.cm.ScalarMappable(cmap=SRED, norm=Normalize(0, 1))
    sm.set_array([])
    nc        = 3
    nr        = int(np.ceil(n_seba / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.5 * nr))
    axes_flat = np.array(axes).ravel()
    plt.colorbar(sm, ax=axes_flat.tolist(), shrink=0.5, label='SEBA value')
    def update(fr):
        t_val  = float(Tspan[fr])
        regime = 'coherent' if (t_val < 5 or t_val > 10) else 'mixing'
        for k, ax in enumerate(axes_flat):
            ax.cla()
            ax.set_aspect('equal'); ax.set_xlim(0, 2); ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)
            ax.set_title(f'SEBA {k + 1}   t={t_val:.1f} [{regime}]', fontsize=9)
            if k < n_seba:
                fld  = S3[:, fr, k]
                mask = fld >= abs_cutoff
                ax.scatter(SPA[0, :, fr], SPA[1, :, fr],
                           c='#e0e0e0', s=5, rasterized=True, zorder=1)
                if mask.any():
                    ax.scatter(SPA[0, mask, fr], SPA[1, mask, fr],
                               c=fld[mask], cmap=SRED, vmin=0, vmax=1,
                               s=8, rasterized=True, zorder=2)
            else:
                ax.axis('off')
        return tuple(axes_flat)
    ani  = FuncAnimation(fig, update, frames=T, interval=100, blit=False)
    path = save or 'anim_seba_sets.gif'
    _save_ani(ani, path, fps)
    plt.close()
    return path

# All Spatial Modes Animation 
def anim_regime_sweep(idl, abs_cutoff=0.25, fps=10, save=None):
    """ Animation of 3 spatial modes side by side """
    Tspan   = idl.TimePoints
    SPA     = idl.SpacePointsarray
    _, N, T = SPA.shape
    nt      = _spatial_modes(idl)
    Vs      = [_prep_idl_evec(idl.evecs_3d, col, SPA) for col in nt]
    sm = plt.cm.ScalarMappable(cmap=BWR, norm=_bwr_norm())
    sm.set_array([])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.colorbar(sm, ax=axes.tolist(), shrink=0.85, pad=0.02,
                 label='eigenfunction value')
    def update(fr):
        t_val  = float(Tspan[fr])
        regime = 'coherent' if (t_val < 5 or t_val > 10) else 'mixing'
        for ci, (ax, V) in enumerate(zip(axes, Vs)):
            ax.cla()
            ax.set_aspect('equal'); ax.set_xlim(0, 2); ax.set_ylim(0, 1)
            ax.set_xlabel(r'$x_1$', fontsize=9)
            ax.set_ylabel(r'$x_2$', fontsize=9)
            ax.set_title(f'Mode {ci + 1}   t={t_val:.1f} [{regime}]', fontsize=10)
            fld = V[:, fr].copy()
            fld[np.abs(fld) < abs_cutoff] = np.nan
            ax.scatter(SPA[0, :, fr], SPA[1, :, fr],
                       c=fld, cmap=BWR, norm=_bwr_norm(), s=6, rasterized=True)
        return tuple(axes)
    ani  = FuncAnimation(fig, update, frames=T, interval=100, blit=False)
    path = save or 'anim_regime_sweep.gif'
    _save_ani(ani, path, fps)
    plt.close()
    return path
