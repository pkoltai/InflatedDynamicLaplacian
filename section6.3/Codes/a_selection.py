"""a_selection.py
Heuristic selection of the temporal diffusion parameter  a """

import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Computation function
def compute_a(tau: float, s1: float, s2: float, multiplier: float = 1.0,) -> dict:
    """ Compute the temporal diffusion parameter a """
    s_max       = max(s1, s2)
    a_min       = tau / s_max
    a           = multiplier * a_min
    t_factor    = a ** 2
    lambda_spat = -(np.pi / s_max) ** 2
    lambda_temp = -(a * np.pi / tau) ** 2

    return { 'a_min': a_min, 'a': a, 't_factor': t_factor, 'lambda_spat': lambda_spat,
              'lambda_temp': lambda_temp,}

# Plotting The Heuristic
def plot_a_heuristic(tau, result, paper_a=None, save=None):
    """Plot temporal vs spatial eigenvalue curves to visualise the heuristic"""
    a_min  = result['a_min']
    a_vals = np.linspace(0.01 * a_min, 3.0 * result['a'], 400)
    lam_temp = -(a_vals * np.pi / tau) ** 2
    lam_spat  = result['lambda_spat']
    plt.figure(figsize=(7, 5))
    plt.plot(a_vals, lam_temp, lw=2, color='darkorange',
             label=r'$\lambda_1^{\rm temp}(a)$')
    plt.axhline(lam_spat, color='steelblue', ls='--', lw=2,
                label=fr'$\lambda_1^{{\rm spat}}$')
    plt.axvline(a_min, color='gray', ls='--', lw=1.5,
                label=fr'$a_{{\min}} = {a_min:.4f}$')
    plt.axvline(result['a'], color='seagreen', lw=2,
                label=fr"$a = {result['a']:.4f}$")
    plt.xlabel('$a$', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.title('Temporal Diffusion Parameter Heuristic')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f'  Saved {save}')
    plt.close()

# Main
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from generate_pts_CMC import Tspan, Ix, Iy
    paper_a    = 30.0601
    multiplier = 4.0
    tau = float(np.asarray(Tspan).ravel()[-1] - np.asarray(Tspan).ravel()[0])
    s1  = float(np.asarray(Ix).ravel()[-1]    - np.asarray(Ix).ravel()[0])
    s2  = float(np.asarray(Iy).ravel()[-1]    - np.asarray(Iy).ravel()[0])
    result = compute_a(tau, s1, s2, multiplier=multiplier)
    print(f'a_min          = {result["a_min"]:.6f}')
    print(f'a (4*a_min)    = {result["a"]:.6f}')
    np.save('a_value.npy', np.float64(result['a']))
    plot_a_heuristic(tau, result, paper_a=paper_a, save='a_heuristic.png')
    print('Saved a_value.npy  a_heuristic.png')
