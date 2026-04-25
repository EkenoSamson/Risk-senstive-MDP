"""
dynamic_pricing_mdp.py — Risk-Averse Dynamic Pricing
=====================================================
Setting: Schlosser & Goensch (2023), grounded in Ruszczynski (2010)

A firm sells C perishable units over T periods.
At each period t with c units left it sets price p.
A customer arrives and buys with probability d(p).
Goal: choose prices to maximise revenue under three risk criteria.

Bellman equation — finite horizon (Theorem 2, Ruszczynski 2010)

    V_t(c) = max_{p in P} { d(p)*p + sigma( [V_{t+1}(c), V_{t+1}(c-1)],
                                             [1-d(p),     d(p)         ] ) }

Only sigma changes across methods. Algorithm is backward induction — one pass.

Outputs
    plots/rev_dist_c_5.png      Revenue distributions starting at c=5
    plots/rev_dist_c_2.png      Revenue distributions starting at c=2
    plots/policy_plot_c_5.png   Optimal prices over time at c=5
    plots/policy_plot_c_2.png   Optimal prices over time at c=2
    plots/value_functions.png   Value functions V(t,c) for all methods
    plots/complexity_scaling.png  Empirical time and space complexity

Dependencies: numpy, matplotlib, scipy
Run        : python dynamic_pricing_mdp.py
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       11,
    'axes.labelsize':  12,
    'axes.titlesize':  12,
    'legend.fontsize': 9,
    'figure.dpi':      180,
})

OUTDIR = 'plots'
os.makedirs(OUTDIR, exist_ok=True)

# ── Colour palette (consistent with notebook) ──────────────────────
COL_EV  = '#378ADD'
COL_CV  = '#D85A30'
COL_MSD = '#1D9E75'


# ══════════════════════════════════════════════════════════════════
# Problem parameters
# ══════════════════════════════════════════════════════════════════

T      = 10
C      = 5
PRICES = np.array([0, 5, 10, 15, 20, 25], dtype=float)


def demand(p):
    """Linear demand: d(p) = max(0, 1 - p/25)."""
    return np.maximum(0., 1. - p / 25.)


# ══════════════════════════════════════════════════════════════════
# Risk transition mappings  sigma(v_next, probs) -> float
# ══════════════════════════════════════════════════════════════════

def sigma_EV(v_next, probs, **kw):
    """
    Expected value — risk neutral.
    Complexity: O(n) where n = len(v_next).
    """
    return float(np.dot(probs, v_next))


def sigma_CVaR(v_next, probs, alpha_risk=0.3, **kw):
    """
    CVaR at level alpha_risk.
    Adversary up-weights bad (low-revenue) outcomes by factor 1/alpha_risk.

    Solved as LP:
        min_{theta, xi}  theta + (1/alpha) * sum_j p_j * xi_j
        s.t.  xi_j >= v_j - theta,  xi_j >= 0

    Complexity: O(n * LP_solve(n)).
    For n=2 (two next states) this is effectively O(1).
    """
    n   = len(v_next)
    obj = np.concatenate([[1.], probs / alpha_risk])
    A   = np.hstack([-np.ones((n, 1)), -np.eye(n)])
    res = linprog(obj, A_ub=A, b_ub=-v_next,
                  bounds=[(None, None)] + [(0., None)] * n,
                  method='highs')
    return float(res.fun) if res.success else float(np.dot(probs, v_next))


def sigma_MSD(v_next, probs, kappa=1.0, **kw):
    """
    Mean-semideviation (r=1).
        sigma = E[v] + kappa * E[(v - E[v])_+]

    Penalises upward deviations from the conditional mean linearly.
    Complexity: O(n).
    """
    mu = float(np.dot(probs, v_next))
    return mu + kappa * float(np.dot(probs, np.maximum(v_next - mu, 0.)))


# ══════════════════════════════════════════════════════════════════
# Backward induction (Theorem 2)
# ══════════════════════════════════════════════════════════════════

def backward_induction(sigma_fn, T_val=T, C_val=C, **params):
    """
    Finite-horizon risk-averse Bellman equations.
    State: (t, c) — period index and remaining inventory.
    Runs backward from t=T to t=0.

    Complexity
        Time : O(T * C * |P| * cost(sigma))
        Space: O(T * C)

    Returns
    -------
    V  : value function  shape (T_val+1, C_val+1)
    pi : optimal price index  shape (T_val, C_val+1)
    """
    V  = np.zeros((T_val + 1, C_val + 1))
    pi = np.zeros((T_val,     C_val + 1), dtype=int)

    for t in reversed(range(T_val)):
        for c in range(1, C_val + 1):
            best_val, best_idx = -np.inf, 0
            for idx, p in enumerate(PRICES):
                d      = float(demand(p))
                probs  = np.array([1. - d, d])
                v_next = np.array([V[t + 1, c], V[t + 1, c - 1]])
                val    = d * p + sigma_fn(v_next, probs, **params)
                if val > best_val:
                    best_val, best_idx = val, idx
            V[t, c]  = best_val
            pi[t, c] = best_idx

    return V, pi


# ══════════════════════════════════════════════════════════════════
# Monte Carlo simulation
# ══════════════════════════════════════════════════════════════════

def simulate(pi, start_c=C, n_sim=5000, seed=42):
    """
    Simulate policy pi for n_sim episodes starting with start_c units.

    Returns
    -------
    mu       : float — mean revenue
    sv       : float — semivariance (downside variance below mean)
    revenues : ndarray (n_sim,)
    """
    rng      = np.random.default_rng(seed)
    revenues = np.zeros(n_sim)
    for sim in range(n_sim):
        c, total = start_c, 0.
        for t in range(T):
            if c == 0:
                break
            p = PRICES[pi[t, c]]
            if rng.random() < float(demand(p)):
                total += p
                c     -= 1
        revenues[sim] = total
    mu = revenues.mean()
    sv = float(np.mean(np.maximum(mu - revenues, 0.) ** 2))
    return mu, sv, revenues


# ══════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════

def plot_revenue_distributions(rev_ev, rev_cv, rev_msd,
                               start_c, bins, filename):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, rev, col in [('Expected value', rev_ev,  COL_EV),
                            (r'CVaR $\alpha=0.3$', rev_cv,  COL_CV),
                            (r'MSD $\kappa=1.0$',  rev_msd, COL_MSD)]:
        ax.hist(rev, bins=bins, alpha=0.3,  color=col, density=True,
                histtype='stepfilled')
        ax.hist(rev, bins=bins, alpha=0.9,  color=col, density=True,
                histtype='step', linewidth=2, label=name)
        ax.axvline(rev.mean(), color=col, linewidth=2, linestyle='--')

    ax.set_xlabel('Total revenue')
    ax.set_ylabel('Density')
    ax.set_title(f'Revenue distributions (5000 simulations)\n'
                 f'Starting inventory $c={start_c}$')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTDIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def plot_price_path(pi_ev, pi_cv, pi_msd, inventory, filename):
    """Optimal price at each period for a fixed inventory level."""
    prices_ev  = [PRICES[pi_ev[t,  inventory]] for t in range(T)]
    prices_cv  = [PRICES[pi_cv[t,  inventory]] for t in range(T)]
    prices_msd = [PRICES[pi_msd[t, inventory]] for t in range(T)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(T), prices_ev,  'o-',  color=COL_EV,  lw=2.5, ms=7,
            alpha=0.8, label='Expected value')
    ax.plot(range(T), prices_cv,  's-',  color=COL_CV,  lw=2.5, ms=7,
            alpha=0.8, label=r'CVaR $\alpha=0.3$')
    ax.plot(range(T), prices_msd, '^--', color=COL_MSD, lw=2.5, ms=7,
            alpha=0.8, label=r'MSD $\kappa=1.0$')
    ax.set_xlabel('Time period $t$')
    ax.set_ylabel(f'Optimal price (at $c={inventory}$)')
    ax.set_title(f'Optimal prices over time  ($c={inventory}$)')
    ax.set_xticks(range(T))
    ax.legend()
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTDIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def plot_value_functions(V_ev, V_cv, V_msd):
    """Value functions V(t, c) for all inventory levels and all methods."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, C))

    for ax, V, name in zip(axes,
                            [V_ev, V_cv, V_msd],
                            ['Expected value',
                             r'CVaR $\alpha=0.3$',
                             r'MSD $\kappa=1.0$']):
        for c_idx, c in enumerate(range(1, C + 1)):
            ax.plot(range(T + 1), V[:, c], color=colors[c_idx],
                    linewidth=2, label=f'$c={c}$')
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Period $t$')
        ax.grid(alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel('$V(t, c)$')
    axes[2].legend(title='Inventory', fontsize=9)
    fig.suptitle('Value functions — risk-averse values are larger\n'
                 '(pessimistic agent overestimates future risk)',
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTDIR, 'value_functions.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def plot_complexity_scaling():
    """
    Empirical time and space complexity as T scales from 10 to 50.

    Time complexity:
        EV / MSD : O(T * C * |P|)           — linear in T
        CVaR     : O(T * C * |P| * LP(n))   — linear in T, larger constant

    Space complexity: O(T * C) — linear in T for V and pi matrices.
    """
    T_values  = [10, 20, 30, 40, 50]
    times_ev  = []
    times_msd = []
    times_cv  = []
    space_B   = []

    for T_val in T_values:
        t0 = time.perf_counter()
        V_, pi_ = backward_induction(sigma_EV, T_val=T_val)
        times_ev.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        backward_induction(sigma_MSD, T_val=T_val, kappa=1.0)
        times_msd.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        backward_induction(sigma_CVaR, T_val=T_val, alpha_risk=0.3)
        times_cv.append(time.perf_counter() - t0)

        space_B.append(V_.nbytes + pi_.nbytes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(T_values, times_ev,  'o-', color=COL_EV,  lw=2, label='EV')
    ax1.plot(T_values, times_msd, 's-', color=COL_MSD, lw=2, label='MSD')
    ax1.plot(T_values, times_cv,  '^-', color=COL_CV,  lw=2,
             label=r'CVaR (LP solver)')
    ax1.set_xlabel('Number of time periods $T$')
    ax1.set_ylabel('Execution time (s)')
    ax1.set_title('Empirical Time Complexity — Scaling with $T$')
    ax1.legend()
    ax1.grid(alpha=0.25)
    ax1.spines[['top', 'right']].set_visible(False)

    ax2.plot(T_values, space_B, 'o-', color='#888780', lw=2,
             label='$V$ + $\\pi$ matrices')
    ax2.set_xlabel('Number of time periods $T$')
    ax2.set_ylabel('Memory (bytes)')
    ax2.set_title('Empirical Space Complexity — Scaling with $T$')
    ax2.legend()
    ax2.grid(alpha=0.25)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'complexity_scaling.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


# ══════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════

def print_summary(results):
    print(f'\n{"Method":<22} {"E[Revenue]":>12} {"Semivariance":>14}')
    print('-' * 50)
    for name, mu, sv in results:
        print(f'{name:<22} {mu:>12.2f} {sv:>14.2f}')


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Solve ──────────────────────────────────────────────────────
    print('Running backward induction ...')
    V_ev,  pi_ev  = backward_induction(sigma_EV)
    V_cv,  pi_cv  = backward_induction(sigma_CVaR, alpha_risk=0.3)
    V_msd, pi_msd = backward_induction(sigma_MSD,  kappa=1.0)

    print(f'V(t=0, c=5):  EV={V_ev[0,5]:.2f}  '
          f'CVaR={V_cv[0,5]:.2f}  MSD={V_msd[0,5]:.2f}')

    # ── Simulate c=5 ───────────────────────────────────────────────
    print('\nSimulating (c=5) ...')
    mu_ev,  sv_ev,  rev_ev  = simulate(pi_ev,  start_c=5)
    mu_cv,  sv_cv,  rev_cv  = simulate(pi_cv,  start_c=5)
    mu_msd, sv_msd, rev_msd = simulate(pi_msd, start_c=5)
    print_summary([('Expected value', mu_ev,  sv_ev),
                   (r'CVaR a=0.3',   mu_cv,  sv_cv),
                   (r'MSD  k=1.0',   mu_msd, sv_msd)])

    # ── Simulate c=2 ───────────────────────────────────────────────
    print('\nSimulating (c=2) ...')
    mu_ev2,  sv_ev2,  rev_ev2  = simulate(pi_ev,  start_c=2)
    mu_cv2,  sv_cv2,  rev_cv2  = simulate(pi_cv,  start_c=2)
    mu_msd2, sv_msd2, rev_msd2 = simulate(pi_msd, start_c=2)
    print_summary([('Expected value', mu_ev2,  sv_ev2),
                   (r'CVaR a=0.3',   mu_cv2,  sv_cv2),
                   (r'MSD  k=1.0',   mu_msd2, sv_msd2)])

    # ── Plots ──────────────────────────────────────────────────────
    print('\nGenerating plots ...')

    plot_revenue_distributions(rev_ev, rev_cv, rev_msd,
                               start_c=5,
                               bins=np.arange(0, 105, 5),
                               filename='rev_dist_c_5.png')

    plot_revenue_distributions(rev_ev2, rev_cv2, rev_msd2,
                               start_c=2,
                               bins=np.arange(0, 60, 5),
                               filename='rev_dist_c_2.png')

    plot_price_path(pi_ev, pi_cv, pi_msd,
                    inventory=5,
                    filename='policy_plot_c_5.png')

    plot_price_path(pi_ev, pi_cv, pi_msd,
                    inventory=2,
                    filename='policy_plot_c_2.png')

    plot_value_functions(V_ev, V_cv, V_msd)

    print('\nComplexity scaling benchmark ...')
    plot_complexity_scaling()

    print('\nAll done.')
