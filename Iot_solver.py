"""
Iot_solver.py — IoT Age-of-Information MDP

"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── Global style ────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════
# Noise distribution
# ══════════════════════════════════════════════════════════════════

WS = np.arange(-5, 6)
PW = np.array([1/5 - abs(w)/25 for w in WS], dtype=float)
PW /= PW.sum()


# ══════════════════════════════════════════════════════════════════
# Risk transition mappings  sigma(v, P) -> R^N
# ══════════════════════════════════════════════════════════════════

def sigma_EV(v, P, **kw):
    """
    Expected value — risk neutral.
    sigma_i = sum_j p_ij * v_j
    Complexity: O(N^2) via matrix-vector product.
    """
    return P @ v


def sigma_CVaR(v, P, alpha=0.3, **kw):
    """
    CVaR risk transition mapping via the dual (adversarial kernel) form.

    Complexity: O(N log N) per state (dominated by the sort, done once).
    Total: O(T * N^2 * log N) — no LP solver needed.
    """
    N = len(v)
    result = np.zeros(N)
    order = np.argsort(-v)          # sort descending once, reuse for all rows

    for i in range(N):
        p = P[i]
        remaining = 1.0
        val = 0.0
        for j in order:
            mass = min(p[j] / alpha, remaining)
            val += mass * v[j]
            remaining -= mass
            if remaining <= 1e-12:
                break
        result[i] = val

    return result


def sigma_MSD(v, P, kappa=1.0, **kw):
    """
    Mean-semideviation (r=1).

        sigma_i = E[v | s_i] + kappa * E[(v - E[v|s_i])_+ | s_i]

    Complexity: O(N^2).
    """
    N = len(v)
    result = np.zeros(N)
    for i in range(N):
        mu  = float(P[i] @ v)
        pen = float(P[i] @ np.maximum(v - mu, 0.))
        result[i] = mu + kappa * pen
    return result


# ══════════════════════════════════════════════════════════════════
# Transition matrices
# ══════════════════════════════════════════════════════════════════

def build_transition(B):
    """
    Build P0, P1 — transition matrices for actions 0 and 1.
    States : {-B, ..., B},  N = 2B+1.

    Because noise support is [-5, 5] and is bounded, each row of P0
    and P1 has at most 11 nonzero entries regardless of N.
    """
    states = np.arange(-B, B + 1)
    N      = len(states)
    s_idx  = {int(s): i for i, s in enumerate(states)}

    P0 = np.zeros((N, N))
    P1 = np.zeros((N, N))
    for i, s in enumerate(states):
        for w, pw in zip(WS, PW):
            s0 = int(np.clip(s + w, -B, B))
            s1 = int(np.clip(w,     -B, B))
            P0[i, s_idx[s0]] += pw
            P1[i, s_idx[s1]] += pw
    return P0, P1


# ══════════════════════════════════════════════════════════════════
# Backward induction  (Theorem 2 of Ruszczyński 2010)
# ══════════════════════════════════════════════════════════════════

def backward_induction(sigma_fn, B=100, T=20, lam=100, **kw):
    """
    Finite-horizon risk-averse Bellman equations.

    v_{T+1}(s) = 0
    v_t(s)     = min_{a in {0,1}} { c(s,a) + sigma(v_{t+1}, s, P^a(s,·)) }

    Parameters
    ----------
    sigma_fn : callable  — one of sigma_EV, sigma_CVaR, sigma_MSD
    B        : int       — state space truncation boundary
    T        : int       — horizon
    lam      : float     — transmission cost lambda
    **kw                 — passed through to sigma_fn (e.g. alpha, kappa)

    Returns
    -------
    states : ndarray (N,)
    V      : ndarray (T+1, N)   V[t] = value function at time index t
    pol    : ndarray (T, N)     pol[t, i] = optimal action at state i, time t
    """
    states  = np.arange(-B, B + 1)
    N       = len(states)
    P0, P1  = build_transition(B)

    c0 = states ** 2               # cost of not transmitting
    c1 = np.full(N, float(lam))   # cost of transmitting

    V   = np.zeros((T + 1, N))
    pol = np.zeros((T,     N), dtype=int)

    for t in range(T - 1, -1, -1):
        q0     = c0 + sigma_fn(V[t + 1], P0, **kw)
        q1     = c1 + sigma_fn(V[t + 1], P1, **kw)
        pol[t] = (q1 < q0).astype(int)
        V[t]   = np.minimum(q0, q1)

    return states, V, pol


# ══════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════

PLOT_TIMES = [1, 5, 10, 19]


def _t_indices(T, times):
    return [T - t for t in times]


def plot_value_and_policy(states, V_ev, pol_ev, V_cv, pol_cv, T=20):
    """Value functions and policies for EV and CVaR."""
    t_idx = _t_indices(T, PLOT_TIMES)
    blues = plt.cm.Blues(np.linspace(0.40, 0.90, len(PLOT_TIMES)))
    reds  = plt.cm.Reds( np.linspace(0.40, 0.90, len(PLOT_TIMES)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f'IoT AoI MDP — Value Functions and Optimal Policies\n'
        f'$T={T}$,  $\\lambda=100$,  $B={int(states[-1])}$',
        fontsize=13)

    for k, (t, ti) in enumerate(zip(PLOT_TIMES, t_idx)):
        axes[0, 0].plot(states, V_ev[ti],   color=blues[k], label=f't={t}')
        axes[0, 1].plot(states, V_cv[ti],   color=reds[k],  label=f't={t}')
        axes[1, 0].plot(states, pol_ev[ti], color=blues[k], label=f't={t}', lw=2)
        axes[1, 1].plot(states, pol_cv[ti], color=reds[k],  label=f't={t}', lw=2)

    titles_top = ['Value Function — Expected Value',
                  r'Value Function — CVaR $\alpha=0.3$']
    titles_bot = ['Optimal Policy — Expected Value',
                  r'Optimal Policy — CVaR $\alpha=0.3$']

    for ax, ttl in zip(axes[0], titles_top):
        ax.set_xlabel('State $s$')
        ax.set_ylabel('$V_t(s)$')
        ax.set_title(ttl)
        ax.legend()
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

    for ax, ttl in zip(axes[1], titles_bot):
        ax.set_xlabel('State $s$')
        ax.set_ylabel('Action $\\hat{\\pi}_t(s)$')
        ax.set_title(ttl + '\n(1 = transmit, 0 = no transmit)')
        ax.set_yticks([0, 1])
        ax.legend()
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'iot_value_policy.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def plot_threshold(states, pol_ev, pol_cv, pol_ms, T=20):
    """Smallest s >= 0 at which agent transmits, vs time remaining."""
    times_rem = np.arange(1, T + 1)

    def threshold(pol):
        thresh = []
        for t in times_rem:
            ti    = T - t
            pos   = states >= 0
            s_pos = states[pos]
            p_pos = pol[ti][pos]
            idx   = np.where(p_pos == 1)[0]
            thresh.append(int(s_pos[idx[0]]) if len(idx) > 0
                          else int(states[-1]) + 1)
        return thresh

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times_rem, threshold(pol_ev), 'o-', color='#378ADD',
            label='Expected value', ms=5)
    ax.plot(times_rem, threshold(pol_cv), 's-', color='#D85A30',
            label=r'CVaR $\alpha=0.3$', ms=5)
    ax.plot(times_rem, threshold(pol_ms), '^-', color='#1D9E75',
            label=r'MSD $\kappa=1.0$', ms=5)
    ax.set_xlabel('Time remaining $t$')
    ax.set_ylabel('Threshold $s^*_t$')
    ax.set_title('Transmission Threshold vs Time Remaining\n'
                 'Agent transmits when $|S_t| \\geq s^*_t$')
    ax.legend()
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTDIR, 'iot_threshold.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def plot_B_sensitivity(T=20, lam=100):
    """Vary B in {50,60,70,80,100} — EV value and policy at t=10."""
    Bs     = [50, 60, 70, 80, 100]
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(Bs)))
    ti     = T - 10

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for k, B in enumerate(Bs):
        states_B, V_B, pol_B = backward_induction(sigma_EV, B=B, T=T, lam=lam)
        axes[0].plot(states_B, V_B[ti],   color=colors[k], label=f'B={B}')
        axes[1].plot(states_B, pol_B[ti], color=colors[k], label=f'B={B}', lw=2)
        print(f'  B={B} done')

    axes[0].set_title('Value Function at $t=10$ — varying $B$')
    axes[0].set_xlabel('State $s$')
    axes[0].set_ylabel('$V_{10}(s)$')
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].set_title('Optimal Policy at $t=10$ — varying $B$')
    axes[1].set_xlabel('State $s$')
    axes[1].set_ylabel('Action')
    axes[1].set_yticks([0, 1])
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[1].spines[['top', 'right']].set_visible(False)

    fig.suptitle('Sensitivity to Boundary $B$ — Expected Value, $t=10$',
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTDIR, 'iot_B_sensitivity.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    T, LAM, B = 20, 100, 100

    print('Solving IoT AoI MDP  (T=20, lambda=100, B=100)')
    print('  Expected Value ...')
    states, V_ev, pol_ev = backward_induction(sigma_EV,  B=B, T=T, lam=LAM)

    print('  CVaR (alpha=0.3) — adversarial kernel ...')
    states, V_cv, pol_cv = backward_induction(sigma_CVaR, B=B, T=T, lam=LAM,
                                              alpha=0.3)
    print('  MSD (kappa=1.0) ...')
    states, V_ms, pol_ms = backward_induction(sigma_MSD,  B=B, T=T, lam=LAM,
                                              kappa=1.0)

    print('\nPlotting value functions and policies ...')
    plot_value_and_policy(states, V_ev, pol_ev, V_cv, pol_cv, T=T)

    print('Plotting transmission threshold ...')
    plot_threshold(states, pol_ev, pol_cv, pol_ms, T=T)

    print('\nBoundary sensitivity ...')
    plot_B_sensitivity(T=T, lam=LAM)

    print('\nAll done.')
