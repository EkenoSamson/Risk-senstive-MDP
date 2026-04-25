# Risk-Sensitive Dynamic Programming

Illustrating risk-averse Bellman equations
(Ruszczynski 2010, Theorem 2) on two different MDPs.

---

## Files

| File | Description |
|------|-------------|
| `Iot_solver.py` | IoT Age-of-Information MDP (Exercise 5.4) |
| `dynamic_pricing_mdp.py` | Risk-Averse Concert Ticket Pricing |
| `requirements.txt` | Python dependencies |

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9 or later.

---

## solve.py — IoT AoI MDP

**Problem.** An IoT device observes a random walk and decides each period
whether to transmit its current reading (cost λ) or stay silent (cost = staleness²).
State: S_t = X_t − Z_t ∈ {−B, ..., B}.

**Covers exercise parts:**
- (a) Backward induction with T=20, λ=100, B=100
- (b) Value functions for t ∈ {1, 5, 10, 19}
- (c) Optimal policies for t ∈ {1, 5, 10, 19}
- (d) Boundary sensitivity B ∈ {50, 60, 70, 80, 100}

**Run:**
```bash
python Iot_solve.py
```

**Runtime:** ~2–4 min (CVaR solves one LP per state per time step at B=100).
For a quick test reduce B in the `if __name__ == '__main__'` block to 50.

**Outputs** (written to `plots/`):

| File | Contents |
|------|----------|
| `iot_value_policy.png` | Value functions and policies — EV vs CVaR (parts b & c) |
| `iot_threshold.png` | Transmission threshold s*(t) vs time remaining |
| `iot_B_sensitivity.png` | Value and policy at t=10 for B ∈ {50,60,70,80,100} (part d) |

---

## dynamic_pricing_mdp.py — Concert Ticket Pricing

**Problem.** A firm sells C=5 perishable tickets over T=10 periods.
Demand d(p) = max(0, 1−p/25). Goal: maximise revenue under three risk criteria.

**Risk measures compared:**
- Expected value (risk-neutral baseline)
- CVaR at α=0.3 (adversary inflates bad-outcome probabilities by 1/α)
- Mean-semideviation at κ=1.0 (penalises upside deviations from conditional mean)

**Run:**
```bash
python dynamic_pricing_mdp.py
```

**Runtime:** < 30 seconds.

**Outputs** (written to `plots/`):

| File | Contents |
|------|----------|
| `rev_dist_c_5.png` | Revenue distributions — starting inventory c=5 |
| `rev_dist_c_2.png` | Revenue distributions — starting inventory c=2 |
| `policy_plot_c_5.png` | Optimal price path at c=5 |
| `policy_plot_c_2.png` | Optimal price path at c=2 |
| `value_functions.png` | V(t,c) for all methods and inventory levels |
| `complexity_scaling.png` | Empirical time and space complexity vs T |

---

## references
[Aditya Mahaja ECSE 506](https://adityam.github.io/stochastic-control/mdps/intro.html#exercises)

[Risk-averse dynamic pricing using mean-semivariance optimization](https://www.sciencedirect.com/science/article/pii/S0377221723002710?via%3Dihub#absh001)
