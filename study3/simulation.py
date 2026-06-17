"""
Author: Joshua Ashkinaze
Description: Simulation engine for Study 3 — Synthetic Social Learning network model.
  Pure computation: no plotting, no file I/O. Import this from generate_data.py.

  AI opinion: each period each model draws a_k ~ N(mu_h + delta*sigma_h, (sigma_ai*sigma_h)^2).
  Agents on the same model get the same draw (correlated); agents on different models are independent.
  Three domains tracked with domain-specific SSL prevalence and beta_ai from Studies 1b and 2.

Inputs:
  - None (parameters set as module-level constants)

Outputs:
  - None (imported by generate_data.py)
"""

import numpy as np

###############################################################################
# Constants (from Studies 1b and 2)
###############################################################################

BETA_H  = 0.28                  # human social influence weight  (Study 2)
K       = 3                     # mean advice-network degree     (GSS CDN)

DOMAINS  = ["moral", "personal", "conventional"]
DLABELS  = {"moral": "Moral", "personal": "Personal", "conventional": "Conventional"}

P_DOMAIN = {"moral": 0.20, "personal": 0.40, "conventional": 0.60}   # Study 1b
BETA_AI  = {"moral": 9.8/31.4, "personal": 7.1/31.4, "conventional": 8.1/31.4}  # Study 2

N      = 1_000
T      = 200
N_REPS = 50

DEFAULT = dict(p_chatbot=0.52, delta=1.0, sigma_ai=0.3, n_models=3, homophily=0.5)


###############################################################################
# Derived quantities
###############################################################################

def p_ssl(p_chatbot, domain):
    """P(SSL in domain) = P(chatbot) × P(domain | chatbot)."""
    return p_chatbot * P_DOMAIN[domain]

def hub_rho(p_chatbot, domain, n_models, n=N):
    """AI hub degree ratio: AI weekly reach / individual peer reach."""
    return p_ssl(p_chatbot, domain) * n / (n_models * K)


###############################################################################
# Network
###############################################################################

def make_network(ssl_any, k, h, rng):
    """
    Stochastic block model on SSL-connected vs non-connected agents.
    h=0.5 → random mixing; h>0.5 → SSL users cluster; h<0.5 → SSL users bridge.
    """
    n       = len(ssl_any)
    ssl_idx = np.where(ssl_any)[0]
    non_idx = np.where(~ssl_any)[0]
    adj = []
    for i in range(n):
        same  = (ssl_idx[ssl_idx != i] if ssl_any[i] else non_idx[non_idx != i])
        other = (non_idx if ssl_any[i] else ssl_idx)
        nbrs, tries = set(), 0
        while len(nbrs) < k and tries < k * 10:
            if len(same) > 0 and (len(other) == 0 or rng.random() < h):
                nbrs.add(int(rng.choice(same)))
            elif len(other) > 0:
                nbrs.add(int(rng.choice(other)))
            tries += 1
        adj.append(np.array(list(nbrs), dtype=int))
    return adj


###############################################################################
# Simulation
###############################################################################

def run_once(p_chatbot=DEFAULT["p_chatbot"],
             delta=DEFAULT["delta"],
             sigma_ai=DEFAULT["sigma_ai"],
             n_models=DEFAULT["n_models"],
             homophily=DEFAULT["homophily"],
             n=N, t_max=T, rng=None):
    """Single replication. Returns dict of time-series arrays (length t_max+1)."""
    if rng is None:
        rng = np.random.default_rng()

    beliefs = {d: rng.beta(2, 2, size=n) for d in DOMAINS}
    b0      = {d: beliefs[d].copy()       for d in DOMAINS}

    ssl     = {d: rng.random(n) < p_ssl(p_chatbot, d) for d in DOMAINS}
    ssl_any = np.any([ssl[d] for d in DOMAINS], axis=0)
    assign  = {d: np.where(ssl[d], rng.integers(0, n_models, size=n), -1)
               for d in DOMAINS}

    ai_params = {}
    for d in DOMAINS:
        mu_h  = float(b0[d].mean())
        sig_h = float(b0[d].std())
        ai_params[d] = (mu_h + delta * sig_h, max(sigma_ai * sig_h, 1e-6))

    adj = make_network(ssl_any, K, homophily, rng)
    deg = np.array([len(a) for a in adj])

    keys = [f"{m}_{d}" for m in ("pop_drift", "nonssl_drift", "polarization")
            for d in DOMAINS]
    ts   = {k: np.zeros(t_max + 1) for k in keys}

    def record(t):
        for d in DOMAINS:
            b, s = beliefs[d], ssl[d]
            ns   = ~s
            mu0  = float(b0[d].mean())
            ts[f"pop_drift_{d}"][t]    = (b.mean() - mu0) * 100
            ts[f"nonssl_drift_{d}"][t] = (b[ns].mean() - mu0) * 100 if ns.any() else 0.
            ts[f"polarization_{d}"][t] = (
                (b[s].mean() - b[ns].mean()) * 100 if s.any() and ns.any() else 0.)

    record(0)
    for t in range(1, t_max + 1):
        ai_draws = {d: np.clip(rng.normal(*ai_params[d], n_models), 0.01, 0.99)
                    for d in DOMAINS}
        for d in DOMAINS:
            b   = beliefs[d].copy()
            asg = assign[d]

            who_s = np.where(deg > 0)[0]
            if len(who_s):
                pos  = (rng.random(len(who_s)) * deg[who_s]).astype(int)
                nbrs = np.array([adj[i][p] for i, p in zip(who_s, pos)])
                b[who_s] += BETA_H * (b[nbrs] - b[who_s])

            who_ai = np.where(asg >= 0)[0]
            if len(who_ai):
                b[who_ai] += BETA_AI[d] * (ai_draws[d][asg[who_ai]] - b[who_ai])

            beliefs[d] = np.clip(b, 0., 1.)
        record(t)

    return ts


def run_cond(overrides, n_reps=N_REPS, base_seed=0, t_max=None):
    """Run n_reps replications; return (mean_dict, std_dict) over time."""
    _t  = t_max if t_max is not None else T
    cfg = {**DEFAULT, "n": N, "t_max": _t}
    cfg.update(overrides)
    runs = [run_once(**cfg, rng=np.random.default_rng(base_seed * 100 + r))
            for r in range(n_reps)]
    keys = list(runs[0].keys())
    M = {k: np.mean([x[k] for x in runs], axis=0) for k in keys}
    S = {k: np.std( [x[k] for x in runs], axis=0) for k in keys}
    return M, S
