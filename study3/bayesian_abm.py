import numpy as np
import pandas as pd
import networkx as nx
import pymc as pm
import arviz as az
from typing import Dict, List, Tuple, Optional
from study3.utils import *
from study3.dataloader import *

def fit_bayesian_parameters(df: pd.DataFrame,
                            colmap: ColumnMap = ColumnMap(),
                            draws: int = 1500,
                            tune: int = 1000,
                            target_accept: float = 0.9) -> Posteriors:
    # Masks
    m_HH = df[colmap.edge_type].str.upper().eq("HH")
    m_AH = df[colmap.edge_type].str.upper().eq("HA") | df[colmap.edge_type].str.upper().eq("AH")

    # Helper to fit a Beta-Bernoulli for rates (tx, update)
    def fit_rate(mask, col) -> az.InferenceData:
        y = df.loc[mask, col].values
        with pm.Model() as model:
            # Weakly informative, centered near observed rate (empirical Bayes via Beta prior conjugacy)
            a0 = 1.0 + y.sum()
            b0 = 1.0 + (len(y) - y.sum())
            theta = pm.Beta("theta", alpha=a0, beta=b0)  # empirical prior
            pm.Bernoulli("y", p=theta, observed=y)
            idata = pm.sample(draws=draws, tune=tune, target_accept=target_accept, chains=4, progressbar=False)
        return idata

    tx_HH = fit_rate(m_HH, colmap.transmitted)
    tx_AH = fit_rate(m_AH, colmap.transmitted)
    upd_HH = fit_rate(m_HH, colmap.updated)
    upd_AH = fit_rate(m_AH, colmap.updated)

    # Geometric step size on |delta_b| (conditionally on update=1 and transmitted=1)
    def fit_step(mask) -> az.InferenceData:
        rows = df.loc[mask & (df[colmap.updated]==1) & (df[colmap.transmitted]==1)]
        # Truncate zeros (no change) if any: model as geometric on positive integers
        k = np.abs(rows[colmap.delta_b].values)
        k = k[k > 0]
        if len(k) == 0:
            # fallback: diffuse prior with no data
            with pm.Model() as model:
                lam = pm.HalfNormal("lambda", sigma=2.0)  # rate for geometric via Poisson approx
                idata = pm.sample_prior_predictive(500)
            return idata

        # Geometric(P): P(k) = (1-P)^(k-1) P, k=1,2,...
        with pm.Model() as model:
            # Prior on P (success prob); translate to mean step 1/P
            P = pm.Beta("P", alpha=2, beta=2)
            k_obs = pm.Geometric("k_obs", p=P, observed=k)  # PyMC Geometric is on k >=1
            idata = pm.sample(draws=draws, tune=tune, target_accept=target_accept, chains=4, progressbar=False)
        # Convert to lambda-like (mean step) if you prefer reporting 1/P
        return idata

    step_HH = fit_step(m_HH)
    step_AH = fit_step(m_AH)

    return Posteriors(tx_HH, tx_AH, upd_HH, upd_AH, step_HH, step_AH)

# Posterior summary helpers
def posterior_mean(idata: az.InferenceData, varname: str) -> float:
    return float(az.extract(idata, var_names=[varname]).to_numpy().mean())

def extract_rates_and_steps(post: Posteriors) -> Dict[str, float]:
    rates = dict(
        p_tx_HH = posterior_mean(post.tx_HH, "theta"),
        p_tx_AH = posterior_mean(post.tx_AH, "theta"),
        p_upd_HH = posterior_mean(post.upd_HH, "theta"),
        p_upd_AH = posterior_mean(post.upd_AH, "theta"),
    )
    # Geometric mean step = 1/P
    try:
        rates["mean_step_HH"] = 1.0 / posterior_mean(post.step_geom_lambda_HH, "P")
    except Exception:
        rates["mean_step_HH"] = 1.0
    try:
        rates["mean_step_AH"] = 1.0 / posterior_mean(post.step_geom_lambda_AH, "P")
    except Exception:
        rates["mean_step_AH"] = 1.0
    return rates

# -----------------------------
# ABM core
# -----------------------------
@dataclass
class ABMConfig:
    B: int = 10                         # belief range [0, B]
    N_h: int = 1000                     # humans
    M_a: int = 2                        # AI models
    market_shares: Optional[np.ndarray] = None  # length M_a
    T: int = 200                        # time steps
    k_hh: int = 6                       # avg human degree to humans
    k_ha: int = 1                       # human degree to AI models
    eta_ai: float = 0.2                 # AI learning rate
    alpha_align: float = 0.0            # alignment strength [0,1]
    delta_deception: float = 0.0        # deception multiplier >=0
    seed: int = 123

@dataclass
class ABMParams:
    p_tx_HH: float
    p_tx_AH: float
    p_upd_HH: float
    p_upd_AH: float
    mean_step_HH: float
    mean_step_AH: float

@dataclass
class ABMResults:
    diversity: List[float]
    drift: List[float]
    convergence_min: List[float]
    convergence_weighted: List[float]
    traj_human_mean: List[float]
    traj_ai_means: List[List[float]]
    b0_mean: float

def make_network(cfg: ABMConfig):
    RNG = np.random.default_rng(cfg.seed)
    # Human-human graph (Watts-Strogatz small world for heterogeneity)
    G_hh = nx.watts_strogatz_graph(n=cfg.N_h, k=cfg.k_hh, p=0.1, seed=cfg.seed)
    # Assign AI connections: for each human, connect to k_ha AI models drawn by market shares
    if cfg.market_shares is None:
        w = np.ones(cfg.M_a) / cfg.M_a
    else:
        w = np.asarray(cfg.market_shares, dtype=float)
        w = w / w.sum()

    # Build bipartite edges as mapping human->list of AI model indices
    ha_edges = {i: RNG.choice(cfg.M_a, size=cfg.k_ha, replace=True, p=w).tolist() for i in range(cfg.N_h)}
    return G_hh, ha_edges, w

def init_beliefs(cfg: ABMConfig):
    RNG = np.random.default_rng(cfg.seed)
    # Humans: heterogeneous integer beliefs
    b_h = RNG.integers(0, cfg.B+1, size=cfg.N_h)
    # AI models start near human mean (can vary)
    a = RNG.integers(0, cfg.B+1, size=cfg.M_a)
    return b_h, a

def step_size(mean_step: float) -> int:
    # Draw from geometric with mean ~ mean_step, fallback step=1 if degenerate
    if mean_step <= 1e-6:
        return 1
    p = 1.0 / max(mean_step, 1e-6)
    # geometric on {1,2,...}
    k = RNG.geometric(p)
    return max(1, int(k))

def simulate_abm(cfg: ABMConfig, pars: ABMParams) -> ABMResults:
    RNG = np.random.default_rng(cfg.seed)
    G_hh, ha_edges, w = make_network(cfg)
    b_h, a = init_beliefs(cfg)

    b0_mean = b_h.mean()

    diversity, drift, conv_min, conv_w = [], [], [], []
    traj_hmean, traj_ameans = [], [[] for _ in range(cfg.M_a)]

    for t in range(cfg.T):
        # AI alignment pull toward initial human norm
        a = np.round((1 - cfg.alpha_align) * a + cfg.alpha_align * b0_mean).astype(int)
        a = np.clip(a, 0, cfg.B)

        # --- Human-Human interactions ---
        for i, j in G_hh.edges():
            # Directional selection: randomly choose who speaks
            if RNG.random() < 0.5: src, dst = i, j
            else: src, dst = j, i

            if RNG.random() < pars.p_tx_HH:
                if RNG.random() < pars.p_upd_HH:
                    s = step_size(pars.mean_step_HH)
                    di = int(np.sign(b_h[src] - b_h[dst])) * s
                    b_h[dst] = clip_int(b_h[dst] + di, 0, cfg.B)

        # --- AI-Human interactions (AI -> human only for clarity) ---
        for i in range(cfg.N_h):
            for m in ha_edges[i]:
                if RNG.random() < pars.p_tx_AH:
                    if RNG.random() < pars.p_upd_AH:
                        s = step_size(pars.mean_step_AH)
                        s = int(np.round((1.0 + cfg.delta_deception) * s))
                        di = int(np.sign(a[m] - b_h[i])) * max(1, s)
                        b_h[i] = clip_int(b_h[i] + di, 0, cfg.B)

        # --- Centralized AI update (per model) ---
        # Humans connected to model m
        for m in range(len(a)):
            # gather listeners linked to m
            idx = [i for i in range(cfg.N_h) if m in ha_edges[i]]
            if len(idx) > 0:
                mean_h = np.mean(b_h[idx])
                a[m] = clip_int(round((1 - cfg.eta_ai) * a[m] + cfg.eta_ai * mean_h), 0, cfg.B)

        # --- Metrics ---
        h_mean = b_h.mean()
        h_var = b_h.var()
        diversity.append(float(h_var))
        drift.append(float(abs(h_mean - b0_mean)))
        ameans = [float(x) for x in a]
        traj_hmean.append(float(h_mean))
        for m in range(cfg.M_a):
            traj_ameans[m].append(ameans[m])

        dists = [abs(h_mean - am) for am in ameans]
        conv_min.append(float(np.min(dists)))
        conv_w.append(float(np.sum(w * np.array(dists))))

    return ABMResults(diversity, drift, conv_min, conv_w, traj_hmean, traj_ameans, float(b0_mean))

# -----------------------------
# Scenario runner
# -----------------------------
def run_scenarios(df: pd.DataFrame,
                  market_vectors: List[np.ndarray],
                  align_list: List[float],
                  deception_list: List[float],
                  config_base: ABMConfig = ABMConfig(),
                  draws: int = 1500, tune: int = 1000) -> pd.DataFrame:
    # Fit Bayesian posteriors from CSV
    post = fit_bayesian_parameters(df, draws=draws, tune=tune)
    rates = extract_rates_and_steps(post)
    pars = ABMParams(**rates)

    rows = []
    for w in market_vectors:
        for alpha in align_list:
            for delta in deception_list:
                cfg = ABMConfig(**{**config_base.__dict__,
                                   "market_shares": np.array(w, dtype=float),
                                   "alpha_align": alpha,
                                   "delta_deception": delta,
                                  })
                res = simulate_abm(cfg, pars)
                rows.append(dict(
                    market_shares=";".join([f"{x:.2f}" for x in (np.array(w)/np.sum(w))]),
                    HHI=np.sum((np.array(w)/np.sum(w))**2),
                    alpha_align=alpha,
                    delta_deception=delta,
                    diversity_end=res.diversity[-1],
                    drift_end=res.drift[-1],
                    conv_min_end=res.convergence_min[-1],
                    conv_w_end=res.convergence_weighted[-1],
                    human_mean_end=res.traj_human_mean[-1],
                    human_mean_start=res.b0_mean,
                ))
    return pd.DataFrame(rows)

# -----------------------------
# Example usage (uncomment to run)
# -----------------------------
# if __name__ == "__main__":
#     df = load_study2_csv("study2_interactions.csv")
#     markets = [np.array([1.0]), np.array([0.7, 0.3]), np.array([0.5, 0.3, 0.2])]
#     aligns = [0.0, 0.25, 0.5, 0.75, 1.0]
#     deceps = [0.0, 0.25, 0.5]
#     results = run_scenarios(df, markets, aligns, deceps)
#     print(results.sort_values(["HHI","alpha_align","delta_deception"]))
