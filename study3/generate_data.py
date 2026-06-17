"""
Author: Joshua Ashkinaze
Description: Runs all simulation conditions for Study 3 and saves results to data.pkl.
  Run this once (takes ~8 min). plot.py reads data.pkl — no re-simulation needed.

Inputs:
  - simulation.py (engine)

Outputs:
  - data.pkl: dict of all precomputed (M, S) results keyed by figure/condition
"""

import pickle
import time
from pathlib import Path

import numpy as np
import simulation as sim

HERE = Path(__file__).parent
EVAL = min(150, sim.T)


###############################################################################
# Condition sets (mirrors what plot.py needs)
###############################################################################

DELTA_VALS  = [0.0, 0.5, 1.0, 2.0]
SIGMA_VALS  = [0.1, 0.3, 0.6, 1.0, 2.0]
HOM_VALS    = [0.2, 0.35, 0.5, 0.65, 0.8]
BETA_VALS   = [0.05, 0.10, 0.14, 0.20, 0.28]   # all 5 for ratio panel
DV_GRID     = [0.0, 0.5, 1.0, 2.0]
SV_GRID     = [0.1, 0.3, 0.6, 1.0, 2.0]

TABLE_CONDITIONS = [
    ("Default (δ=1σ, σ_AI=0.3, h=0.5)",  {}),
    ("No misalignment (δ=0)",             {"delta": 0.0}),
    ("δ = 0.5σ",                          {"delta": 0.5}),
    ("δ = 2.0σ",                          {"delta": 2.0}),
    ("Low AI variance (σ_AI=0.1)",        {"sigma_ai": 0.1}),
    ("High AI variance (σ_AI=1.0)",       {"sigma_ai": 1.0}),
    ("Cancellation (σ_AI=2.0)",           {"sigma_ai": 2.0}),
    ("SSL clustering (h=0.8)",            {"homophily": 0.80}),
    ("SSL bridging (h=0.2)",              {"homophily": 0.20}),
    ("Low adoption (p=0.20)",             {"p_chatbot": 0.20}),
    ("High adoption (p=0.90)",            {"p_chatbot": 0.90}),
    ("1 AI model",                        {"n_models": 1}),
    ("10 AI models",                      {"n_models": 10}),
]


###############################################################################
# Runner helpers
###############################################################################

def _rc(overrides, seed, t_max=None):
    return sim.run_cond(overrides, n_reps=sim.N_REPS, base_seed=seed, t_max=t_max)

def _rc_beta(bv, seed, t_max=None):
    """run_cond with BETA_H temporarily set to bv."""
    old = sim.BETA_H
    sim.BETA_H = bv
    result = _rc({}, seed, t_max=t_max)
    sim.BETA_H = old
    return result


###############################################################################
# Main
###############################################################################

def generate():
    data = {}
    t0 = time.time()

    # --- Fig 1: default condition, full time series ---
    print("Fig 1 data …")
    data["fig1"] = _rc({}, seed=1)

    # --- Fig 2A: delta sweep, full time series ---
    print("Fig 2A data (delta sweep) …")
    data["fig2_delta"] = {dv: _rc({"delta": dv}, seed=10 + int(dv * 10))
                          for dv in DELTA_VALS}

    # --- Fig 2B: sigma_ai sweep at delta=1, full time series ---
    print("Fig 2B data (sigma sweep) …")
    data["fig2_sigma"] = {sv: _rc({"sigma_ai": sv, "delta": 1.0}, seed=20 + int(sv * 10))
                          for sv in SIGMA_VALS}

    # --- Fig 2C: heatmap at EVAL ---
    print("Fig 2C data (heatmap) …")
    hmap = np.zeros((len(DV_GRID), len(SV_GRID)))
    for i, dv in enumerate(DV_GRID):
        for j, sv in enumerate(SV_GRID):
            M, _ = _rc({"delta": dv, "sigma_ai": sv}, seed=30 + i * 10 + j, t_max=EVAL)
            hmap[i, j] = float(M[f"pop_drift_conventional"][EVAL])
    data["fig2_heatmap"] = {"dv_grid": DV_GRID, "sv_grid": SV_GRID, "hmap": hmap}

    # --- Fig 3: domain gradient, delta sweep at EVAL ---
    print("Fig 3 data (domain gradient) …")
    data["fig3"] = {dv: _rc({"delta": dv}, seed=40 + int(dv * 10), t_max=EVAL)
                    for dv in [0.5, 1.0, 2.0]}  # drop δ=0 (trivially zero)

    # --- Fig 4: homophily sweep, full time series ---
    print("Fig 4 data (homophily) …")
    data["fig4"] = {hv: _rc({"homophily": hv}, seed=50 + int(hv * 10))
                    for hv in HOM_VALS}

    # --- Fig S1: beta_h sweep at EVAL (all 5 values for ratio panel) ---
    print("Fig S1 data (beta sensitivity) …")
    data["figS1"] = {bv: _rc_beta(bv, seed=80 + int(bv * 100), t_max=EVAL)
                     for bv in BETA_VALS}

    # --- Table: all 13 conditions at EVAL ---
    print("Table data …")
    table_rows = []
    for label, ov in TABLE_CONDITIONS:
        M, _ = _rc(ov, seed=95, t_max=EVAL)
        d = "conventional"
        table_rows.append({
            "label": label,
            "pop":    round(float(M[f"pop_drift_{d}"][EVAL]), 1),
            "nonssl": round(float(M[f"nonssl_drift_{d}"][EVAL]), 1),
            "polar":  round(float(M[f"polarization_{d}"][EVAL]), 1),
        })
    data["table"] = table_rows

    out = HERE / "data.pkl"
    with open(out, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved {out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    generate()
