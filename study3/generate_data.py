"""
Author: Joshua Ashkinaze
Description: Runs all simulation conditions for Study 3 and saves results to data.pkl.
  Parallelized with multiprocessing — pass --workers N to control CPU count,
  --reps N to control replications per condition (default 100).
  plot.py reads data.pkl — no re-simulation needed.

Inputs:
  - simulation.py (engine)

Outputs:
  - data.pkl: dict of all precomputed (M, A) results keyed by figure/condition
"""

import argparse
import pickle
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm
import simulation as sim

HERE = Path(__file__).parent
EVAL = min(150, sim.T)


###############################################################################
# Condition sets
###############################################################################

NMODEL_VALS = [1, 3, 5, 10, 20]
PHI_VALS    = [0.0, 0.25, 0.5, 0.75, 1.0]
DELTA_VALS  = [0.0, 0.5, 1.0, 2.0]
SYCO_VALS   = [0.0, 0.25, 0.5, 0.75, 0.99]
HOM_VALS    = [0.2, 0.35, 0.5, 0.65, 0.8]
BETA_VALS       = [0.05, 0.10, 0.14, 0.20, 0.28]
BETA_AI_SCALES  = [0.5, 0.75, 1.0, 1.25, 1.5]   # multipliers on all BETA_AI values

NM_GRID  = [1, 3, 5, 10, 20]
PHI_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

TABLE_CONDITIONS = [
    (r"Default ($\delta=1$, $n=3$, $\varphi=0$, $s=0$, $h=0.5$)", {}),
    (r"No misalignment ($\delta=0$)",                               {"delta": 0.0}),
    (r"High misalignment ($\delta=2$)",                             {"delta": 2.0}),
    (r"Monopoly ($n=1$)",                                           {"n_models": 1}),
    (r"10 models ($n=10$)",                                         {"n_models": 10}),
    (r"20 models ($n=20$)",                                         {"n_models": 20}),
    (r"Identical models ($\varphi=1.0$)",                           {"phi": 1.0}),
    (r"Correlated models ($\varphi=0.5$)",                          {"phi": 0.5}),
    (r"10 models + identical ($n=10$, $\varphi=1$)",                {"n_models": 10, "phi": 1.0}),
    (r"10 models + independent ($n=10$, $\varphi=0$)",              {"n_models": 10, "phi": 0.0}),
    (r"Near-full sycophancy ($s=0.99$)",                            {"sycophancy": 0.99}),
    (r"Half sycophancy ($s=0.5$)",                                  {"sycophancy": 0.5}),
    (r"SSL clustering ($h=0.8$)",                                   {"homophily": 0.80}),
    (r"Low adoption ($p=0.20$)",                                    {"p_chatbot": 0.20}),
]


###############################################################################
# Worker (module-level so multiprocessing can pickle it)
###############################################################################

def _worker(args):
    """Run one condition. args = (overrides, seed, t_max, n_reps, beta_h)."""
    overrides, seed, t_max, n_reps, beta_h = args
    if beta_h is not None:
        sim.BETA_H = beta_h   # safe: each process has its own copy
    return sim.run_cond(overrides, n_reps=n_reps, base_seed=seed, t_max=t_max)


###############################################################################
# Job builder helpers
###############################################################################

def _job(overrides, seed, t_max=None, n_reps=None, beta_h=None):
    return (overrides, seed, t_max, n_reps, beta_h)


###############################################################################
# Main
###############################################################################

def generate(n_reps: int, workers: int):
    t0 = time.time()
    print(f"n_reps={n_reps}  workers={workers}  EVAL={EVAL}")

    # Build flat job list: (key, job_args)
    # Keys are used to reassemble results after parallel execution.
    jobs = []   # list of (result_key, job_args)

    jobs.append(("fig1",        _job({},                  seed=1,   n_reps=n_reps)))
    jobs.append(("fig1_noai",   _job({"p_chatbot": 0.0},  seed=200, n_reps=n_reps)))

    for nm in NMODEL_VALS:
        jobs.append((f"fig2_nmodels/{nm}", _job({"n_models": nm}, seed=10+nm, n_reps=n_reps)))

    for pv in PHI_VALS:
        jobs.append((f"fig2_phi/{pv}", _job({"phi": pv}, seed=20+int(pv*10), n_reps=n_reps)))

    for i, pv in enumerate(PHI_GRID):
        for j, nm in enumerate(NM_GRID):
            jobs.append((f"fig2_heatmap/{i}/{j}",
                         _job({"phi": pv, "n_models": nm}, seed=30+i*10+j,
                              t_max=EVAL, n_reps=n_reps)))

    for dv in DELTA_VALS:
        jobs.append((f"fig2_delta/{dv}", _job({"delta": dv}, seed=25+int(dv*10), n_reps=n_reps)))

    for sv in SYCO_VALS:
        jobs.append((f"fig2_syco/{sv}", _job({"sycophancy": sv}, seed=60+int(sv*10), n_reps=n_reps)))

    for sv in SYCO_VALS:
        for nm in NMODEL_VALS:
            jobs.append((f"fig5_syco_nm/{sv}/{nm}",
                         _job({"sycophancy": sv, "n_models": nm},
                              seed=70+int(sv*10)+nm, t_max=EVAL, n_reps=n_reps)))

    for nm in [1, 3, 10]:
        jobs.append((f"fig3/{nm}", _job({"n_models": nm}, seed=40+nm, t_max=EVAL, n_reps=n_reps)))

    for hv in HOM_VALS:
        jobs.append((f"fig4/{hv}", _job({"homophily": hv}, seed=50+int(hv*10), n_reps=n_reps)))

    for bv in BETA_VALS:
        jobs.append((f"figS1/{bv}", _job({}, seed=80+int(bv*100), t_max=EVAL,
                                         n_reps=n_reps, beta_h=bv)))

    # Robustness: sweep beta_h across key findings (opt-out ratio, n_models effect)
    for bv in BETA_VALS:
        for nm in NMODEL_VALS:
            jobs.append((f"figS2_bh/{bv}/{nm}",
                         _job({"n_models": nm}, seed=100+int(bv*100)+nm,
                              t_max=EVAL, n_reps=n_reps, beta_h=bv)))

    # Robustness: sweep beta_ai_scale across key findings
    for sc in BETA_AI_SCALES:
        jobs.append((f"figS2_bai/{sc}",
                     _job({"beta_ai_scale": sc}, seed=200+int(sc*100),
                          t_max=EVAL, n_reps=n_reps)))
    for sc in BETA_AI_SCALES:
        for nm in NMODEL_VALS:
            jobs.append((f"figS2_bai_nm/{sc}/{nm}",
                         _job({"beta_ai_scale": sc, "n_models": nm},
                              seed=300+int(sc*100)+nm, t_max=EVAL, n_reps=n_reps)))

    for idx, (label, ov) in enumerate(TABLE_CONDITIONS):
        jobs.append((f"table/{idx}", _job(ov, seed=95, t_max=EVAL, n_reps=n_reps)))

    keys_list = [k for k, _ in jobs]
    args_list = [a for _, a in jobs]

    print(f"Total conditions: {len(jobs)}")

    with Pool(workers) as pool:
        results = list(tqdm(pool.imap(_worker, args_list, chunksize=1),
                            total=len(args_list), desc="conditions"))

    # Reassemble
    raw = dict(zip(keys_list, results))

    data = {}

    data["fig1"]      = raw["fig1"]
    data["fig1_noai"] = raw["fig1_noai"]

    data["fig2_nmodels"] = {nm: raw[f"fig2_nmodels/{nm}"] for nm in NMODEL_VALS}
    data["fig2_phi"]     = {pv: raw[f"fig2_phi/{pv}"]     for pv in PHI_VALS}
    data["fig2_delta"]   = {dv: raw[f"fig2_delta/{dv}"]   for dv in DELTA_VALS}
    data["fig2_syco"]    = {sv: raw[f"fig2_syco/{sv}"]    for sv in SYCO_VALS}

    hmap = np.zeros((len(PHI_GRID), len(NM_GRID)))
    for i, pv in enumerate(PHI_GRID):
        for j, nm in enumerate(NM_GRID):
            _, A = raw[f"fig2_heatmap/{i}/{j}"]
            hmap[i, j] = float(A["pop_drift_conventional"][EVAL])
    data["fig2_heatmap"] = {"nm_grid": NM_GRID, "phi_grid": PHI_GRID, "hmap": hmap}

    data["fig5_syco_nm"] = {(sv, nm): raw[f"fig5_syco_nm/{sv}/{nm}"]
                            for sv in SYCO_VALS for nm in NMODEL_VALS}

    data["fig3"]  = {nm: raw[f"fig3/{nm}"]   for nm in [1, 3, 10]}
    data["fig4"]  = {hv: raw[f"fig4/{hv}"]   for hv in HOM_VALS}
    data["figS1"] = {bv: raw[f"figS1/{bv}"]  for bv in BETA_VALS}

    # Robustness sweeps
    data["figS2_bh"]     = {(bv, nm): raw[f"figS2_bh/{bv}/{nm}"]
                            for bv in BETA_VALS for nm in NMODEL_VALS}
    data["figS2_bai"]    = {sc: raw[f"figS2_bai/{sc}"]  for sc in BETA_AI_SCALES}
    data["figS2_bai_nm"] = {(sc, nm): raw[f"figS2_bai_nm/{sc}/{nm}"]
                            for sc in BETA_AI_SCALES for nm in NMODEL_VALS}

    table_rows = []
    for idx, (label, _) in enumerate(TABLE_CONDITIONS):
        _, A = raw[f"table/{idx}"]
        d = "conventional"
        table_rows.append({
            "label":      label,
            "pop_abs":    round(float(A[f"pop_drift_{d}"][EVAL]), 1),
            "nonssl_abs": round(float(A[f"nonssl_drift_{d}"][EVAL]), 1),
            "polar_abs":  round(float(A[f"polarization_{d}"][EVAL]), 1),
        })
    data["table"] = table_rows

    out = HERE / "data.pkl"
    with open(out, "wb") as f:
        pickle.dump(data, f)
    elapsed = time.time() - t0
    print(f"\nSaved {out}  ({elapsed:.1f}s  /  {elapsed/60:.1f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps",    type=int, default=200,
                        help="Replications per condition (default 100; use 500+ for publication)")
    parser.add_argument("--workers", type=int, default=cpu_count(),
                        help=f"Parallel workers (default: all CPUs = {cpu_count()})")
    args = parser.parse_args()
    generate(n_reps=args.reps, workers=args.workers)
