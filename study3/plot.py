"""
Author: Joshua Ashkinaze
Description: Produces all Study 3 figures and results_auto.tex from precomputed data.pkl.
  Run generate_data.py first, then iterate on this file freely.

  Under the bidirectional AI model, the primary outcome is S["pop_drift_*"] —
  the SD of drift across replications — measuring how much AI destabilizes beliefs.
  Mean drift (M) is near zero by design and is not the main quantity of interest.

Inputs:
  - data.pkl (from generate_data.py)

Outputs:
  - fig1_optout.pdf/png
  - fig2_drivers.pdf/png
  - fig2_summary.pdf/png
  - fig3_domains.pdf/png
  - fig4_homophily.pdf/png
  - figS1_beta_sens.pdf/png
  - results_auto.tex
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns

from simulation import DOMAINS, DLABELS, P_DOMAIN, DEFAULT, T

HERE    = Path(__file__).parent
FIG_DIR = HERE
EVAL    = min(150, T)


###############################################################################
# Aesthetic
###############################################################################

PALETTE = [
    "#00A896",  # teal
    "#826AED",  # purple
    "#E3B505",  # yellow
    "#89DAFF",  # cyan
    "#F45B69",  # red
    "#F18805",  # orange
    "#D41876",  # magenta
    "#020887",  # blue
]

DCOLORS = {"moral": PALETTE[0], "personal": PALETTE[1], "conventional": PALETTE[2]}


def make_aesthetic():
    sns.set(style="white", context="paper")
    sns.set_palette(sns.color_palette(PALETTE))
    small, medium = 12, 15
    plt.rcParams.update({
        "font.family":                   "Arial",
        "font.weight":                   "regular",
        "axes.labelsize":                medium,
        "axes.titlesize":                medium,
        "xtick.labelsize":               small,
        "ytick.labelsize":               small,
        "legend.fontsize":               small,
        "axes.titlecolor":               "#424242",
        "text.color":                    "#424242",
        "xtick.labelcolor":              "#424242",
        "ytick.labelcolor":              "#424242",
        "axes.spines.right":             False,
        "axes.spines.top":               False,
        "axes.linewidth":                0.8,
        "axes.grid":                     False,
        "axes.titlelocation":            "left",
        "axes.titleweight":              "regular",
        "axes.titlepad":                 12,
        "figure.facecolor":              "white",
        "axes.facecolor":                "white",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.3,
        "figure.constrained_layout.w_pad": 0.3,
        "legend.frameon":                True,
        "legend.framealpha":             0.95,
        "legend.facecolor":              "white",
        "legend.borderpad":              0.4,
        "legend.handlelength":           1.5,
        "savefig.dpi":                   300,
        "savefig.transparent":           True,
        "savefig.bbox":                  "tight",
        "savefig.pad_inches":            0.2,
        "figure.autolayout":             False,
    })


###############################################################################
# Helpers
###############################################################################

def shade(ax, m, s, color, label=None, lw=2.2, alpha=0.12, ls="-"):
    """Plot mean line with ±1 SD band."""
    x = np.arange(len(m))
    ax.fill_between(x, m - s, m + s, color=color, alpha=alpha)
    ax.plot(x, m, color=color, lw=lw, label=label, ls=ls)


def seq_colors(n, base, lo=0.35, hi=1.0):
    """Light-to-dark single-hue ramp with n stops."""
    r, g, b = mc.to_rgb(base)
    return [(r*f + 1*(1-f), g*f + 1*(1-f), b*f + 1*(1-f))
            for f in np.linspace(lo, hi, n)]


def savefig(fig, name):
    p = FIG_DIR / f"{name}.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(str(p).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf")


###############################################################################
# Figures
###############################################################################

def fig1_optout(data):
    """Non-users exposed to same belief variability as users."""
    make_aesthetic()
    _, A     = data["fig1"]
    _, A_0   = data["fig1_noai"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title("Non-Users Are Exposed As Much As Users", fontsize=14, pad=12)

    # No-AI baseline: average mean|shift| across domains (should be ~0)
    noai_avg = np.mean([A_0[f"pop_drift_{d}"] for d in DOMAINS], axis=0)
    ax.plot(np.arange(T + 1), noai_avg, color="#aaaaaa", lw=1.4, ls=":",
            label="No AI (baseline)", zorder=1)

    for d in DOMAINS:
        ax.plot(np.arange(T + 1), A[f"pop_drift_{d}"],
                color=DCOLORS[d], lw=2.2)
        ax.plot(np.arange(T + 1), A[f"nonssl_drift_{d}"],
                color=DCOLORS[d], lw=1.8, ls="--", alpha=0.9)

    ax.axhline(0, color="#ccc", lw=0.8)
    ax.set_xlabel("Period")
    ax.set_ylabel("mean |shift|")

    legend_handles = []
    for d in DOMAINS:
        pct = int(P_DOMAIN[d] * DEFAULT["p_chatbot"] * 100)
        legend_handles.append(
            plt.Line2D([0], [0], color=DCOLORS[d],
                       label=f"{DLABELS[d]} ({pct}% SSL)"))

    style_handles = [
        plt.Line2D([0], [0], color="#555", lw=2.2, ls="-",  label="All agents (with AI)"),
        plt.Line2D([0], [0], color="#555", lw=1.8, ls="--", label="Non-users (with AI)"),
        plt.Line2D([0], [0], color="#aaa", lw=1.4, ls=":",  label="No AI (baseline)"),
    ]
    fig.legend(handles=legend_handles + style_handles, fontsize=10,
               loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3,
               title="Domain (SSL %)", title_fontsize=9)

    savefig(fig, "fig1_optout")


def fig2_drivers(data):
    """Market structure drives belief shift: n_models, phi, misalignment.

    Layout: A. n_models sweep | B. phi sweep | C. heatmap | D. misalignment sweep
    Sycophancy has its own dedicated figure (fig5).
    """
    make_aesthetic()
    nmodel_vals = [1, 3, 5, 10, 20]
    phi_vals    = [0.0, 0.25, 0.5, 0.75, 1.0]
    delta_vals  = [0.0, 0.5, 1.0, 2.0]
    d           = "conventional"

    nm_cols    = [PALETTE[4], PALETTE[5], PALETTE[2], PALETTE[0], PALETTE[1]]
    phi_cols   = [PALETTE[0], PALETTE[2], PALETTE[5], PALETTE[6], PALETTE[4]]
    delta_cols = ["#aaaaaa", PALETTE[1], PALETTE[0], PALETTE[4]]

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))

    def _panel_letter(ax, letter):
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    # Panel A: n_models sweep
    nm_handles = []
    for nm, color in zip(nmodel_vals, nm_cols):
        _, A = data["fig2_nmodels"][nm]
        axes[0].plot(np.arange(T + 1), A[f"pop_drift_{d}"], color=color, lw=2.2)
        nm_handles.append(plt.Line2D([0],[0], color=color, lw=2.2, label=f"n={nm}"))
    axes[0].axhline(0, color="#ccc", lw=0.8)
    axes[0].set_xlabel("Period")
    axes[0].set_ylabel("mean |shift|")
    axes[0].set_title("More models → less societal shift", fontsize=11, pad=10)
    axes[0].legend(handles=nm_handles, title="n models (φ=0)", title_fontsize=9,
                   fontsize=8, bbox_to_anchor=(0, -0.18), loc="upper left",
                   borderaxespad=0, ncol=3)
    _panel_letter(axes[0], "A")

    # Panel B: phi sweep
    phi_handles = []
    for pv, color in zip(phi_vals, phi_cols):
        _, A = data["fig2_phi"][pv]
        axes[1].plot(np.arange(T + 1), A[f"pop_drift_{d}"], color=color, lw=2.2)
        phi_handles.append(plt.Line2D([0],[0], color=color, lw=2.2, label=f"φ={pv}"))
    axes[1].axhline(0, color="#ccc", lw=0.8)
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("mean |shift|")
    axes[1].set_title("Correlated models → more societal shift", fontsize=11, pad=10)
    axes[1].legend(handles=phi_handles, title="Inter-model corr. (n=3)", title_fontsize=9,
                   fontsize=8, bbox_to_anchor=(0, -0.18), loc="upper left",
                   borderaxespad=0, ncol=3)
    _panel_letter(axes[1], "B")

    # Panel C: heatmap n_models × phi
    hd = data["fig2_heatmap"]
    hmap, nm_grid, phi_grid = hd["hmap"], hd["nm_grid"], hd["phi_grid"]
    im = axes[2].imshow(hmap, cmap="Blues", aspect="auto", vmin=0)
    axes[2].set_xticks(range(len(nm_grid)))
    axes[2].set_xticklabels([str(v) for v in nm_grid], fontsize=10)
    axes[2].set_yticks(range(len(phi_grid)))
    axes[2].set_yticklabels([str(v) for v in phi_grid], fontsize=10)
    axes[2].set_xlabel("No. of AI models")
    axes[2].set_ylabel("Inter-model correlation (φ)")
    axes[2].set_title("Shift by n models × correlation", fontsize=11, pad=10)
    vmax = hmap.max()
    for i in range(len(phi_grid)):
        for j in range(len(nm_grid)):
            v  = hmap[i, j]
            tc = "white" if v > vmax * 0.6 else "#424242"
            axes[2].text(j, i, f"{v:.1f}", ha="center", va="center",
                         fontsize=10, color=tc)
    plt.colorbar(im, ax=axes[2], shrink=0.85, label="mean |shift| at t=150")
    _panel_letter(axes[2], "C")

    # Panel D: delta (misalignment) sweep
    delta_handles = []
    for dv, color in zip(delta_vals, delta_cols):
        _, A = data["fig2_delta"][dv]
        axes[3].plot(np.arange(T + 1), A[f"pop_drift_{d}"], color=color, lw=2.2)
        delta_handles.append(plt.Line2D([0],[0], color=color, lw=2.2, label=f"δ={dv}σ"))
    axes[3].axhline(0, color="#ccc", lw=0.8)
    axes[3].set_xlabel("Period")
    axes[3].set_ylabel("mean |shift|")
    axes[3].set_title("Misalignment → more societal shift", fontsize=11, pad=10)
    axes[3].legend(handles=delta_handles, title="Misalignment (δ)", title_fontsize=9,
                   fontsize=8, bbox_to_anchor=(0, -0.18), loc="upper left",
                   borderaxespad=0, ncol=2)
    _panel_letter(axes[3], "D")

    fig.subplots_adjust(wspace=0.38)
    savefig(fig, "fig2_drivers")


def fig3_domains(data):
    """Higher SSL prevalence → more belief variability: domain on x-axis."""
    make_aesthetic()
    nmodel_vals = [1, 3, 10]
    prevalences = {"moral": 10, "personal": 21, "conventional": 31}
    x           = np.arange(len(DOMAINS))

    nm_cols  = [PALETTE[4], PALETTE[0], PALETTE[1]]  # red=monopoly, teal=default, purple=10
    nm_marks = ["o", "s", "D"]
    nm_labs  = ["n=1 (monopoly)", "n=3 (default)", "n=10"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    # collect all values first to set shared y-axis range
    all_vals = []
    for metric in ["pop_drift", "nonssl_drift"]:
        for nm in nmodel_vals:
            _, A = data["fig3"][nm]
            all_vals.extend([float(A[f"{metric}_{d}"][EVAL]) for d in DOMAINS])
    y_max = max(all_vals) * 1.12

    jitter = [-0.12, 0.0, 0.12]  # x-offset per n_models level

    for metric, ax, title in [
        ("pop_drift",    axes[0], "All agents"),
        ("nonssl_drift", axes[1], "Non-users only"),
    ]:
        for (nm, color, marker, lab), jit in zip(
            zip(nmodel_vals, nm_cols, nm_marks, nm_labs), jitter
        ):
            _, A = data["fig3"][nm]
            vals = [float(A[f"{metric}_{d}"][EVAL]) for d in DOMAINS]
            ax.scatter(x + jit, vals, color=color, marker=marker, s=90,
                       label=lab, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{DLABELS[d]}\n({prevalences[d]}% SSL)" for d in DOMAINS], fontsize=11)
        ax.set_ylim(0, y_max)
        ax.set_ylabel("mean |shift| at t=150")
        ax.set_title(title, fontsize=13, pad=10)
        ax.axhline(0, color="#ccc", lw=0.8)

    leg_handles = [plt.Line2D([0],[0], color=c, marker=m, lw=1.5, ms=8, label=l)
                   for c, m, l in zip(nm_cols, nm_marks, nm_labs)]
    fig.legend(handles=leg_handles, title="No. of AI models", title_fontsize=10,
               fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3)

    savefig(fig, "fig3_domains")


def fig4_homophily(data):
    """Homophily modulates user/non-user divergence across all three groups."""
    make_aesthetic()
    hom_vals = [0.2, 0.35, 0.5, 0.65, 0.8]
    hom_labs = ["h=0.2", "h=0.35", "h=0.5", "h=0.65", "h=0.8"]
    # Discrete blue→gray→red palette: easy to distinguish at a glance
    h_cols = ["#2166AC", "#74ADD1", "#AAAAAA", "#F4A582", "#D6604D"]

    groups = [
        ("pop_drift",    "-",  "All agents"),
        ("ssl_drift",    "--", "AI users"),
        ("nonssl_drift", ":",  "Non-AI users"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    for ax, (metric, ls, group_title) in zip(axes, groups):
        for hv, color, hlabel in zip(hom_vals, h_cols, hom_labs):
            _, A = data["fig4"][hv]
            avg = np.mean([A[f"{metric}_{d}"] for d in DOMAINS], axis=0)
            ax.plot(np.arange(T + 1), avg, color=color, lw=2.2, ls=ls, label=hlabel)
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.set_xlabel("Period")
        ax.set_title(group_title, fontsize=12, pad=10)

    axes[0].set_ylabel("mean |shift|")

    legend_handles = [plt.Line2D([0], [0], color=c, lw=2.2, label=l)
                      for c, l in zip(h_cols, hom_labs)]
    fig.legend(handles=legend_handles,
               loc="lower center", bbox_to_anchor=(0.5, -0.18),
               ncol=5, fontsize=10,
               title="Network homophily (h=0.5 = random mixing)", title_fontsize=9)

    savefig(fig, "fig4_homophily")


def figS1_beta_sensitivity(data):
    """Finding 1 holds across all tested social influence weights."""
    make_aesthetic()
    beta_vals = [0.05, 0.14, 0.28]
    beta_all  = [0.05, 0.10, 0.14, 0.20, 0.28]
    b_cols    = [PALETTE[3], PALETTE[0], PALETTE[4]]  # cyan, teal, red
    b_labs    = ["β_h = 0.05 (low)", "β_h = 0.14 (mid)", "β_h = 0.28 (Study 2)"]
    d         = "conventional"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={"width_ratios": [1.6, 1]})

    axes[0].set_title("Opt-out offers no protection at any β_h", fontsize=12, pad=10)
    for bv, color, label in zip(beta_vals, b_cols, b_labs):
        _, A = data["figS1"][bv]
        axes[0].plot(np.arange(EVAL + 1), A[f"pop_drift_{d}"],
                     color=color, lw=2.2, label=label)
        axes[0].plot(np.arange(EVAL + 1), A[f"nonssl_drift_{d}"],
                     color=color, lw=1.6, ls="--", alpha=0.9)

    axes[0].axhline(0, color="#ccc", lw=0.8)
    axes[0].set_xlabel("Period")
    axes[0].set_ylabel("mean |shift|")

    color_handles = [plt.Line2D([0],[0], color=c, lw=2.2, label=l)
                     for c, l in zip(b_cols, b_labs)]
    style_note = [plt.Line2D([0],[0], color="#555", lw=2.2, ls="-",  label="All agents"),
                  plt.Line2D([0],[0], color="#555", lw=1.6, ls="--", label="Non-users")]
    fig.legend(handles=color_handles + style_note, fontsize=10,
               loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=5)

    # Panel B: ratio non-user mean|shift| / pop mean|shift| at EVAL
    axes[1].set_title("Non-user/user ratio near 1.0 across all β_h", fontsize=12, pad=10)
    equil_pop    = [float(data["figS1"][bv][1][f"pop_drift_{d}"][EVAL])    for bv in beta_all]
    equil_nonssl = [float(data["figS1"][bv][1][f"nonssl_drift_{d}"][EVAL]) for bv in beta_all]
    ratios = [ns/pp if pp > 0.5 else 0. for ns, pp in zip(equil_nonssl, equil_pop)]

    axes[1].axhline(1.0, color="#424242", lw=1.4, ls="--", zorder=1)
    axes[1].text(4.5, 1.004, "Perfect tracking", fontsize=9, color="#424242", va="bottom")
    axes[1].plot([bv*100 for bv in beta_all], ratios,
                 "o-", color=PALETTE[0], lw=2, ms=8, zorder=2)
    for bv, r in zip(beta_all, ratios):
        axes[1].annotate(f"{r:.2f}", (bv*100, r),
                         textcoords="offset points", xytext=(0, 10),
                         fontsize=10, ha="center", color="#424242")
    axes[1].set_xlabel("Social influence weight β_h (×100)")
    axes[1].set_ylabel("Non-user / all-agent mean|shift| ratio\n(axis starts at 0.9)")
    axes[1].set_ylim(0.90, 1.05)
    axes[1].set_xlim(0, 32)

    savefig(fig, "figS1_beta_sens")


def figS2_robustness(data):
    """Robustness: key findings hold across beta_h and beta_ai_scale values.

    Panel A: mean|shift| at t=150 vs n_models, one line per beta_h value.
    Panel B: mean|shift| at t=150 vs n_models, one line per beta_ai_scale value.
    Panel C: opt-out ratio (non-user / pop drift) vs beta_ai_scale.
    All panels: conventional domain. Lines going down = more models → less drift (directional finding).
    """
    make_aesthetic()
    d          = "conventional"
    nm_vals    = [1, 3, 5, 10, 20]
    beta_vals  = [0.05, 0.10, 0.14, 0.20, 0.28]
    ai_scales  = [0.5, 0.75, 1.0, 1.25, 1.5]

    bh_cols = seq_colors(5, PALETTE[0], lo=0.25, hi=1.0)
    ai_cols = seq_colors(5, PALETTE[4], lo=0.25, hi=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # Panel A: n_models effect across beta_h values
    bh_handles = []
    for bv, color in zip(beta_vals, bh_cols):
        ys = [float(data["figS2_bh"][(bv, nm)][1][f"pop_drift_{d}"][EVAL])
              for nm in nm_vals]
        axes[0].plot(nm_vals, ys, color=color, lw=2.0, marker="o", ms=5)
        bh_handles.append(plt.Line2D([0],[0], color=color, lw=2.0,
                                     label=f"β_h={bv}"))
    axes[0].axhline(0, color="#ccc", lw=0.8)
    axes[0].set_xlabel("No. of AI models")
    axes[0].set_ylabel("mean |shift| at t=150")
    axes[0].set_title("More models → less drift: holds across β_h", fontsize=11, pad=10)
    axes[0].legend(handles=bh_handles, title="β_h", title_fontsize=9, fontsize=8)

    # Panel B: n_models effect across beta_ai_scale values
    ai_handles = []
    for sc, color in zip(ai_scales, ai_cols):
        ys = [float(data["figS2_bai_nm"][(sc, nm)][1][f"pop_drift_{d}"][EVAL])
              for nm in nm_vals]
        axes[1].plot(nm_vals, ys, color=color, lw=2.0, marker="o", ms=5)
        ai_handles.append(plt.Line2D([0],[0], color=color, lw=2.0,
                                     label=f"×{sc}"))
    axes[1].axhline(0, color="#ccc", lw=0.8)
    axes[1].set_xlabel("No. of AI models")
    axes[1].set_ylabel("mean |shift| at t=150")
    axes[1].set_title("More models → less drift: holds across β_AI scale", fontsize=11, pad=10)
    axes[1].legend(handles=ai_handles, title="β_AI scale", title_fontsize=9, fontsize=8)

    # Panel C: opt-out ratio across beta_ai_scale
    ratios = []
    for sc in ai_scales:
        _, A = data["figS2_bai"][sc]
        pop    = float(A[f"pop_drift_{d}"][EVAL])
        nonssl = float(A[f"nonssl_drift_{d}"][EVAL])
        ratios.append(nonssl / pop if pop > 0.5 else 0.)
    axes[2].plot([f"×{s}" for s in ai_scales], ratios,
                 "o-", color=PALETTE[0], lw=2.2, ms=8)
    for sc, r in zip(ai_scales, ratios):
        axes[2].annotate(f"{r:.2f}", (f"×{sc}", r),
                         textcoords="offset points", xytext=(0, 8),
                         fontsize=10, ha="center", color="#424242")
    axes[2].axhline(1.0, color="#424242", lw=1.2, ls="--")
    axes[2].set_ylim(0.85, 1.05)
    axes[2].set_xlabel("β_AI scale factor")
    axes[2].set_ylabel("Non-user / all-agent ratio at t=150")
    axes[2].set_title("Non-users track users: holds across β_AI scale", fontsize=11, pad=10)

    savefig(fig, "figS2_robustness")


###############################################################################
# Results table
###############################################################################

def fig5_sycophancy(data):
    """Sycophancy slows but does not stop belief drift — a pure speed effect.

    Panel A: full time series — all curves converge to the same level.
    Panel B: zoom t=0–40 — curves separate early, showing the speed difference.
    Panel C: periods to reach 90% of max drift — the speed effect quantified.
    """
    make_aesthetic()
    syco_vals   = [0.0, 0.25, 0.5, 0.75, 0.99]
    d           = "conventional"
    syco_cols   = seq_colors(5, PALETTE[4], lo=0.25, hi=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    s_handles  = []
    t90_vals   = []
    for sv, color in zip(syco_vals, syco_cols):
        _, A = data["fig2_syco"][sv]
        ts   = A[f"pop_drift_{d}"]
        axes[0].plot(np.arange(T + 1), ts, color=color, lw=2.2)
        axes[1].plot(np.arange(41),     ts[:41], color=color, lw=2.2)
        s_handles.append(plt.Line2D([0],[0], color=color, lw=2.2, label=f"s={sv}"))
        # Periods to reach 90% of the series maximum
        peak    = ts.max()
        crosses = np.where(ts >= 0.9 * peak)[0]
        t90_vals.append(int(crosses[0]) if len(crosses) else T)

    for ax, title, xlabel in [
        (axes[0], "Same destination, different speed", "Period"),
        (axes[1], "Early periods: curves separate",   "Period (zoom t≤40)"),
    ]:
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mean |shift|")
        ax.set_title(title, fontsize=12, pad=10)

    axes[0].legend(handles=s_handles, title="Sycophancy (s)", title_fontsize=9,
                   fontsize=9, loc="upper left")

    # Panel C: periods to 90% of max drift vs s
    axes[2].plot(syco_vals, t90_vals, "o-", color=PALETTE[4], lw=2.2, ms=8)
    for sv, t90 in zip(syco_vals, t90_vals):
        axes[2].annotate(str(t90), (sv, t90),
                         textcoords="offset points", xytext=(0, 8),
                         fontsize=10, ha="center", color="#424242")
    axes[2].set_xlabel("Sycophancy (s)")
    axes[2].set_ylabel("Periods to 90% of max drift")
    axes[2].set_title("Sycophancy delays but cannot prevent drift", fontsize=12, pad=10)
    axes[2].set_xlim(-0.05, 1.05)

    savefig(fig, "fig5_sycophancy")


def fig6_variance_polarization(data):
    """New DVs: belief variance and AI-user vs non-user polarization.

    Panel A: Population belief variance over time, by n_models (market structure → variance).
    Panel B: AI-user polarization (mean_ssl - mean_nonssl) over time, by homophily.
    Panel C: Variance over time by sycophancy level — does sycophancy affect spread?
    """
    make_aesthetic()
    d         = "conventional"
    syco_vals = [0.0, 0.25, 0.5, 0.75, 0.99]
    hom_vals  = [0.2, 0.35, 0.5, 0.65, 0.8]
    nm_vals   = [1, 3, 5, 10, 20]

    syco_cols = seq_colors(5, PALETTE[4], lo=0.25, hi=1.0)
    h_cols    = ["#2166AC", "#74ADD1", "#AAAAAA", "#F4A582", "#D6604D"]
    nm_cols   = [PALETTE[4], PALETTE[5], PALETTE[2], PALETTE[0], PALETTE[1]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # Panel A: variance by n_models
    for nm, color in zip(nm_vals, nm_cols):
        _, A = data["fig2_nmodels"][nm]
        axes[0].plot(np.arange(T + 1), A[f"pop_var_{d}"],
                     color=color, lw=2.2, label=f"n={nm}")
    axes[0].axhline(0, color="#ccc", lw=0.8)
    axes[0].set_xlabel("Period")
    axes[0].set_ylabel("Belief variance (0–100 scale)")
    axes[0].set_title("A monopoly homogenizes beliefs; more models preserve spread", fontsize=11, pad=10)
    axes[0].legend(title="n models (φ=0)", title_fontsize=9, fontsize=8)

    # Panel B: AI-user polarization by homophily
    for hv, color in zip(hom_vals, h_cols):
        _, A = data["fig4"][hv]
        avg_polar = np.mean([A[f"polarization_{dom}"] for dom in DOMAINS], axis=0)
        axes[1].plot(np.arange(T + 1), avg_polar, color=color, lw=2.2, label=f"h={hv}")
    axes[1].axhline(0, color="#ccc", lw=0.8)
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("AI users − Non-AI users (pts)")
    axes[1].set_title("Homophily (h) slows diffusion but not long-run divergence", fontsize=11, pad=10)
    axes[1].legend(title="Homophily", title_fontsize=9, fontsize=8)

    # Panel C: variance by sycophancy
    for sv, color in zip(syco_vals, syco_cols):
        _, A = data["fig2_syco"][sv]
        axes[2].plot(np.arange(T + 1), A[f"pop_var_{d}"],
                     color=color, lw=2.2, label=f"s={sv}")
    axes[2].axhline(0, color="#ccc", lw=0.8)
    axes[2].set_xlabel("Period")
    axes[2].set_ylabel("Belief variance (0–100 scale)")
    axes[2].set_title("Sycophancy reduces belief variance", fontsize=11, pad=10)
    axes[2].legend(title="Sycophancy (s)", title_fontsize=9, fontsize=8)

    savefig(fig, "fig6_variance_polarization")


def write_results_tex(data):
    rows = data["table"]
    df = pd.DataFrame([{
        "Condition":              r["label"],
        "Pop. mean|shift| (pts)": r["pop_abs"],
        "Non-User mean|shift|":   r["nonssl_abs"],
        "Polarization mean|shift|": r["polar_abs"],
    } for r in rows])
    latex = df.to_latex(
        index=False,
        escape=False,
        caption=(
            r"Mean absolute belief shift across 100 replications at period 150, conventional domain. "
            r"Each replication draws fresh model centers from "
            r"$\mathcal{N}(\mu_h,\,(\delta\sigma_h)^2)$. "
            r"$\beta_{\mathrm{AI}}=0.26$, $\beta_h=0.28$, $k=3$."
        ),
        label="tab:conditions",
        position="ht",
        column_format="lrrr",
    )
    (HERE / "results_auto.tex").write_text(latex)
    print("  saved results_auto.tex")


###############################################################################
# Accompanying tables
###############################################################################

def write_figure_tables(data):
    """Write tables_auto.tex with one summary table per figure."""
    d = "conventional"
    parts = []

    # ── Table 1: Fig 1 — non-user tracking by domain ──────────────────────
    rows1 = []
    _, A    = data["fig1"]
    _, A_0  = data["fig1_noai"]
    for dom in DOMAINS:
        pct = int(P_DOMAIN[dom] * DEFAULT["p_chatbot"] * 100)
        rows1.append({
            "Domain":            f"{DLABELS[dom]} ({pct}\\% SSL)",
            "All agents":        round(float(A[f"pop_drift_{dom}"][EVAL]), 1),
            "Non-users":         round(float(A[f"nonssl_drift_{dom}"][EVAL]), 1),
            "No-AI baseline":    round(float(A_0[f"pop_drift_{dom}"][EVAL]), 1),
        })
    df1 = pd.DataFrame(rows1)
    parts.append(df1.to_latex(
        index=False, escape=False,
        caption="Fig.~1: mean $|$shift$|$ at $t=150$ by domain (all agents, non-users, no-AI baseline).",
        label="tab:fig1", position="ht", column_format="lrrr",
    ))

    # ── Table 2: Fig 2 — market structure drivers ─────────────────────────
    col_shift = r"mean $|$shift$|$"
    rows2 = []
    for nm in [1, 3, 5, 10, 20]:
        _, A = data["fig2_nmodels"][nm]
        rows2.append({"Driver": r"$n_{\text{models}}=" + str(nm) + r"$",
                      col_shift: round(float(A[f"pop_drift_{d}"][EVAL]), 1)})
    for pv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        _, A = data["fig2_phi"][pv]
        rows2.append({"Driver": r"$\varphi=" + str(pv) + r"$",
                      col_shift: round(float(A[f"pop_drift_{d}"][EVAL]), 1)})
    for dv in [0.0, 0.5, 1.0, 2.0]:
        _, A = data["fig2_delta"][dv]
        rows2.append({"Driver": r"$\delta=" + str(dv) + r"$",
                      col_shift: round(float(A[f"pop_drift_{d}"][EVAL]), 1)})
    df2 = pd.DataFrame(rows2)
    parts.append(df2.to_latex(
        index=False, escape=False,
        caption="Fig.~2: mean $|$shift$|$ at $t=150$ across market-structure parameters (conventional domain). "
                "Each row varies one parameter; others held at default ($n=3$, $\\varphi=0$, $\\delta=1$).",
        label="tab:fig2", position="ht", column_format="lr",
    ))

    # ── Table 3: Fig 3 — domain gradient ──────────────────────────────────
    rows3 = []
    for nm in [1, 3, 10]:
        _, A = data["fig3"][nm]
        for dom in DOMAINS:
            pct = int(P_DOMAIN[dom] * DEFAULT["p_chatbot"] * 100)
            rows3.append({
                "Domain":     f"{DLABELS[dom]} ({pct}\\% SSL)",
                "n models":   nm,
                "All agents": round(float(A[f"pop_drift_{dom}"][EVAL]), 1),
                "Non-users":  round(float(A[f"nonssl_drift_{dom}"][EVAL]), 1),
            })
    df3 = pd.DataFrame(rows3)
    parts.append(df3.to_latex(
        index=False, escape=False,
        caption="Fig.~3: mean $|$shift$|$ at $t=150$ by domain and number of AI models.",
        label="tab:fig3", position="ht", column_format="lrrr",
    ))

    # ── Table 4: Fig 4 — homophily ────────────────────────────────────────
    rows4 = []
    for hv in [0.2, 0.35, 0.5, 0.65, 0.8]:
        _, A = data["fig4"][hv]
        pop_avg    = float(np.mean([A[f"pop_drift_{dom}"][EVAL]    for dom in DOMAINS]))
        ssl_avg    = float(np.mean([A[f"ssl_drift_{dom}"][EVAL]    for dom in DOMAINS]))
        nonssl_avg = float(np.mean([A[f"nonssl_drift_{dom}"][EVAL] for dom in DOMAINS]))
        rows4.append({
            "Homophily ($h$)": hv,
            "All agents":      round(pop_avg, 1),
            "AI users":        round(ssl_avg, 1),
            "Non-AI users":    round(nonssl_avg, 1),
        })
    df4 = pd.DataFrame(rows4)
    parts.append(df4.to_latex(
        index=False, escape=False,
        caption="Fig.~4: mean $|$shift$|$ at $t=150$ by homophily level, averaged across domains.",
        label="tab:fig4", position="ht", column_format="lrrr",
    ))

    # ── Table 5: Fig 5 — sycophancy ───────────────────────────────────────
    rows5 = []
    syco_vals = [0.0, 0.25, 0.5, 0.75, 0.99]
    for sv in syco_vals:
        _, A = data["fig2_syco"][sv]
        ts   = A[f"pop_drift_{d}"]
        peak    = ts.max()
        crosses = np.where(ts >= 0.9 * peak)[0]
        t90     = int(crosses[0]) if len(crosses) else T
        rows5.append({
            "Sycophancy ($s$)":        sv,
            "Endpoint mean $|$shift$|$": round(float(ts[EVAL]), 1),
            "Periods to 90\\% of max":  t90,
        })
    df5 = pd.DataFrame(rows5)
    parts.append(df5.to_latex(
        index=False, escape=False,
        caption="Fig.~5: sycophancy effect on drift speed. "
                "Endpoint drift is nearly identical across $s$ values; periods to 90\\% of max drift grows with $s$.",
        label="tab:fig5", position="ht", column_format="lrr",
    ))

    # ── Table 6: Fig 6 — variance and polarization ───────────────────────
    rows6 = []
    for nm in [1, 3, 5, 10, 20]:
        _, A = data["fig2_nmodels"][nm]
        rows6.append({
            r"$n_{\text{models}}$": nm,
            "Pop. variance":        round(float(A[f"pop_var_{d}"][EVAL]), 2),
            "mean $|$shift$|$":     round(float(A[f"pop_drift_{d}"][EVAL]), 1),
        })
    df6a = pd.DataFrame(rows6)
    parts.append(df6a.to_latex(
        index=False, escape=False,
        caption="Fig.~6A: belief variance and mean $|$shift$|$ at $t=150$ by number of AI models "
                "(conventional domain, $\\varphi=0$, $\\delta=1$).",
        label="tab:fig6a", position="ht", column_format="lrr",
    ))

    rows6b = []
    syco_vals6 = [0.0, 0.25, 0.5, 0.75, 0.99]
    for sv in syco_vals6:
        _, A = data["fig2_syco"][sv]
        rows6b.append({
            "Sycophancy ($s$)": sv,
            "Pop. variance":    round(float(A[f"pop_var_{d}"][EVAL]), 2),
            "mean $|$shift$|$": round(float(A[f"pop_drift_{d}"][EVAL]), 1),
        })
    df6b = pd.DataFrame(rows6b)
    parts.append(df6b.to_latex(
        index=False, escape=False,
        caption="Fig.~6C: belief variance and mean $|$shift$|$ at $t=150$ by sycophancy level "
                "(conventional domain, default market parameters).",
        label="tab:fig6b", position="ht", column_format="lrr",
    ))

    (HERE / "tables_auto.tex").write_text("\n\n".join(parts))
    print("  saved tables_auto.tex")


def fig2_summary(data):
    """Endpoint summary: mean|shift| vs. parameter value, 3 market-structure drivers."""
    make_aesthetic()
    d = "conventional"

    drivers = [
        {
            "label":    "No. of AI models",
            "vals":     [1, 3, 5, 10, 20],
            "getter":   lambda v: float(data["fig2_nmodels"][v][1][f"pop_drift_{d}"][EVAL]),
            "color":    PALETTE[0],
            "letter":   "A",
            "subtitle": "More models → less shift",
        },
        {
            "label":    "Inter-model correlation (φ)",
            "vals":     [0.0, 0.25, 0.5, 0.75, 1.0],
            "getter":   lambda v: float(data["fig2_phi"][v][1][f"pop_drift_{d}"][EVAL]),
            "color":    PALETTE[1],
            "letter":   "B",
            "subtitle": "Correlated models → more shift",
        },
        {
            "label":    "AI misalignment (δ)",
            "vals":     [0.0, 0.5, 1.0, 2.0],
            "getter":   lambda v: float(data["fig2_delta"][v][1][f"pop_drift_{d}"][EVAL]),
            "color":    PALETTE[4],
            "letter":   "C",
            "subtitle": "More misalignment → more shift",
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, dr in zip(axes, drivers):
        xs = dr["vals"]
        ys = [dr["getter"](v) for v in xs]
        ax.plot(xs, ys, color=dr["color"], lw=2.2, zorder=2)
        ax.scatter(xs, ys, color=dr["color"], s=60, zorder=3)
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.set_xlabel(dr["label"])
        ax.set_ylabel("mean |shift| at t=150")
        ax.set_title(dr["subtitle"], fontsize=11, pad=10)
        ax.text(-0.12, 1.05, dr["letter"], transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    fig.subplots_adjust(wspace=0.38)
    savefig(fig, "fig2_summary")


###############################################################################
# Main
###############################################################################

# Helper needed inside fig2_drivers without importing sim directly
from simulation import hub_rho, DEFAULT as _DEFAULT, N as _N
def sim_hub_rho(n_models):
    return hub_rho(_DEFAULT["p_chatbot"], "conventional", n_models, _DEFAULT.get("n", _N))


if __name__ == "__main__":
    pkl = HERE / "data.pkl"
    if not pkl.exists():
        raise FileNotFoundError("data.pkl not found — run generate_data.py first")

    with open(pkl, "rb") as f:
        data = pickle.load(f)

    fig1_optout(data)
    fig2_drivers(data)
    fig2_summary(data)
    fig3_domains(data)
    fig4_homophily(data)
    fig5_sycophancy(data)
    fig6_variance_polarization(data)
    figS1_beta_sensitivity(data)
    figS2_robustness(data)
    write_results_tex(data)
    write_figure_tables(data)
    print("Done.")
