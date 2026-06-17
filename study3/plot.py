"""
Author: Joshua Ashkinaze
Description: Produces all Study 3 figures and results_auto.tex from precomputed data.pkl.
  Run generate_data.py first, then iterate on this file freely — no re-simulation needed.

Inputs:
  - data.pkl (from generate_data.py)

Outputs:
  - fig1_optout.pdf/png
  - fig2_drivers.pdf/png
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
        "font.family": "Arial",
        "font.weight": "regular",
        "axes.labelsize": medium,
        "axes.titlesize": medium,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "legend.fontsize": small,
        "axes.titlecolor": "#424242",
        "text.color": "#424242",
        "xtick.labelcolor": "#424242",
        "ytick.labelcolor": "#424242",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.titlelocation": "left",
        "axes.titleweight": "regular",
        "axes.titlepad": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.3,
        "figure.constrained_layout.w_pad": 0.3,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "white",
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.5,
        "savefig.dpi": 300,
        "savefig.transparent": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "figure.autolayout": False,
    })


###############################################################################
# Plot helpers
###############################################################################

def shade(ax, m, s, color, label=None, lw=2.2, alpha=0.10, ls="-"):
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
    """Opt-out does not protect: solid (all) and dashed (non-users) track together."""
    make_aesthetic()
    M, S = data["fig1"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title("AI Influence Spreads Beyond AI Users", fontsize=14, pad=12)

    for d in DOMAINS:
        shade(ax, M[f"pop_drift_{d}"], S[f"pop_drift_{d}"], DCOLORS[d], lw=2.2, alpha=0.08)
        ax.plot(np.arange(T + 1), M[f"nonssl_drift_{d}"],
                color=DCOLORS[d], lw=1.8, ls="--", alpha=0.9)

    ax.axhline(0, color="#ccc", lw=0.8)
    ax.set_xlabel("Period")
    ax.set_ylabel("Belief shift (0–100 scale)")

    # Annotate gap for ALL three domains at EVAL
    for d in DOMAINS:
        pv = float(M[f"pop_drift_{d}"][EVAL])
        nv = float(M[f"nonssl_drift_{d}"][EVAL])
        ax.annotate(
            f"{abs(pv - nv):.1f}",
            xy=(EVAL, (pv + nv) / 2),
            fontsize=8.5, color=DCOLORS[d], ha="left",
            xytext=(EVAL + 3, (pv + nv) / 2),
        )

    # Legend outside: right of plot
    legend_handles = []
    for d in DOMAINS:
        pct = int(P_DOMAIN[d] * DEFAULT["p_chatbot"] * 100)
        from matplotlib.lines import Line2D
        legend_handles.append(
            Line2D([0], [0], color=DCOLORS[d],
                   label=f"{DLABELS[d]} ({pct}% SSL)"))

    style_handles = [
        plt.Line2D([0], [0], color="#555", lw=2.2, ls="-",  label="All agents"),
        plt.Line2D([0], [0], color="#555", lw=1.8, ls="--", label="Non-users"),
    ]
    # Single legend below the plot — avoids clipping at right edge
    fig.legend(handles=legend_handles + style_handles, fontsize=10,
               loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3,
               title="Domain (SSL %)   —solid = all agents   – – dashed = non-users",
               title_fontsize=9)

    savefig(fig, "fig1_optout")


def fig2_drivers(data):
    """Misalignment sets the destination; AI opinion variance cancels it."""
    make_aesthetic()
    delta_vals = [0.0, 0.5, 1.0, 2.0]
    sigma_vals = [0.1, 0.3, 0.6, 1.0, 2.0]
    d          = "conventional"

    # Distinct palette colors — gray=baseline (δ=0), then purple/yellow/red spread far apart
    d_cols = ["#aaaaaa", PALETTE[1], PALETTE[2], PALETTE[4]]  # gray, purple, yellow, red
    s_cols = [PALETTE[4], PALETTE[5], PALETTE[2], PALETTE[0], PALETTE[1]]  # red→orange→yellow→teal→purple

    panel_titles = [
        "Misalignment drives cumulative drift",
        "High variance cancels drift",
        "Drift peaks at high δ, low variance",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: delta sweep — legend lower right (lines separate and flat there)
    for dv, color in zip(delta_vals, d_cols):
        M, S = data["fig2_delta"][dv]
        shade(axes[0], M[f"pop_drift_{d}"], S[f"pop_drift_{d}"],
              color, label=f"δ = {dv}σ", lw=2.2, alpha=0.10)
    axes[0].axhline(0, color="#ccc", lw=0.8)
    axes[0].set_xlabel("Period")
    axes[0].set_ylabel("Belief shift (0–100)")
    axes[0].set_title(panel_titles[0], fontsize=12, pad=10)
    # Legend below entire figure via fig.legend with explicit handles
    delta_handles = [plt.Line2D([0],[0], color=c, lw=2.2, label=f"δ = {dv}σ")
                     for dv, c in zip(delta_vals, d_cols)]
    fig.legend(handles=delta_handles, title="Misalignment (δ)", title_fontsize=10,
               fontsize=10, loc="lower left", bbox_to_anchor=(0.02, -0.12), ncol=4)

    # Panel B: sigma_ai sweep — direct right-edge labels, no legend box
    sigma_end = {}
    for sv, color in zip(reversed(sigma_vals), reversed(s_cols)):
        M, S = data["fig2_sigma"][sv]
        shade(axes[1], M[f"pop_drift_{d}"], S[f"pop_drift_{d}"],
              color, lw=2.2, alpha=0.10)
        sigma_end[sv] = (color, float(M[f"pop_drift_{d}"][-1]))
    for sv, (color, yval) in sorted(sigma_end.items()):
        axes[1].text(T + 2, yval, f"σ={sv}", color=color, fontsize=9,
                     va="center", clip_on=False)
    axes[1].axhline(0, color="#ccc", lw=0.8)
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("Belief shift (0–100)")
    axes[1].set_title(panel_titles[1], fontsize=12, pad=10)

    # Panel C: heatmap — annotate all cells (4×5 is legible)
    axes[2].set_title(panel_titles[2], fontsize=12, pad=10)
    hd = data["fig2_heatmap"]
    hmap, dv_grid, sv_grid = hd["hmap"], hd["dv_grid"], hd["sv_grid"]
    vmax = max(abs(hmap).max(), 0.1)
    im = axes[2].imshow(hmap, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    axes[2].set_xticks(range(len(sv_grid)))
    axes[2].set_xticklabels([str(v) for v in sv_grid], fontsize=11)
    axes[2].set_yticks(range(len(dv_grid)))
    axes[2].set_yticklabels([f"{v}σ" for v in dv_grid], fontsize=11)
    axes[2].set_xlabel("AI opinion variance (σ_AI)")
    axes[2].set_ylabel("Misalignment (δ)")
    for i in range(len(dv_grid)):
        for j in range(len(sv_grid)):
            v  = hmap[i, j]
            tc = "white" if abs(v) > vmax * 0.55 else "#424242"
            axes[2].text(j, i, f"{v:.1f}", ha="center", va="center",
                         fontsize=11, color=tc)
    plt.colorbar(im, ax=axes[2], shrink=0.85, label="Belief shift at period 150")

    savefig(fig, "fig2_drivers")


def fig3_domains(data):
    """Higher SSL prevalence → more drift: domain on x so gradient reads left-to-right."""
    make_aesthetic()
    delta_vals  = [0.5, 1.0, 2.0]
    prevalences = {"moral": 10, "personal": 21, "conventional": 31}
    x           = np.arange(len(DOMAINS))

    # Distinct colors for delta levels — sequential ramp gave indistinguishable lines
    delta_cols  = [PALETTE[3], PALETTE[0], PALETTE[4]]   # cyan, teal, red
    delta_marks = ["o", "s", "D"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for metric, ax, title in [
        ("pop_drift",    axes[0], "Drift by domain: all agents"),
        ("nonssl_drift", axes[1], "Drift by domain: non-users only"),
    ]:
        for dv, color, marker in zip(delta_vals, delta_cols, delta_marks):
            vals = [float(data["fig3"][dv][0][f"{metric}_{d}"][EVAL]) for d in DOMAINS]
            errs = [float(data["fig3"][dv][1][f"{metric}_{d}"][EVAL]) for d in DOMAINS]
            ax.errorbar(x, vals, yerr=errs, fmt=marker, color=color, lw=1.8,
                        ms=8, capsize=4, label=f"δ = {dv}σ", zorder=3)
            ax.plot(x, vals, color=color, lw=1.5, alpha=0.5, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{DLABELS[d]}\n({prevalences[d]}% SSL)" for d in DOMAINS], fontsize=11)
        ax.set_ylabel("Belief shift")
        ax.set_title(title, fontsize=13, pad=10)
        ax.axhline(0, color="#ccc", lw=0.8)

    # Explicit handles to avoid duplicates from two panels
    leg_handles = [plt.Line2D([0],[0], color=c, marker=m, lw=1.8, ms=8, label=f"δ = {dv}σ")
                   for dv, c, m in zip(delta_vals, delta_cols, delta_marks)]
    fig.legend(handles=leg_handles, title="AI misalignment", title_fontsize=10, fontsize=10,
               loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3)

    savefig(fig, "fig3_domains")


def fig4_homophily(data):
    """Homophily redistributes AI influence — doesn't reduce the total."""
    make_aesthetic()
    # Reduced to 3 h values — cleaner lines, clearer gradient story
    hom_vals = [0.2, 0.5, 0.8]
    hom_labs = ["h=0.2 (bridging)", "h=0.5 (random mixing)", "h=0.8 (clustering)"]
    h_cols   = [PALETTE[1], PALETTE[0], PALETTE[4]]  # purple, teal, red — clearly distinct

    panel_titles = [
        "AI impact is invariant to h",
        "Belief gap widens with clustering",
        "Non-users gain partial shelter",
    ]
    ylabels = ["Belief shift", "User − non-user gap", "Belief shift (non-users)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    line_handles = []
    for hv, color, label in zip(hom_vals, h_cols, hom_labs):
        M, S = data["fig4"][hv]
        for ax, metric in zip(axes, ["pop_drift", "polarization", "nonssl_drift"]):
            avg_m = np.mean([M[f"{metric}_{d}"] for d in DOMAINS], axis=0)
            # No CI bands — they're too narrow to see and add clutter
            h_line, = ax.plot(np.arange(T + 1), avg_m, color=color, lw=2.2, label=label)
        line_handles.append(h_line)

    for ax, ylabel, title in zip(axes, ylabels, panel_titles):
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.set_xlabel("Period")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, pad=10)

    axes[0].text(0.97, 0.60, "Lines overlap\nby design",
                 transform=axes[0].transAxes, fontsize=9, color="#999",
                 ha="right", va="top")

    # Single shared legend above all panels
    fig.legend(handles=line_handles, labels=hom_labs,
               loc="upper center", bbox_to_anchor=(0.5, 1.07),
               ncol=3, fontsize=11,
               title="Network homophily (h)", title_fontsize=10)

    savefig(fig, "fig4_homophily")


def figS1_beta_sensitivity(data):
    """Finding 1 holds across all tested social influence weights."""
    make_aesthetic()
    beta_vals  = [0.05, 0.14, 0.28]               # 3 for Panel A (avoid overplotting)
    beta_all   = [0.05, 0.10, 0.14, 0.20, 0.28]   # all 5 for ratio panel
    b_cols     = [PALETTE[3], PALETTE[0], PALETTE[4]]  # cyan, teal, red — distinct
    b_labs     = ["β_h = 0.05 (low)", "β_h = 0.14 (mid)", "β_h = 0.28 (Study 2)"]
    d          = "conventional"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={"width_ratios": [1.6, 1]})

    # Panel A: 3 representative β_h trajectories
    axes[0].set_title("Opt-out offers no protection at any β_h", fontsize=12, pad=10)
    for bv, color, label in zip(beta_vals, b_cols, b_labs):
        M, S = data["figS1"][bv]
        shade(axes[0], M[f"pop_drift_{d}"], S[f"pop_drift_{d}"],
              color, lw=2.2, label=label, alpha=0.12)
        axes[0].plot(np.arange(EVAL + 1), M[f"nonssl_drift_{d}"],
                     color=color, lw=1.6, ls="--", alpha=0.9)

    axes[0].axhline(0, color="#ccc", lw=0.8)
    axes[0].set_xlabel("Period")
    axes[0].set_ylabel("Belief shift (0–100)")

    # Single legend above figure: β_h colors + solid/dashed line type
    color_handles = [plt.Line2D([0],[0], color=c, lw=2.2, label=l)
                     for c, l in zip(b_cols, b_labs)]
    style_note = [plt.Line2D([0],[0], color="#555", lw=2.2, ls="-",  label="All agents"),
                  plt.Line2D([0],[0], color="#555", lw=1.6, ls="--", label="Non-users")]
    fig.legend(handles=color_handles + style_note, fontsize=10,
               loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=5)

    # Panel B: ratio plot — no legend needed (self-annotated); tighter y-axis
    axes[1].set_title("Non-user/user ratio near 1.0 across all β_h", fontsize=12, pad=10)
    equil_pd = [float(data["figS1"][bv][0][f"pop_drift_{d}"][EVAL])    for bv in beta_all]
    equil_nu = [float(data["figS1"][bv][0][f"nonssl_drift_{d}"][EVAL]) for bv in beta_all]
    ratios   = [nu/pd if pd > 0.5 else 0. for nu, pd in zip(equil_nu, equil_pd)]

    axes[1].axhline(1.0, color="#424242", lw=1.4, ls="--", zorder=1)
    # Label the reference line at its left edge, not right (avoids collision with last dot)
    axes[1].text(4.5, 1.004, "Perfect tracking",
                 fontsize=9, color="#424242", va="bottom")
    axes[1].plot([bv*100 for bv in beta_all], ratios,
                 "o-", color=PALETTE[0], lw=2, ms=8, zorder=2)
    for bv, r in zip(beta_all, ratios):
        axes[1].annotate(f"{r:.2f}", (bv*100, r),
                         textcoords="offset points", xytext=(0, 10),
                         fontsize=10, ha="center", color="#424242")
    axes[1].set_xlabel("Social influence weight β_h (×100)")
    axes[1].set_ylabel("Non-user / all-user shift")
    axes[1].set_ylim(0.90, 1.05)   # tightened: all points cluster near 1.0
    axes[1].set_xlim(0, 32)

    savefig(fig, "figS1_beta_sens")


###############################################################################
# Results table
###############################################################################

def write_results_tex(data):
    rows = data["table"]
    df = pd.DataFrame([{
        "Condition":      r["label"],
        "Pop. Shift":     r["pop"],
        "Non-User Shift": r["nonssl"],
        "Polarization":   r["polar"],
    } for r in rows])
    latex = df.to_latex(
        index=False,
        escape=False,
        caption=(
            "Simulation outcomes at period 150, conventional domain. "
            r"AI opinion $a \sim \mathcal{N}(\mu_h + \delta\sigma_h,\,(\sigma_{AI}\sigma_h)^2)$ "
            r"drawn fresh each period. $\beta_{AI}=0.26$, $\beta_h=0.28$ (Study 2); $k=3$."
        ),
        label="tab:conditions",
        position="ht",
        column_format="lrrr",
    )
    (HERE / "results_auto.tex").write_text(latex)
    print("  saved results_auto.tex")


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    pkl = HERE / "data.pkl"
    if not pkl.exists():
        raise FileNotFoundError("data.pkl not found — run generate_data.py first")

    with open(pkl, "rb") as f:
        data = pickle.load(f)

    fig1_optout(data)
    fig2_drivers(data)
    fig3_domains(data)
    fig4_homophily(data)
    figS1_beta_sensitivity(data)
    write_results_tex(data)
    print("Done.")
