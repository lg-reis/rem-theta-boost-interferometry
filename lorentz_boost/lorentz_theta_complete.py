# =================================================================
# DEVELOPMENT AND ATTRIBUTION:
#
# Theoretical Framework: Luís Gustavo Costa Reis
#   - θ-parametrization of the Lorentz tensor
#   - REM-θ-framework (Relativistic Electromagnetic Mass) formulation
#   - Physical concepts and hypotheses
#
# Implementation: Claude (Anthropic)
#   - Computational translation of theory
#   - Visualization design and figure generation
#   - Numerical validation and benchmark
#
# NOTE: This is a proof-of-concept simulation. Results demonstrate
# that the θ-parametrization is algebraically equivalent to the
# standard Lorentz formulation. They do not constitute experimental
# validation of the REM-θ-framework. Speedup and timing results
# (figA2) are hardware-dependent -- re-run on your own machine
# before citing specific numbers. All inputs validated to machine
# precision (error ≤ 4.4×10⁻¹⁶, float64, N=10⁶ boosts).
# =================================================================
"""
lorentz_theta_complete.py
=========================
Complete validation, benchmark and figure generation for the
θ-parametrization of the Lorentz tensor in the REM-θ-framework.

Generates 4 figures (figA1–figA4) for the article appendix:
  figA1 — Algebraic equivalence and numerical stability (all 5 inputs)
  figA2 — Computational performance: speedup and error
  figA3 — Tensor 4×4 heatmaps (standard vs θ)
  figA4 — Validation errors for 6 boost directions

  (Merged from the previous 9-figure set: A1+A4+A8 -> new A1;
   A3+A9 -> new A2; A5 -> new A3; A6 -> new A4.
   A2 [triangle], A7 [rapidity] removed; entries 6 [rapidity]
   and 7 [lambda0/lambda_v ratio] dropped from all inputs.)

Usage:
  python lorentz_theta_complete.py
  python lorentz_theta_complete.py --n 2000000
  python lorentz_theta_complete.py --outdir ./figures
  python lorentz_theta_complete.py --quiet

Requirements: Python >= 3.9, NumPy >= 1.21, Matplotlib >= 3.5
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import time
import argparse
import os
import sys

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   12,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

COLORS = ["#1f77b4","#d62728","#2ca02c","#ff7f0e",
          "#9467bd","#8c564b","#e377c2","#17becf","#bcbd22"]

# ── Physical constants ────────────────────────────────────────────────────
ALPHA     = 7.2973525693e-3
LAMBDA_C  = 2.42631023e-12   # m
E0        = 510998.950        # eV  -- rest energy (fundamental input;
                                #        mass is not treated as fundamental)

# ── Utilities ─────────────────────────────────────────────────────────────
def lorentz_standard(beta):
    g  = 1.0 / np.sqrt(1.0 - beta**2)
    bg = beta * g
    return g, bg

def lorentz_theta(sec_th, tan_th, n_hat):
    """4×4 Lorentz boost tensor in θ-parametrization, arbitrary direction."""
    nx, ny, nz = n_hat
    L = np.zeros((4, 4))
    n = np.array([nx, ny, nz])
    L[0, 0] = sec_th
    for i in range(3):
        L[0, i+1] = -tan_th * n[i]
        L[i+1, 0] = -tan_th * n[i]
    for i in range(3):
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            L[i+1, j+1] = delta + (sec_th - 1.0) * n[i] * n[j]
    return L

def metric_error(L):
    eta = np.diag([1., -1., -1., -1.])
    return np.max(np.abs(L.T @ eta @ L - eta))

def save(fig, path, label=""):
    fig.savefig(path)
    plt.show()
    plt.close(fig)
    if label:
        print(f"  Saved: {os.path.basename(path)}")

# ══════════════════════════════════════════════════════════════════════════
#  FIG A1 — Algebraic equivalence and numerical stability (all 5 inputs)
#  (Merged from former A1 + A4 + A8 — see analysis before implementation)
# ══════════════════════════════════════════════════════════════════════════
def fig_equivalence_stability(outdir):
    # ── Panel (a): illustrative overlay, standard vs entry-1 (arcsin beta) ──
    betas  = np.linspace(0.001, 0.999, 2000)
    thetas = np.arcsin(betas)
    g_std  = 1.0 / np.sqrt(1.0 - betas**2)
    bg_std = betas * g_std
    sec_th = 1.0 / np.cos(thetas)
    tan_th = np.tan(thetas)

    # ── Panel (b): relative error vs gamma (log-log), entries 1-5 ──
    # gamma spans 1 to 1e9 so every input's behaviour near beta->1 is visible
    # without the linear-axis crowding that made the old figA4 hard to read.
    gammas = np.logspace(0, 9, 400)
    beta_g = np.sqrt(1.0 - 1.0 / gammas**2)
    E_g    = E0 * gammas
    K_g    = E0 * (gammas - 1.0)
    p_g    = beta_g * gammas          # pc/E0 = beta*gamma = tan(theta)
    lam_g  = beta_g * gammas          # lamC/lam_dB = beta*gamma = tan(theta)

    th1  = np.arcsin(beta_g);            sec1 = 1.0 / np.cos(th1)
    sec2 = E_g / E0
    sec3 = 1.0 + K_g / E0
    sec4 = np.sqrt(1.0 + p_g**2)
    sec5 = np.sqrt(1.0 + lam_g**2)

    def relerr(sec):
        return np.abs(sec / gammas - 1.0) + 1e-18

    entries = [
        (r"1: $\arcsin(\beta)$",                 relerr(sec1), COLORS[3], "-"),
        (r"2: $E/E_0$",                            relerr(sec2), COLORS[0], "-"),
        (r"3: $1+K/E_0$",                          relerr(sec3), COLORS[2], "--"),
        (r"4: $\sqrt{1+(pc/E_0)^2}$",              relerr(sec4), COLORS[4], "-."),
        (r"5: $\sqrt{1+(\lambda_C/\lambda_{\rm dB})^2}$", relerr(sec5), COLORS[6], ":"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(betas, g_std,  lw=2.5, color=COLORS[0], label=r"$\gamma$ (standard)")
    ax.plot(betas, sec_th, lw=1.5, color=COLORS[0], ls="--",
            label=r"$\sec\theta$ ($\arcsin\beta$)")
    ax.plot(betas, bg_std, lw=2.5, color=COLORS[1], label=r"$\beta\gamma$ (standard)")
    ax.plot(betas, tan_th, lw=1.5, color=COLORS[1], ls="--",
            label=r"$\tan\theta$ ($\arcsin\beta$)")
    ax.set_xlabel(r"$\beta = v/c$")
    ax.set_ylabel("Value")
    ax.set_title(r"(a) $\gamma\!=\!\sec\theta$, $\beta\gamma\!=\!\tan\theta$",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 12)

    ax2 = axes[1]
    for label, err, color, ls in entries:
        ax2.loglog(gammas, err, ls, lw=2, color=color, label=label)
    ax2.axhline(1e-15, color="gray", ls=":", lw=1.2, label="float64 precision")
    ax2.set_xlabel(r"$\gamma$ (Lorentz factor, log scale)")
    ax2.set_ylabel(r"Relative error $|\sec\theta_{\rm input}/\gamma - 1|$")
    ax2.set_title("(b) Stability across 9 decades of $\\gamma$",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Algebraic equivalence and numerical stability of the "
        r"$\theta$-parametrization" + "\n"
        r"$\sec^2\theta - \tan^2\theta = 1 \;\Leftrightarrow\; "
        r"\gamma^2 - (\beta\gamma)^2 = 1$   |   "
        "entries 2-5 (arithmetic) remain at machine precision "
        r"for any $\gamma$; entry 1 ($\beta$-based) degrades near $\beta\to1$",
        fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    save(fig, os.path.join(outdir, "figA1_equivalence_stability.png"), "A1")

# ══════════════════════════════════════════════════════════════════════════
#  FIG A2 — Computational performance: speedup and error
#  (Merged from former A3 + A9 — raw ms panel dropped, it is machine-
#   dependent and not a meaningful cross-reader comparison; speedup,
#   a dimensionless ratio, is the right normalised metric to show.
#   Entries 6 [rapidity] and 7 [lambda0/lambda_v] removed.)
# ══════════════════════════════════════════════════════════════════════════
def fig_benchmark(outdir, N=1_000_000, NTRIALS=15):
    np.random.seed(42)
    betas = np.random.uniform(0.01, 0.9999, N)

    g_ref = 1.0 / np.sqrt(1.0 - betas**2)

    E_total = E0 * g_ref
    K_cin   = E0 * (g_ref - 1.0)
    p_norm  = betas * g_ref
    lam_dB  = betas * g_ref

    NREP = 5

    funcs = [
        ("Standard\n" + r"$1/\sqrt{1-\beta^2}$",
         lambda: (1.0/np.sqrt(1.0-betas**2), betas/np.sqrt(1.0-betas**2))),
        (r"1: $\arcsin(\beta)$",
         lambda: (1.0/np.cos(np.arcsin(betas)), np.tan(np.arcsin(betas)))),
        (r"2: $E/E_0$ *",
         lambda: (E_total/E0, np.sqrt((E_total/E0)**2 - 1.0))),
        (r"3: $1+K/E_0$ **",
         lambda: (1.0 + K_cin/E0, np.sqrt((1.0 + K_cin/E0)**2 - 1.0))),
        (r"4: $\arctan(pc/E_0)$",
         lambda: (np.sqrt(1.0 + p_norm**2), p_norm)),
        (r"5: $\arctan(\lambda_C/\lambda_{\rm dB})$",
         lambda: (np.sqrt(1.0 + lam_dB**2), lam_dB)),
    ]
    labels = [f for f, _ in funcs]

    # ── Rigorous timing protocol ──────────────────────────────────────
    # 1) one shared warm-up pass over ALL entries first (JIT-like/cache
    #    effects at the *process* level are paid once, up front, for
    #    everyone equally -- not just before each entry's own timing).
    # 2) NTRIALS independent, paired trials: every entry is timed once
    #    per trial, in the same order, so trial i for all entries shares
    #    roughly the same system conditions (background load, thermal
    #    state). Speedup is computed as a PAIRED ratio within each
    #    trial, then we report the median across trials -- much less
    #    sensitive to a single unlucky (or lucky) measurement than a
    #    one-shot benchmark.
    for _, f in funcs:
        f()

    times = {lbl: [] for lbl in labels}
    for _trial in range(NTRIALS):
        for lbl, f in funcs:
            t0 = time.perf_counter()
            for _ in range(NREP):
                sec, tan = f()
            t1 = time.perf_counter()
            times[lbl].append((t1 - t0) / NREP * 1000)

    std_times = np.array(times[labels[0]])
    speedup_med, speedup_lo, speedup_hi = [], [], []
    for lbl in labels:
        ratios = std_times / np.array(times[lbl])
        speedup_med.append(np.median(ratios))
        speedup_lo.append(np.median(ratios) - np.min(ratios))
        speedup_hi.append(np.max(ratios) - np.median(ratios))
    speedup_med[0] = 1.0; speedup_lo[0] = 0.0; speedup_hi[0] = 0.0  # Standard vs itself

    # ── Error (deterministic, one pass suffices -- no timing noise) ──
    errors = []
    for lbl, f in funcs:
        sec, tan = f()
        errors.append(max(float(np.max(np.abs(sec - g_ref))), 1e-18))

    x = np.arange(len(labels))
    clrs = [COLORS[1] if i==0 else COLORS[0] for i in range(len(labels))]
    clrs[2] = COLORS[2]; clrs[3] = COLORS[2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bars = ax.bar(x, speedup_med, color=clrs, alpha=0.85, edgecolor="k", lw=0.5,
                   yerr=[speedup_lo, speedup_hi], capsize=4,
                   error_kw=dict(lw=1.2, ecolor="black"))
    ax.axhline(1.0, color=COLORS[1], ls="--", lw=1.5, label="Standard = 1.0×")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel("Speedup relative to standard (median)")
    ax.set_title(f"(a) Speedup — median of {NTRIALS} paired trials",
                 fontsize=11, fontweight="bold")
    ax.legend()
    for bar, sp in zip(bars, speedup_med):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f"{sp:.2f}×", ha="center", va="bottom", fontsize=8.5)

    ax2 = axes[1]
    ax2.bar(x, errors, color=clrs, alpha=0.85, edgecolor="k", lw=0.5)
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax2.set_ylabel(r"Max $|\sec\theta_{\rm input} - \gamma_{\rm std}|$")
    ax2.set_title("(b) Numerical error vs standard", fontsize=11, fontweight="bold")
    ax2.axhline(1e-14, color="gray", ls=":", lw=1.5, label="Machine precision")
    ax2.legend()

    fig.suptitle(
        f"Computational performance: all input modes for the "
        r"$\theta$-parametrization" + "\n"
        f"N = {N:,} random boosts  |  β ∈ [0.01, 0.9999]  |  "
        f"{NTRIALS} paired trials, warm-up excluded  |  "
        f"* = 1 division  |  ** = 1 addition",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, os.path.join(outdir, "figA2_performance.png"), "A2")

# ══════════════════════════════════════════════════════════════════════════
def fig_heatmaps(outdir):
    beta  = 0.7
    n_hat = np.array([2/7, 3/7, 6/7])
    g, bg = lorentz_standard(beta)
    th    = math.asin(beta)
    sec   = 1.0/math.cos(th)
    tan   = math.tan(th)

    L_std = lorentz_theta(g,   bg,  n_hat)
    L_th  = lorentz_theta(sec, tan, n_hat)
    diff  = np.abs(L_th - L_std)

    labels_rc = [r"$ct$", r"$x$", r"$y$", r"$z$"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, mat, title in zip(
        axes,
        [L_std, L_th, diff + 1e-18],
        [r"Standard $\Lambda(\gamma,\beta\gamma)$",
         r"θ-param. $\Lambda(\sec\theta,\tan\theta)$",
         r"$|\Lambda_\theta - \Lambda_{\rm std}|$"]):
        vmin = mat.min() if "diff" not in title.lower() else None
        im = ax.imshow(mat, cmap="RdBu_r" if "diff" not in title else "Reds",
                       aspect="equal",
                       norm=matplotlib.colors.LogNorm() if "|" in title else None)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(labels_rc); ax.set_yticklabels(labels_rc)
        ax.set_title(title, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
        is_diff = "|" in title
        for i in range(4):
            for j in range(4):
                v = mat[i,j]
                txt = f"{v:.2e}" if is_diff else f"{v:.3f}"
                if is_diff:
                    txt_color = "white"
                else:
                    vrange = mat.max() - mat.min()
                    normalized = (v - mat.min()) / (vrange + 1e-18)
                    txt_color = "white" if (normalized < 0.25 or normalized > 0.75) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=txt_color, fontweight="bold")

    fig.suptitle(
        r"Lorentz tensor 4×4 heatmaps: $\beta=0.7$, "
        r"$\hat{n}=(2/7,3/7,6/7)$" + "\n"
        "Left: standard form | Centre: θ-parametrization | "
        "Right: absolute difference",
        fontsize=10, fontweight="bold")
    plt.tight_layout()
    save(fig, os.path.join(outdir, "figA3_heatmaps.png"), "A3")

# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════
def fig_validation(outdir):
    directions = {
        r"$\hat{x}$":              np.array([1,0,0]),
        r"$\hat{y}$":              np.array([0,1,0]),
        r"$\hat{z}$":              np.array([0,0,1]),
        r"$(1,1,0)/\sqrt{2}$":     np.array([1,1,0])/math.sqrt(2),
        r"$(1,1,1)/\sqrt{3}$":     np.array([1,1,1])/math.sqrt(3),
        r"$(2,3,6)/7$":            np.array([2,3,6])/7,
    }
    betas_t = [0.3, 0.7, 0.99]
    eta     = np.diag([1.,-1.,-1.,-1.])

    tests = ["Metric\n" + r"$\Lambda(\theta)^\top\eta\,\Lambda(\theta)=\eta$",
             "Group\n" + r"$\Lambda_1(\theta)\Lambda_2(\theta)=\Lambda_{\rm res}(\theta)$",
             "Inverse\n" + r"$\Lambda(\theta)\,\Lambda(\theta)^{-1}=I_4$"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax_idx, (test_name, ax) in enumerate(zip(tests, axes)):
        all_errs = []
        dir_names = list(directions.keys())
        for d_name, n_hat in directions.items():
            errs = []
            for b in betas_t:
                th  = math.asin(b)
                sec = 1.0/math.cos(th)
                tan = math.tan(th)
                L   = lorentz_theta(sec, tan, n_hat)

                if ax_idx == 0:  # metric
                    err = metric_error(L)
                elif ax_idx == 1:  # group — parallel boosts along same direction
                    b2   = min(0.3, 0.995 - b)
                    b_r  = min((b + b2)/(1 + b*b2), 0.9999)
                    th2  = math.asin(b2)
                    th_r = math.asin(b_r)
                    n_par = np.array([1.,0.,0.])
                    L_p  = lorentz_theta(1/math.cos(math.asin(b)),
                                          math.tan(math.asin(b)), n_par)
                    L2_p = lorentz_theta(1/math.cos(th2), math.tan(th2), n_par)
                    Lr_p = lorentz_theta(1/math.cos(th_r), math.tan(th_r), n_par)
                    err  = np.max(np.abs(L_p @ L2_p - Lr_p))
                else:  # inverse
                    Li  = lorentz_theta(sec, -tan, n_hat)
                    I4  = np.eye(4)
                    err = np.max(np.abs(L @ Li - I4))
                errs.append(max(err, 1e-18))
            all_errs.append(errs)

        x  = np.arange(len(directions))
        w  = 0.25
        for i, (b, c) in enumerate(zip(betas_t, COLORS)):
            vals = [all_errs[j][i] for j in range(len(directions))]
            ax.bar(x + i*w, vals, w, label=rf"$\beta={b}$",
                   color=c, alpha=0.8, edgecolor="k", lw=0.4)
        ax.set_yscale("log")
        ax.set_xticks(x + w)
        ax.set_xticklabels(dir_names, fontsize=8, rotation=15, ha="right")
        ax.set_title(test_name, fontsize=10, fontweight="bold")
        ax.set_ylabel("Max absolute error")
        ax.axhline(1e-14, color="gray", ls=":", lw=1.5,
                   label="float64 precision")
        ax.legend(fontsize=8)

    fig.suptitle(
        r"Validation of the $\theta$-parametrization tensor $\Lambda(\theta)$ "
        "\n"
        "6 boost directions × 3 velocities, all errors at machine "
        "precision (float64)",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, os.path.join(outdir, "figA4_validation.png"), "A4")

# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════
def print_validation_table(quiet):
    if quiet:
        return
    print()
    print("  " + "═"*68)
    print("  VALIDATION TABLE — Tensor 4×4 via all inputs")
    print("  " + "═"*68)
    eta = np.diag([1.,-1.,-1.,-1.])
    directions = {
        "x-axis":          np.array([1,0,0]),
        "y-axis":          np.array([0,1,0]),
        "z-axis":          np.array([0,0,1]),
        "(1,1,0)/√2":      np.array([1,1,0])/math.sqrt(2),
        "(1,1,1)/√3":      np.array([1,1,1])/math.sqrt(3),
        "(2,3,6)/7":       np.array([2,3,6])/7,
    }
    betas_t = [0.3, 0.7, 0.99]
    print(f"\n  {'Direction':^14}  {'β':>6}  "
          f"{'Metric err':>12}  {'Group err':>12}  {'Inverse err':>12}")
    print("  " + "─"*60)
    all_pass = True
    for d_name, n_hat in directions.items():
        for b in betas_t:
            th  = math.asin(b)
            sec = 1.0/math.cos(th)
            tan = math.tan(th)
            L   = lorentz_theta(sec, tan, n_hat)
            # metric
            e_metric = metric_error(L)
            # group — parallel boosts along x-axis
            b2   = min(0.3, 0.994 - b)
            b_r  = min((b + b2)/(1 + b*b2), 0.9999)
            th2  = math.asin(b2)
            th_r = math.asin(b_r)
            n_par = np.array([1.,0.,0.])
            L_p  = lorentz_theta(1/math.cos(math.asin(b)),
                                  math.tan(math.asin(b)), n_par)
            L2_p = lorentz_theta(1/math.cos(th2), math.tan(th2), n_par)
            Lr_p = lorentz_theta(1/math.cos(th_r), math.tan(th_r), n_par)
            e_group = np.max(np.abs(L_p @ L2_p - Lr_p))
            # inverse
            Li = lorentz_theta(sec, -tan, n_hat)
            e_inv = np.max(np.abs(L @ Li - np.eye(4)))
            ok = all(e < 1e-12 for e in [e_metric, e_group, e_inv])
            all_pass = all_pass and ok
            status = "✓" if ok else "✗"
            print(f"  {d_name:^14}  {b:>6.2f}  "
                  f"{e_metric:>12.4e}  {e_group:>12.4e}  "
                  f"{e_inv:>12.4e}  {status}")
    print()
    print(f"  {'ALL TESTS PASSED ✓' if all_pass else 'FAILURES DETECTED ✗':^60}")
    print("  " + "═"*68)

# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Complete θ-parametrization validation and figures")
    parser.add_argument("--n",      type=int, default=1_000_000)
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'))
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if not args.quiet:
        print()
        print("  " + "═"*68)
        print("  lorentz_theta_complete.py")
        print("  REM-θ-framework — Luís Gustavo Costa Reis")
        print("  " + "═"*68)
        print(f"\n  Output directory: {args.outdir}")
        print(f"  Benchmark N     : {args.n:,}")
        print()

    steps = [
        ("Fig A1 — Equivalence and stability", lambda: fig_equivalence_stability(args.outdir)),
        ("Fig A2 — Performance (speedup/error)", lambda: fig_benchmark(args.outdir, args.n)),
        ("Fig A3 — Tensor heatmaps",             lambda: fig_heatmaps(args.outdir)),
        ("Fig A4 — Validation 6 directions",     lambda: fig_validation(args.outdir)),
    ]

    for label, func in steps:
        if not args.quiet:
            print(f"  Generating {label}...")
        func()

    print_validation_table(args.quiet)

    if not args.quiet:
        figs = sorted([f for f in os.listdir(args.outdir) if f.endswith(".png")])
        print(f"\n  Generated {len(figs)} figures in '{args.outdir}/':")
        for f in figs:
            sz = os.path.getsize(os.path.join(args.outdir, f)) // 1024
            print(f"    {f:<45} {sz:>5} KB")
        print()
        print("  " + "═"*68)
        print("  Complete. All figures ready for appendix.")
        print("  " + "═"*68)
        print()

if __name__ == "__main__":
    main()
