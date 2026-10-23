# =================================================================
# DEVELOPMENT AND ATTRIBUTION:
#
# Theoretical Framework: Luís Gustavo Costa Reis
#   - θ-parametrization of the Lorentz transformation matrix
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
# are hardware-dependent -- re-run on your own machine before citing
# specific numbers. Error figures are printed at run time and written
# to lorentz_results.txt; they grow with gamma (see the random-matrix
# test) and are NOT a single machine-precision constant.
# =================================================================
"""
lorentz_theta_complete.py
=========================
Validation, benchmark and figure generation for the θ-parametrization
of the Lorentz transformation matrix in the REM-θ-framework.

Figures (Section 3.1 of the article, in order of appearance):
  figA2 — Computational performance and precision of the inputs
  figA3 — Element-by-element comparison of the 4x4 transformation matrix
          (independent standard matrix vs θ-parametrized matrix)
  figA1 — Algebraic equivalence and numerical stability (all 6 inputs
          + the standard formula), against a 60-digit reference
  figA4 — Structural properties of Λ(θ): metric, collinear composition
          and inverse for 6 boost directions x 3 velocities

Extra numerical studies (printed and written to lorentz_results.txt):
  * random-matrix test: N boosts, random directions, beta in [0.01, 0.9999]
  * comparison of float64 matrices with a 60-digit (decimal) reference
  * float32 study (beta input vs momentum input)
  * theta -> 0 study: (gamma-1) and tan(theta) near the Galilean limit
  * multiplicative composition rule tan(pi/4 + theta/2)

Benchmark (v1.8): every input starts from the raw physical quantity and
normalizes it inside the timed region; each input is timed as the arithmetic
form and as the explicit theta path; gamma and beta*gamma are both checked
against the 60-digit reference (see run_benchmark).

Usage:
  python lorentz_theta_complete.py
  python lorentz_theta_complete.py --n 2000000 --nmatrix 1000000
  python lorentz_theta_complete.py --outdir ./figures --quiet
  python lorentz_theta_complete.py --skip-figures

Requirements: Python >= 3.9, NumPy >= 1.21, Matplotlib >= 3.5
(the 60-digit reference uses the standard-library `decimal` module).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import time
import argparse
import os
from decimal import Decimal, getcontext

getcontext().prec = 60

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

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
          "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]

# ── Physical constants ────────────────────────────────────────────────────
E0 = 510998.950        # eV -- electron rest energy (fundamental input;
                       #       mass is not treated as fundamental)

ETA = np.diag([1.0, -1.0, -1.0, -1.0])
PERF_BETA_RANGE = (0.01, 0.9999)

# ══════════════════════════════════════════════════════════════════════════
#  Exact (60-digit) reference helpers
# ══════════════════════════════════════════════════════════════════════════
def _dec(x):
    return Decimal(float(x))


def exact_inputs(gammas):
    """Physical inputs for exactly-known gamma values.

    `gammas` is a float64 array whose values are treated as exact reals.
    The inputs (beta, E/E0, K/E0, pc/E0, t/tau) are computed in 60-digit
    arithmetic and then rounded to float64 -- i.e. what a user holding a
    float64 measurement of each quantity would actually have.

    t/tau (time dilation) is numerically identical to E/E0: both equal
    gamma by definition (sec(theta) = gamma = E/E0 = t/tau). It is kept
    as its own array -- not aliased to E -- so every input in this script
    goes through the same "round an exact value to float64" step, and so
    that renaming it later (e.g. if a different reference time is used)
    does not silently touch the energy input.
    """
    beta, E, K, p, T = [], [], [], [], []
    one = Decimal(1)
    for g in gammas:
        G = _dec(g)
        p_dec = ((G - one) * (G + one)).sqrt()      # beta*gamma = tan(theta)
        beta.append(float(p_dec / G))
        E.append(float(G))
        K.append(float(G - one))
        p.append(float(p_dec))
        T.append(float(G))                          # t/tau = gamma, exact
    return (np.array(beta), np.array(E), np.array(K), np.array(p), np.array(T))


def rel_err_exact(sec, gammas):
    """|sec/gamma - 1| against the exact gamma (60 digits).

    `gammas` may be floats (treated as exact reals) or Decimals.
    """
    out = np.empty(len(gammas))
    for i, (s, g) in enumerate(zip(sec, gammas)):
        if not np.isfinite(s):
            out[i] = np.inf
            continue
        G = g if isinstance(g, Decimal) else _dec(g)
        out[i] = float(abs(_dec(s) - G) / G)
    return out


def exact_inputs_from_beta(betas):
    """Inputs for a *given* float64 beta (the true beta is exactly `beta`).

    Returns (beta, E/E0, K/E0, pc/E0, t/tau, gamma_exact) with gamma_exact
    a list of 60-digit Decimals. Unlike exact_inputs(), beta is NOT
    recovered from a rounded gamma, so the standard formula is not tested
    circularly.
    """
    E, K, p, T, G_list = [], [], [], [], []
    one = Decimal(1)
    for b in betas:
        B = _dec(b)
        G = one / (one - B * B).sqrt()
        G_list.append(G)
        E.append(float(G))
        K.append(float(G - one))
        p.append(float(B * G))
        T.append(float(G))                          # t/tau = gamma, exact
    return (np.asarray(betas, dtype=float), np.array(E), np.array(K),
            np.array(p), np.array(T), G_list)


def sec_methods(beta, E_ratio, K_ratio, p_ratio, T_ratio):
    """gamma = sec(theta) from each input (float64 arrays).

    s6 (time dilation, t/tau) is the same pass-through as s2 (E/E0): both
    quantities equal gamma directly, by definition (sec(theta) = gamma =
    E/E0 = t/tau). The two columns in fig_equivalence_stability() /
    run_benchmark() are therefore expected to track each other almost
    exactly -- this is not a bug, see the note in exact_inputs().
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        std = 1.0 / np.sqrt(1.0 - beta ** 2)
        th1 = np.arcsin(beta)
        s1 = 1.0 / np.cos(th1)
    s2 = E_ratio
    s3 = 1.0 + K_ratio
    s4 = np.sqrt(1.0 + p_ratio * p_ratio)
    s5 = np.sqrt(1.0 + p_ratio * p_ratio)      # lambda_C/lambda_dB = tan(theta)
    # theta = arcsec(t/tau) by definition, so sec(theta) = T_ratio exactly;
    # evaluating it this way (rather than np.arccos(1.0/T_ratio) followed by
    # 1/cos(...)) is both the algebraically simplest form and the
    # numerically stable one -- going through arccos explicitly would
    # reintroduce the beta -> 0 cancellation discussed for item 3.
    s6 = T_ratio
    return std, s1, s2, s3, s4, s5, s6


# ══════════════════════════════════════════════════════════════════════════
#  Boost matrices
# ══════════════════════════════════════════════════════════════════════════
def _unit(n_hat):
    n = np.asarray(n_hat, dtype=float)
    if abs(np.linalg.norm(n) - 1.0) > 1e-9:
        raise ValueError("n_hat must be a unit vector (|n_hat| = 1); "
                         f"got |n_hat| = {np.linalg.norm(n):.12g}")
    return n


def boost_theta(sec_th, tan_th, n_hat):
    """4x4 boost in the θ-parametrization, arbitrary direction (Eq. 3.5).

    sec(θ)-1 is evaluated as tan²θ/(1+sec θ), which is algebraically
    identical and free of cancellation when θ -> 0.
    """
    n = _unit(n_hat)
    secm1 = tan_th ** 2 / (1.0 + sec_th)
    L = np.zeros((4, 4))
    L[0, 0] = sec_th
    L[0, 1:] = -tan_th * n
    L[1:, 0] = -tan_th * n
    L[1:, 1:] = np.eye(3) + secm1 * np.outer(n, n)
    return L


def boost_standard(beta, n_hat):
    """Textbook boost (Eq. 3.4), written independently of the θ-parametrized matrix."""
    n = _unit(n_hat)
    g = 1.0 / math.sqrt(1.0 - beta ** 2)
    L = np.zeros((4, 4))
    L[0, 0] = g
    L[0, 1:] = -g * beta * n
    L[1:, 0] = -g * beta * n
    L[1:, 1:] = np.eye(3) + (g - 1.0) * np.outer(n, n)
    return L


def boost_theta_from_beta(beta, n_hat):
    th = math.asin(beta)
    return boost_theta(1.0 / math.cos(th), math.tan(th), n_hat)


def boost_decimal(beta, n_hat):
    """Exact-arithmetic (60 digits) boost; returns a 4x4 list of Decimals."""
    B = _dec(beta)
    nd = [_dec(x) for x in n_hat]
    nn = sum(x * x for x in nd).sqrt()
    nd = [x / nn for x in nd]
    G = Decimal(1) / (Decimal(1) - B * B).sqrt()
    L = [[Decimal(0)] * 4 for _ in range(4)]
    L[0][0] = G
    for i in range(3):
        L[0][i + 1] = -G * B * nd[i]
        L[i + 1][0] = -G * B * nd[i]
        for j in range(3):
            L[i + 1][j + 1] = (Decimal(1) if i == j else Decimal(0)) \
                              + (G - 1) * nd[i] * nd[j]
    return L


def max_abs_diff_decimal(L_float, L_dec):
    return max(float(abs(_dec(L_float[i, j]) - L_dec[i][j]))
               for i in range(4) for j in range(4))


def metric_error(L):
    return float(np.max(np.abs(L.T @ ETA @ L - ETA)))


def save(fig, path, label=""):
    fig.savefig(path)
    plt.close(fig)
    if label:
        print(f"  Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════
#  Numerical studies (no figures)
# ══════════════════════════════════════════════════════════════════════════
def tensor_random_test(N=1_000_000, seed=12345, chunk=100_000):
    """N random boosts (random unit directions), theta-parametrized matrix from beta.

    Returns metric/inverse residuals and their scaling with gamma.
    """
    rng = np.random.default_rng(seed)
    b_all = rng.uniform(*PERF_BETA_RANGE, N)
    err_m = np.empty(N)
    err_i = np.empty(N)
    for s in range(0, N, chunk):
        b = b_all[s:s + chunk]
        m = len(b)
        n = rng.normal(size=(m, 3))
        n /= np.linalg.norm(n, axis=1)[:, None]
        th = np.arcsin(b)
        sec = 1.0 / np.cos(th)
        tan = np.tan(th)
        secm1 = tan ** 2 / (1.0 + sec)
        L = np.empty((m, 4, 4))
        L[:, 0, 0] = sec
        L[:, 0, 1:] = -tan[:, None] * n
        L[:, 1:, 0] = -tan[:, None] * n
        L[:, 1:, 1:] = np.eye(3)[None] + secm1[:, None, None] * n[:, :, None] * n[:, None, :]
        M = np.einsum("nji,jk,nkl->nil", L, ETA, L) - ETA
        err_m[s:s + m] = np.abs(M).reshape(m, -1).max(axis=1)
        Li = L.copy()
        Li[:, 0, 1:] *= -1.0
        Li[:, 1:, 0] *= -1.0
        I = np.einsum("nij,njk->nik", L, Li) - np.eye(4)
        err_i[s:s + m] = np.abs(I).reshape(m, -1).max(axis=1)
    gam = 1.0 / np.sqrt(1.0 - b_all ** 2)
    bands = [(0.01, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 0.9999)]
    band_rows = []
    for lo, hi in bands:
        sel = (b_all >= lo) & (b_all < hi + (1e-12 if hi == 0.9999 else 0))
        band_rows.append((lo, hi, float(err_m[sel].max())))
    k = int(err_m.argmax())
    return {
        "N": N, "seed": seed,
        "metric_max": float(err_m.max()),
        "metric_median": float(np.median(err_m)),
        "metric_p99": float(np.percentile(err_m, 99)),
        "inverse_max": float(err_i.max()),
        "metric_over_gamma2_max": float((err_m / gam ** 2).max()),
        "gamma_at_max": float(gam[k]),
        "gamma_max": float(gam.max()),
        "bands": band_rows,
    }


def tensor_reference_check(nref=500, seed=2024):
    """float64 theta / standard matrices vs the 60-digit reference."""
    rng = np.random.default_rng(seed)
    worst_th = worst_std = 0.0
    for _ in range(nref):
        b = rng.uniform(*PERF_BETA_RANGE)
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        ref = boost_decimal(b, n)
        worst_th = max(worst_th, max_abs_diff_decimal(boost_theta_from_beta(b, n), ref))
        worst_std = max(worst_std, max_abs_diff_decimal(boost_standard(b, n), ref))
    return {"nref": nref, "theta_vs_exact": worst_th, "standard_vs_exact": worst_std}


def float32_study(gammas=(3, 10, 30, 70, 100, 300, 1000, 3000)):
    """Relative error of gamma in float32: beta input vs momentum input."""
    rows = []
    g = np.array(gammas, dtype=float)
    beta, E, K, p, T = exact_inputs(g)
    b32 = beta.astype(np.float32)
    p32 = p.astype(np.float32)
    one = np.float32(1.0)
    std = one / np.sqrt(one - b32 * b32)
    ars = one / np.cos(np.arcsin(b32))
    sp = np.sqrt(one + p32 * p32)
    e_std = rel_err_exact(std.astype(float), g)
    e_ars = rel_err_exact(ars.astype(float), g)
    e_p = rel_err_exact(sp.astype(float), g)
    for i, gm in enumerate(gammas):
        rows.append((gm, e_std[i], e_ars[i], e_p[i]))
    return rows


def small_theta_study(betas=(1e-2, 1e-4, 1e-6, 1e-8),
                      ks=(1e-2, 1e-4, 1e-6, 1e-8, 1e-10)):
    """theta -> 0: relative error of (gamma-1) and of tan(theta)."""
    rows_g1 = []
    for beta in betas:
        B = Decimal(beta)
        g1 = Decimal(1) / (Decimal(1) - B * B).sqrt() - 1
        sec = 1.0 / math.cos(math.asin(beta))
        naive = sec - 1.0
        t = beta / math.sqrt(1.0 - beta ** 2)
        stable = t * t / (1.0 + math.sqrt(1.0 + t * t))
        rows_g1.append((beta,
                        float(abs(_dec(naive) - g1) / g1),
                        float(abs(_dec(stable) - g1) / g1)))
    rows_tan = []
    for kx in ks:
        x = Decimal(kx)
        tan_exact = (x * (x + 2)).sqrt()
        xf = float(kx)
        naive = math.sqrt((1.0 + xf) ** 2 - 1.0)      # sqrt(gamma^2 - 1)
        stable = math.sqrt(xf * (xf + 2.0))           # sqrt(x(x+2))
        rows_tan.append((kx,
                         float(abs(_dec(naive) - tan_exact) / tan_exact),
                         float(abs(_dec(stable) - tan_exact) / tan_exact)))
    return rows_g1, rows_tan


def composition_check():
    """tan(pi/4 + theta/2) is multiplicative under collinear composition."""
    worst = 0.0
    for b1 in np.linspace(0.05, 0.9, 18):
        for b2 in np.linspace(0.05, 0.9, 18):
            th1, th2 = math.asin(b1), math.asin(b2)
            thr = math.asin((b1 + b2) / (1.0 + b1 * b2))
            lhs = math.tan(math.pi / 4 + thr / 2)
            rhs = math.tan(math.pi / 4 + th1 / 2) * math.tan(math.pi / 4 + th2 / 2)
            worst = max(worst, abs(lhs - rhs) / rhs)
    return worst


# ══════════════════════════════════════════════════════════════════════════
#  Structural validation (metric, group, inverse) -- 6 directions x 3 beta
# ══════════════════════════════════════════════════════════════════════════
DIRECTIONS = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
    "(1,1,0)/√2": np.array([1.0, 1.0, 0.0]) / math.sqrt(2),
    "(1,1,1)/√3": np.array([1.0, 1.0, 1.0]) / math.sqrt(3),
    "(2,3,6)/7": np.array([2.0, 3.0, 6.0]) / 7.0,
}
BETAS_VALIDATION = [0.3, 0.7, 0.99]


def second_beta(b):
    """Second collinear boost used in the group test (same rule everywhere)."""
    return min(0.3, 0.995 - b)


def validation_errors():
    """errs[test][direction][beta] with test in {metric, group, inverse}."""
    errs = {t: {d: [] for d in DIRECTIONS} for t in ("metric", "group", "inverse")}
    for d_name, n_hat in DIRECTIONS.items():
        for b in BETAS_VALIDATION:
            th = math.asin(b)
            sec, tan = 1.0 / math.cos(th), math.tan(th)
            L = boost_theta(sec, tan, n_hat)
            errs["metric"][d_name].append(metric_error(L))
            b2 = second_beta(b)
            b_r = (b + b2) / (1.0 + b * b2)               # velocity addition
            L1 = boost_theta_from_beta(b, n_hat)
            L2 = boost_theta_from_beta(b2, n_hat)          # same direction n_hat
            Lr = boost_theta_from_beta(b_r, n_hat)
            errs["group"][d_name].append(float(np.max(np.abs(L1 @ L2 - Lr))))
            Li = boost_theta(sec, -tan, n_hat)
            errs["inverse"][d_name].append(float(np.max(np.abs(L @ Li - np.eye(4)))))
    return errs


# ══════════════════════════════════════════════════════════════════════════
#  FIG A1 — Algebraic equivalence and numerical stability
# ══════════════════════════════════════════════════════════════════════════
def fig_equivalence_stability(outdir):
    # (a) illustrative overlay, standard vs entry 1 (arcsin beta)
    betas = np.linspace(0.001, 0.999, 2000)
    thetas = np.arcsin(betas)
    g_std = 1.0 / np.sqrt(1.0 - betas ** 2)
    bg_std = betas * g_std
    sec_th = 1.0 / np.cos(thetas)
    tan_th = np.tan(thetas)

    # (b) relative error vs gamma against a 60-digit reference. Every input
    #     (beta, E, K, p, lambda_C/lambda_dB) is computed exactly and then
    #     rounded to float64, as a user holding a float64 value would have.
    gammas = np.logspace(0, 9, 400)
    beta_g, E_g, K_g, p_g, T_g = exact_inputs(gammas)
    std, s1, s2, s3, s4, s5, s6 = sec_methods(beta_g, E_g, K_g, p_g, T_g)

    def relerr(sec):
        e = rel_err_exact(sec, gammas)
        e = np.where(np.isfinite(e), e, np.nan)      # inf -> not drawn
        return e + 1e-18

    entries = [
        (r"Standard $1/\sqrt{1-\beta^2}$",            relerr(std), "black",    "-"),
        (r"1: $\theta=\arcsin(\beta)$",                       relerr(s1),  COLORS[3], "-"),
        (r"2: $\theta=\arccos(E_0/E)$",                       relerr(s2),  COLORS[0], "-"),
        (r"3: $\theta=\mathrm{arcsec}(1+K/E_0)$",             relerr(s3),  COLORS[2], "--"),
        (r"4: $\theta=\arctan(pc/E_0)$",                      relerr(s4),  COLORS[4], "-."),
        (r"5: $\theta=\arctan(\lambda_C/\lambda_{\rm dB})$",  relerr(s5),  COLORS[6], ":"),
        (r"6: $\theta=\mathrm{arcsec}(t/\tau)$",              relerr(s6),  COLORS[7], "--"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(betas, g_std, lw=2.5, color=COLORS[0], label=r"$\gamma$ (standard)")
    ax.plot(betas, sec_th, lw=1.5, color=COLORS[0], ls="--",
            label=r"$\sec\theta$ ($\theta=\arcsin\beta$)")
    ax.plot(betas, bg_std, lw=2.5, color=COLORS[1], label=r"$\beta\gamma$ (standard)")
    ax.plot(betas, tan_th, lw=1.5, color=COLORS[1], ls="--",
            label=r"$\tan\theta$ ($\theta=\arcsin\beta$)")
    ax.set_xlabel(r"$\beta = v/c$")
    ax.set_ylabel("Value")
    ax.set_title(r"(a) $\gamma\!=\!\sec\theta$, $\beta\gamma\!=\!\tan\theta$",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 12)

    ax2 = axes[1]
    for label, err, color, ls in entries:
        ax2.loglog(gammas, err, ls, lw=2, color=color, label=label)
    ax2.axhline(1e-15, color="gray", ls=":", lw=1.2, label="float64 precision")
    ax2.annotate(r"standard formula $\to\infty$" + "\n" +
                 r"($\beta$ rounds to 1 for $\gamma\gtrsim10^{8}$)",
                 xy=(1.2e8, 1e2), xytext=(2e4, 1e5), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax2.set_xlabel(r"$\gamma$ (Lorentz factor, log scale)")
    ax2.set_ylabel(r"Relative error $|\sec\theta_{\rm input}/\gamma - 1|$")
    ax2.set_title("(b) Error vs 60-digit reference, 9 decades of $\\gamma$",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left", labelspacing=1.4,
              handlelength=1.8, borderpad=0.8)

    fig.suptitle(
        "Algebraic equivalence and numerical stability of the "
        r"$\theta$-parametrization" + "\n"
        r"$\sec^2\theta - \tan^2\theta = 1 \;\Leftrightarrow\; "
        r"\gamma^2 - (\beta\gamma)^2 = 1$   |   "
        "inputs $E$, $K$, $p$, $\\lambda_{\\rm dB}$, $t/\\tau$ keep the information on $\\gamma$; "
        r"$\beta$-based inputs (standard, entry 1) degrade as $\beta\to1$",
        fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    save(fig, os.path.join(outdir, "figA1_equivalence_stability.png"), "A1")


# ══════════════════════════════════════════════════════════════════════════
#  FIG A2 — Computational performance: speedup and error (v1.8)
#
#  Equal footing (v1.8): every input mode starts from the RAW physical
#  quantity (v, E, K, pc, lambda_dB, t) and performs its own normalization
#  (one division by c, E0, lambda_C or tau) inside the timed region, and
#  every mode returns BOTH gamma = sec(theta) and beta*gamma = tan(theta).
#  Each input is evaluated along two paths:
#    arithmetic : closed form obtained from the theta identities
#                 (e.g. sec(arctan x) = sqrt(1+x^2)), theta never evaluated
#    theta path : theta evaluated explicitly (arcsin/arccos/arctan), then
#                 sec(theta) = 1/cos(theta) and tan(theta)
#  For input 1 (velocity) the arithmetic form IS the standard formula.
# ══════════════════════════════════════════════════════════════════════════
C_LIGHT = 299792458.0            # m/s (exact)
LAMBDA_C = 2.42631023538e-12     # m, electron Compton wavelength (CODATA 2018)
TAU_REF = 2.1969811e-6           # s, reference proper time (muon lifetime)

ENTRY_NAMES = [
    ("1", r"$v$",                r"$\theta=\arcsin(v/c)$"),
    ("2", r"$E$",                r"$\theta=\arccos(E_0/E)$"),
    ("3", r"$K$",                r"$\theta=\mathrm{arcsec}(1+K/E_0)$"),
    ("4", r"$p$",                r"$\theta=\arctan(pc/E_0)$"),
    ("5", r"$\lambda_{\rm dB}$", r"$\theta=\arctan(\lambda_C/\lambda_{\rm dB})$"),
    ("6", r"$t/\tau$",           r"$\theta=\mathrm{arcsec}(t/\tau)$"),
]


def raw_from_beta(b):
    """Raw physical inputs (float64) for an array of beta values."""
    g = 1.0 / np.sqrt(1.0 - b * b)
    return {"v": b * C_LIGHT, "E": E0 * g, "K": E0 * (g - 1.0),
            "pc": E0 * b * g, "lam": LAMBDA_C / (b * g), "t": TAU_REF * g}


def raw_exact(betas):
    """Raw inputs computed in 60 digits and rounded to float64, plus the
    exact gamma and beta*gamma (Decimal lists). The true beta is exactly
    the float64 value given."""
    c, e0, lc, tau = Decimal(C_LIGHT), Decimal(E0), Decimal(LAMBDA_C), Decimal(TAU_REF)
    one = Decimal(1)
    out = {k: [] for k in ("v", "E", "K", "pc", "lam", "t")}
    G_l, BG_l = [], []
    for b in betas:
        B = _dec(b)
        G = one / (one - B * B).sqrt()
        BG = B * G
        G_l.append(G); BG_l.append(BG)
        out["v"].append(float(B * c)); out["E"].append(float(e0 * G))
        out["K"].append(float(e0 * (G - one))); out["pc"].append(float(e0 * BG))
        out["lam"].append(float(lc / BG)); out["t"].append(float(tau * G))
    return {k: np.array(v) for k, v in out.items()}, G_l, BG_l


def make_methods(R):
    """Timed functions. Each receives nothing (closes over the raw arrays R),
    normalizes its own input and returns (gamma, beta*gamma)."""
    v, E, K, pc, lam, t = R["v"], R["E"], R["K"], R["pc"], R["lam"], R["t"]

    def std():
        b = v / C_LIGHT
        g = 1.0 / np.sqrt(1.0 - b * b)
        return g, b * g

    def th_out(th):
        return 1.0 / np.cos(th), np.tan(th)

    arith = {
        "1": std,                                         # identical to standard
        "2": lambda: (lambda x: (x, np.sqrt((x - 1.0) * (x + 1.0))))(E / E0),
        "3": lambda: (lambda k: (1.0 + k, np.sqrt(k * (k + 2.0))))(K / E0),
        "4": lambda: (lambda x: (np.sqrt(1.0 + x * x), x))(pc / E0),
        "5": lambda: (lambda x: (np.sqrt(1.0 + x * x), x))(LAMBDA_C / lam),
        "6": lambda: (lambda x: (x, np.sqrt((x - 1.0) * (x + 1.0))))(t / TAU_REF),
    }
    theta = {
        "1": lambda: th_out(np.arcsin(v / C_LIGHT)),
        "2": lambda: th_out(np.arccos(E0 / E)),
        "3": lambda: th_out(np.arccos(E0 / (E0 + K))),
        "4": lambda: th_out(np.arctan(pc / E0)),
        "5": lambda: th_out(np.arctan(LAMBDA_C / lam)),
        "6": lambda: th_out(np.arccos(TAU_REF / t)),
    }
    return std, arith, theta


def _max_rel(vals, exact):
    m = 0.0
    for s, x in zip(vals, exact):
        if not np.isfinite(s):
            return np.inf
        m = max(m, float(abs(_dec(s) - x) / x))
    return m


def error_table(betas):
    """Max relative error of gamma and beta*gamma for every method."""
    R, G, BG = raw_exact(betas)
    std, arith, theta = make_methods(R)
    res = {}
    g, bg = std()
    res["std"] = (_max_rel(g, G), _max_rel(bg, BG))
    for key in arith:
        for path, d in (("arith", arith), ("theta", theta)):
            g, bg = d[key]()
            res[(key, path)] = (_max_rel(g, G), _max_rel(bg, BG))
    return res


def run_benchmark(N=1_000_000, NTRIALS=15, NREP=5, seed=42):
    rng = np.random.default_rng(seed)
    b = rng.uniform(*PERF_BETA_RANGE, N)
    R = raw_from_beta(b)
    std, arith, theta = make_methods(R)

    timed = [("std", std)]
    for key, _, _ in ENTRY_NAMES:
        if key != "1":                      # arithmetic 1 == standard
            timed.append(((key, "arith"), arith[key]))
        timed.append(((key, "theta"), theta[key]))

    for _, f in timed:                      # shared warm-up pass
        f()
    times = {lbl: [] for lbl, _ in timed}
    for _trial in range(NTRIALS):           # paired trials, random order each
        for i in rng.permutation(len(timed)):
            lbl, f = timed[i]
            t0 = time.perf_counter()
            for _ in range(NREP):
                f()
            times[lbl].append((time.perf_counter() - t0) / NREP * 1000.0)

    std_t = np.array(times["std"])
    speed = {}
    for lbl, _ in timed:
        r = std_t / np.array(times[lbl])
        speed[lbl] = (float(np.median(r)), float(r.min()), float(r.max()))
    speed[("1", "arith")] = speed["std"]

    M = 2000
    err_main = error_table(b[:M])
    lo = np.sort(10.0 ** rng.uniform(-8, -2, 500))
    err_low = error_table(lo)
    return {"speed": speed, "err_main": err_main, "err_low": err_low,
            "N": N, "NTRIALS": NTRIALS, "M": M, "Mlow": len(lo)}


def fig_benchmark(outdir, bench):
    keys = [k for k, _, _ in ENTRY_NAMES]
    xt = ["Std"] + [f"{k}: {n}" for k, n, _ in ENTRY_NAMES]
    x = np.arange(len(xt))
    sp = bench["speed"]
    C_STD, C_AR, C_TH = COLORS[1], COLORS[0], COLORS[3]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # (a) speedup
    ax = axes[0]
    w = 0.38
    s = sp["std"]
    ax.bar(0, s[0], 0.6, color=C_STD, edgecolor="k", lw=0.5)
    for i, k in enumerate(keys, start=1):
        for off, path, col in ((-w / 2, "arith", C_AR), (w / 2, "theta", C_TH)):
            m, lo_, hi_ = sp[(k, path)]
            ax.bar(i + off, m, w, color=col, edgecolor="k", lw=0.5,
                   yerr=[[m - lo_], [hi_ - m]], capsize=3,
                   error_kw=dict(lw=1, ecolor="black"))
            ax.text(i + off, hi_ + 0.03, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=7.5, rotation=90)
    ax.axhline(1.0, color=C_STD, ls="--", lw=1.3)
    ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=9)
    ax.set_ylabel("Speedup relative to standard (median)")
    ax.set_title(f"(a) Speedup, median of {bench['NTRIALS']} trials", fontsize=11,
                 fontweight="bold")
    ax.set_ylim(0, max(v[2] for v in sp.values()) * 1.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_STD, label="Standard"),
                       Patch(color=C_AR, label="Arithmetic form"),
                       Patch(color=C_TH, label=r"Explicit $\theta$ path")],
              fontsize=8.5, loc="upper left")

    # (b), (c) errors
    for ax, key_err, title in ((axes[1], "err_main",
                                r"(b) Max rel. error, $\beta\in[0.01,0.9999]$"),
                               (axes[2], "err_low",
                                r"(c) Max rel. error, $\beta\in[10^{-8},10^{-2}]$")):
        E = bench[key_err]
        w = 0.2
        gs, bgs = E["std"]
        ax.bar(-w / 2, max(gs, 1e-18), w, color=C_STD, edgecolor="k", lw=0.5)
        ax.bar(w / 2, max(bgs, 1e-18), w, color=C_STD, edgecolor="k", lw=0.5,
               hatch="///", alpha=0.6)
        for i, k in enumerate(keys, start=1):
            vals = [(E[(k, "arith")][0], C_AR, None, 1.0),
                    (E[(k, "arith")][1], C_AR, "///", 0.6),
                    (E[(k, "theta")][0], C_TH, None, 1.0),
                    (E[(k, "theta")][1], C_TH, "///", 0.6)]
            for j, (v_, col, h, a) in enumerate(vals):
                ax.bar(i + (j - 1.5) * w, max(v_, 1e-18), w, color=col,
                       edgecolor="k", lw=0.5, hatch=h, alpha=a)
        ax.set_yscale("log")
        ax.axhline(2.2e-16, color="gray", ls=":", lw=1.3)
        ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=9)
        ax.set_ylabel("Max relative error vs 60-digit reference")
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ * 8)
        ax.set_title(title, fontsize=11, fontweight="bold")
    axes[1].legend(handles=[Patch(facecolor="white", edgecolor="k", label=r"$\gamma=\sec\theta$"),
                            Patch(facecolor="white", edgecolor="k", hatch="///",
                                  label=r"$\beta\gamma=\tan\theta$"),
                            Line2D([0], [0], color="gray", ls=":", label="float64 epsilon")],
                   fontsize=8.5, loc="upper right")

    fig.suptitle(
        r"Computational performance of the $\theta$-parametrization inputs — "
        "all inputs start from the raw physical quantity (normalization timed)\n"
        f"N = {bench['N']:,} values, β ∈ [0.01, 0.9999], float64, "
        f"{bench['NTRIALS']} trials in random order, warm-up excluded; "
        f"errors on {bench['M']} (b) and {bench['Mlow']} (c) samples",
        fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    save(fig, os.path.join(outdir, "figA2_performance.png"), "A2")


# ══════════════════════════════════════════════════════════════════════════
#  FIG A3 — heatmaps
# ══════════════════════════════════════════════════════════════════════════
def fig_heatmaps(outdir):
    beta = 0.7
    n_hat = np.array([2 / 7, 3 / 7, 6 / 7])
    L_std = boost_standard(beta, n_hat)                 # textbook form
    L_th = boost_theta_from_beta(beta, n_hat)           # θ form
    diff = np.abs(L_th - L_std)

    labels_rc = [r"$ct$", r"$x$", r"$y$", r"$z$"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    titles = [r"Standard $\Lambda(\gamma,\beta\gamma)$",
              r"θ-param. $\Lambda(\sec\theta,\tan\theta)$",
              r"$|\Lambda_\theta - \Lambda_{\rm std}|$"]
    for k, (ax, mat, title) in enumerate(zip(axes, [L_std, L_th, diff + 1e-18], titles)):
        is_diff = (k == 2)
        im = ax.imshow(mat, cmap="RdBu_r", aspect="equal",
                       norm=matplotlib.colors.LogNorm() if is_diff else None)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(labels_rc)
        ax.set_yticklabels(labels_rc)
        ax.set_title(title, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(4):
            for j in range(4):
                v = mat[i, j]
                if is_diff:
                    txt = "0" if diff[i, j] == 0.0 else f"{diff[i, j]:.2e}"
                else:
                    txt = f"{v:.3f}"
                if is_diff:
                    txt_color = "white"
                else:
                    vr = mat.max() - mat.min()
                    nz = (v - mat.min()) / (vr + 1e-18)
                    txt_color = "white" if (nz < 0.25 or nz > 0.75) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=txt_color, fontweight="bold")

    fig.suptitle(
        r"Lorentz transformation matrix 4×4, element-by-element comparison: $\beta=0.7$, "
        r"$\hat{n}=(2/7,3/7,6/7)$" + "\n"
        "Left: standard form | Center: θ-parametrization | "
        "Right: absolute difference",
        fontsize=10, fontweight="bold")
    plt.tight_layout()
    save(fig, os.path.join(outdir, "figA3_heatmaps.png"), "A3")


# ══════════════════════════════════════════════════════════════════════════
#  FIG A4 — validation for 6 directions x 3 velocities
# ══════════════════════════════════════════════════════════════════════════
def fig_validation(outdir, errs):
    tests = [("metric", "Metric\n" + r"$\Lambda(\theta)^\top\eta\,\Lambda(\theta)=\eta$"),
             ("group", "Group (collinear, along each $\\hat{n}$)\n"
                       r"$\Lambda_1(\theta)\Lambda_2(\theta)=\Lambda_{\rm res}(\theta)$"),
             ("inverse", "Inverse\n" + r"$\Lambda(\theta)\,\Lambda(-\theta)=I_4$")]
    dir_tex = [r"$\hat{x}$", r"$\hat{y}$", r"$\hat{z}$",
               r"$(1,1,0)/\sqrt{2}$", r"$(1,1,1)/\sqrt{3}$", r"$(2,3,6)/7$"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for (key, title), ax in zip(tests, axes):
        x = np.arange(len(DIRECTIONS))
        w = 0.25
        for i, (b, c) in enumerate(zip(BETAS_VALIDATION, COLORS)):
            vals = [max(errs[key][d][i], 1e-18) for d in DIRECTIONS]
            ax.bar(x + i * w, vals, w, label=rf"$\beta={b}$",
                   color=c, alpha=0.8, edgecolor="k", lw=0.4)
        ax.set_yscale("log")
        ax.set_xticks(x + w)
        ax.set_xticklabels(dir_tex, fontsize=8, rotation=15, ha="right")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Max absolute error")
        ax.axhline(2.2e-16, color="gray", ls=":", lw=1.5, label="float64 epsilon")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(
        r"Structural properties of the $\theta$-parametrized transformation matrix $\Lambda(\theta)$"
        "\n"
        "6 boost directions × 3 velocities (float64)",
        fontsize=11, fontweight="bold")
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, os.path.join(outdir, "figA4_validation.png"), "A4")


# ══════════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════════
def build_report(tens, ref, bench, errs, f32, small, comp):
    L = []
    L.append("REM-θ-framework — Lorentz boost: numerical results")
    L.append("=" * 62)
    L.append("(speedups are hardware-dependent; errors are deterministic)")
    L.append("")
    L.append(f"[1] Random-matrix test: N = {tens['N']:,}, random unit directions, "
             f"beta in [0.01, 0.9999] (seed {tens['seed']})")
    L.append(f"    metric residual  max(|L^T eta L - eta|): max = {tens['metric_max']:.2e}  "
             f"median = {tens['metric_median']:.2e}  p99 = {tens['metric_p99']:.2e}")
    L.append(f"    inverse residual max(|L(th)L(-th) - I|): max = {tens['inverse_max']:.2e}")
    L.append(f"    max (metric residual / gamma^2) = {tens['metric_over_gamma2_max']:.2e}"
             f"   [gamma at worst case = {tens['gamma_at_max']:.1f}; max gamma = {tens['gamma_max']:.1f}]")
    for lo, hi, m in tens["bands"]:
        L.append(f"    beta in [{lo:.2f}, {hi:.4f}]: max metric residual = {m:.2e}")
    L.append("")
    L.append(f"[2] float64 matrices vs 60-digit reference ({ref['nref']} random boosts)")
    L.append(f"    theta matrix    : max |diff| = {ref['theta_vs_exact']:.2e}")
    L.append(f"    standard matrix : max |diff| = {ref['standard_vs_exact']:.2e}")
    L.append("")
    L.append(f"[3] Benchmark (v1.8, equal footing): N = {bench['N']:,}, {bench['NTRIALS']} trials,"
             " random order; every input starts from the raw quantity")
    L.append("    (v, E, K, pc, lambda_dB, t) and returns gamma and beta*gamma; "
             "normalization is timed.")
    L.append("    Max relative errors vs 60-digit reference: gamma / beta*gamma, "
             f"main range ({bench['M']} samples) and low beta 1e-8..1e-2 ({bench['Mlow']} samples)")
    L.append(f"    {'input':<10s}{'path':<7s}{'speedup (min-max)':<22s}"
             f"{'err g main':>11s}{'err bg main':>12s}{'err g low':>11s}{'err bg low':>11s}")
    plain = {"1": "1 v", "2": "2 E", "3": "3 K", "4": "4 p", "5": "5 lam_dB", "6": "6 t/tau"}
    rows = [("std", "Standard", "-")]
    for k in plain:
        rows += [((k, "arith"), plain[k], "arith"), ((k, "theta"), plain[k], "theta")]
    for key, nm, path in rows:
        m, lo_, hi_ = bench["speed"][key]
        em, el = bench["err_main"][key], bench["err_low"][key]
        L.append(f"    {nm:<10s}{path:<7s}{m:5.2f}x ({lo_:4.2f}-{hi_:4.2f})      "
                 f"{em[0]:11.1e}{em[1]:12.1e}{el[0]:11.1e}{el[1]:11.1e}")
    L.append("    (arith of input 1 is the standard formula itself; inputs 4 and 5 run the same")
    L.append("     operations after normalization, so their difference is timing noise)")
    L.append("")
    L.append("[4] Structural validation (6 directions x 3 beta = 54 cases)")
    allmax = max(max(errs[t][d]) for t in errs for d in errs[t])
    for t in ("metric", "group", "inverse"):
        m = max(max(errs[t][d]) for d in errs[t])
        by_beta = [max(errs[t][d][i] for d in errs[t]) for i in range(len(BETAS_VALIDATION))]
        L.append(f"    {t:<8s} max = {m:.2e}   (by beta {BETAS_VALIDATION}: "
                 + ", ".join(f"{v:.1e}" for v in by_beta) + ")")
    below07 = max(errs[t][d][i] for t in errs for d in errs[t]
                  for i, b in enumerate(BETAS_VALIDATION) if b <= 0.7)
    L.append(f"    overall max = {allmax:.2e};  max for beta <= 0.7 = {below07:.2e}")
    L.append(f"    collinear composition rule tan(pi/4+theta/2): max relative deviation = {comp:.1e}")
    L.append("")
    L.append("[5] float32: relative error of gamma (input beta vs input p)")
    L.append("    gamma     standard(beta32)  arcsin(beta32)  sqrt(1+p32^2)")
    for g, a, b_, c in f32:
        L.append(f"    {g:6.0f}    {a:14.1e}  {b_:14.1e}  {c:13.1e}")
    L.append("")
    g1, tn = small
    L.append("[6] theta -> 0")
    L.append("    (gamma-1): naive (sec-1) vs stable t^2/(1+sec), relative error")
    for beta, a, b_ in g1:
        L.append(f"      beta = {beta:.0e}:  naive {a:.1e}   stable {b_:.1e}")
    L.append("    tan(theta) from K/E0 = x: naive sqrt(gamma^2-1) vs stable sqrt(x(x+2))")
    for kx, a, b_ in tn:
        L.append(f"      K/E0 = {kx:.0e}:  naive {a:.1e}   stable {b_:.1e}")
    return "\n".join(L) + "\n"


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="θ-parametrization validation, benchmark and figures")
    parser.add_argument("--n", type=int, default=1_000_000,
                        help="benchmark size (scalar gamma, beta*gamma)")
    parser.add_argument("--nmatrix", "--ntensor", dest="ntensor", type=int, default=1_000_000,
                        help="number of random 4x4 boosts in the random-matrix test")
    parser.add_argument("--nref", type=int, default=500,
                        help="samples compared with the 60-digit reference")
    parser.add_argument("--outdir", type=str, default=os.path.join(here, "figures"))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    say = (lambda *a, **k: None) if args.quiet else print

    say()
    say("  " + "═" * 68)
    say("  lorentz_theta_complete.py")
    say("  REM-θ-framework — Luís Gustavo Costa Reis")
    say("  " + "═" * 68)
    say(f"\n  Output directory : {args.outdir}")
    say(f"  Benchmark N      : {args.n:,}")
    say(f"  Random matrices  : {args.ntensor:,}\n")

    say("  Running numerical studies...")
    tens = tensor_random_test(args.ntensor)
    ref = tensor_reference_check(args.nref)
    errs = validation_errors()
    f32 = float32_study()
    small = small_theta_study()
    comp = composition_check()
    say("  Running benchmark...")
    bench = run_benchmark(args.n)

    if not args.skip_figures:
        for label, func in [
            ("Fig A1 — Equivalence and stability", lambda: fig_equivalence_stability(args.outdir)),
            ("Fig A2 — Performance (speedup/error)", lambda: fig_benchmark(args.outdir, bench)),
            ("Fig A3 — Matrix heatmaps", lambda: fig_heatmaps(args.outdir)),
            ("Fig A4 — Validation 6 directions", lambda: fig_validation(args.outdir, errs)),
        ]:
            say(f"  Generating {label}...")
            func()

    report = build_report(tens, ref, bench, errs, f32, small, comp)
    if not args.quiet:
        print()
        print(report)
    with open(os.path.join(here, "lorentz_results.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    say("  Saved: lorentz_results.txt")

    ok = (max(max(errs[t][d]) for t in errs for d in errs[t]) < 1e-12
          and tens["metric_over_gamma2_max"] < 1e-14)
    say(f"\n  {'ALL STRUCTURAL TESTS PASSED ✓' if ok else 'FAILURES DETECTED ✗'}\n")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
