"""
fitting.py — HWHM extraction and Bayesian posterior for diffusion models
=========================================================================

Two independent analysis paths live here:

1. HWHM extraction (frequentist cross-check)
   ------------------------------------------
   ``extract_hwhm`` — fits a phenomenological elastic + Lorentzian model to
   each Q-bin independently using scipy ``curve_fit``, returning Γ(Q) values
   that can be plotted against Q² as a quick sanity check before running MCMC.

   ``save_hwhm_csv`` — writes the extracted Γ(Q) table to disk.

2. Bayesian posterior (main analysis)
   ------------------------------------
   ``build_data_bins``  — bins and averages spectra for MCMC input.
   ``log_likelihood``   — Gaussian log-likelihood using NNLS-solved amplitudes.
   ``log_prior``        — uniform prior over a physically motivated (D, l) box.
   ``log_posterior``    — prior + likelihood.
   ``find_map``         — maximum a posteriori point estimate via Nelder-Mead.

Profile likelihood
------------------
The spectral amplitudes (a_el, a_ql, bg) are solved analytically by NNLS at
every candidate (D, l).  This marginalises them out of the sampling problem,
reducing it from 5D to 2D.  The MCMC sampler in sampling.py therefore only
ever evaluates the 2D log_posterior defined here.
"""

from __future__ import annotations

import csv
import os

import numpy as np
from scipy.optimize import curve_fit, minimize, nnls
from scipy.signal import fftconvolve

from .models import ce, gnorm, lorentz, make_basis
from .config import Config


# ═══════════════════════════════════════════════════════════════════════════════
# HWHM extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_hwhm(d: dict, cfg: Config | None = None):
    """
    Extract Γ(Q) by fitting a Lorentzian independently in each Q-bin.

    This is a fast, model-independent cross-check.  It does not use the CE
    model — it fits a phenomenological elastic + Lorentzian shape to each
    averaged bin spectrum and reports the best-fit HWHM gamma_val.

    The results are used to:
    - Plot Γ(Q) vs Q² for a quick visual assessment of diffusion regime.
    - Compute the EISF as a diagnostic for geometrically confined motion.
    - Cross-check the Bayesian MCMC posterior.

    Q-binning
    ---------
    Detectors are grouped into n_bins quantile bins (percentile edges).
    Quantile binning ensures roughly equal numbers of detectors per bin
    regardless of the angular distribution on the instrument.

    Per-bin model
    -------------
    S(ω) ≈ a_el · el(ω) + a_ql · ql(ω) + bg

    where:
        el(ω) = Gaussian at resolution width sigma_res (elastic component)
        ql(ω) = Lorentzian(gamma_val) convolved with el(ω) (quasi-elastic)
        bg    = flat background

    Parameters
    ----------
    d : dict
        Dataset dict from ``read_nxspe`` with sigma_res set by
        ``assign_resolution``.
    cfg : Config or None
        Analysis configuration.  Uses default Config() if None.

    Returns
    -------
    q_out : ndarray
        Q bin centres (Å⁻¹) for successfully fitted bins.
    g_out : ndarray
        Fitted HWHM Γ (meV) per bin.
    ge_out : ndarray
        Standard error on Γ per bin, from the curve_fit covariance matrix.
    eisf_out : ndarray
        Elastic incoherent structure factor per bin:
        EISF = a_el / (a_el + a_ql).
        Should decrease toward 0 with increasing Q for pure translational
        diffusion.  A plateau indicates confinement.
    """
    if cfg is None:
        cfg = Config()

    # ── Setup ─────────────────────────────────────────────────────────────────
    good  = d["good"].copy()    # copy to avoid mutating the original
    q_arr = d["q"][good]        # Q per good detector
    e     = d["e"]              # energy axis (shifted to ω = 0)
    sr    = d["sigma_res"]      # instrument resolution sigma (meV)

    # Apply Q-range cut from Config
    q_mask = (q_arr >= cfg.q_min) & (q_arr <= cfg.q_max)
    good   = good[q_mask]
    q_arr  = q_arr[q_mask]

    if len(good) < 4:
        print(f"  WARNING: only {len(good)} detectors in Q range [{cfg.q_min}, {cfg.q_max}]")

    # Energy window mask: only fit within ±ewin_hwhm
    emask   = (e >= -cfg.ewin_hwhm) & (e <= cfg.ewin_hwhm)
    ew      = e[emask]

    # Quantile bin edges — equal number of detectors per bin
    q_edges = np.percentile(q_arr, np.linspace(0, 100, cfg.n_bins + 1))

    # ── Per-bin model ─────────────────────────────────────────────────────────
    def _model(x, a_el, a_ql, gamma_val, bg):
        """
        Elastic + quasi-elastic + background model for one Q-bin.

        Parameters
        ----------
        x        : energy axis (meV)
        a_el     : elastic amplitude
        a_ql     : quasi-elastic amplitude
        gamma_val: Lorentzian HWHM (meV) — the parameter we extract
        bg       : flat background
        """
        # Elastic: resolution Gaussian, normalised to peak = 1
        el = np.exp(-0.5 * (x / sr) ** 2)
        el /= el.max() if el.max() > 0 else 1.0

        # Quasi-elastic: Lorentzian convolved with resolution Gaussian
        gamma_safe = max(gamma_val, 1e-5)   # prevent divide-by-zero
        dt  = x[1] - x[0]
        lor = (1.0 / np.pi) * gamma_safe / (x**2 + gamma_safe**2)
        # Convolve then multiply by dt to approximate the continuous integral
        ql  = fftconvolve(lor, el / (el.sum() * dt + 1e-30), mode="same") * dt
        ql /= ql.max() if ql.max() > 0 else 1.0

        return a_el * el + a_ql * ql + bg

    # ── Loop over Q-bins ──────────────────────────────────────────────────────
    q_out, g_out, ge_out, eisf_out = [], [], [], []

    for k in range(cfg.n_bins):
        # Last bin uses ≤ to include detectors at exactly q_max
        if k < cfg.n_bins - 1:
            in_bin = np.where((q_arr >= q_edges[k]) & (q_arr <  q_edges[k+1]))[0]
        else:
            in_bin = np.where((q_arr >= q_edges[k]) & (q_arr <= q_edges[k+1]))[0]

        if len(in_bin) < 2:
            continue  # not enough detectors to average meaningfully

        # ── Average spectra within bin ────────────────────────────────────────
        specs = np.array([d["data"][good[j]][emask] for j in in_bin])
        errs_ = np.array([d["errs"][good[j]][emask] for j in in_bin])

        spec = np.nanmean(specs, axis=0)
        spec = np.where(np.isfinite(spec), spec, 0.0)

        # Quadrature-averaged error; floor at 5% of peak to avoid zero-division
        err       = np.sqrt(np.nanmean(errs_**2, axis=0))
        err_floor = max(spec.max() * 0.05, 1e-12)
        err       = np.where(err > 0, err, err_floor)

        q_mid = q_arr[in_bin].mean()

        # ── curve_fit starting point and bounds ───────────────────────────────
        spec_peak = spec.max() if spec.max() > 0 else 1.0
        p0 = [
            spec_peak * 0.5,          # a_el  — roughly half the peak
            spec_peak * 0.5,          # a_ql  — roughly half the peak
            max(sr, 0.05),            # gamma — start at resolution width
            max(spec.min(), 0.0),     # bg    — start at spectral floor
        ]

        try:
            popt, pcov = curve_fit(
                _model, ew, spec, p0=p0, sigma=err,
                bounds=(
                    [0, 0, sr * 0.1, 0],                      # lower bounds
                    [np.inf, np.inf, cfg.ewin_hwhm * 0.9, np.inf],  # upper bounds
                ),
                maxfev=8000
            )

            gamma_val = abs(popt[2])
            # Error from diagonal of covariance matrix; fallback to 10% if non-finite
            gamma_err = (
                np.sqrt(pcov[2, 2]) if np.isfinite(pcov[2, 2]) else gamma_val * 0.1
            )

            # EISF = elastic fraction of total signal (excluding background)
            denom    = popt[0] + popt[1]
            eisf_val = popt[0] / denom if denom > 0 else 0.5

            q_out.append(q_mid)
            g_out.append(gamma_val)
            ge_out.append(gamma_err)
            eisf_out.append(eisf_val)

        except Exception as exc:
            print(f"  Q-bin {k} (Q≈{q_mid:.3f} Å⁻¹) fit failed: {exc}")

    return (
        np.array(q_out),
        np.array(g_out),
        np.array(ge_out),
        np.array(eisf_out),
    )


def save_hwhm_csv(q_hwhm, g_hwhm, g_err, eisf, save_dir: str) -> str:
    """
    Write the HWHM extraction results to a CSV file.

    Parameters
    ----------
    q_hwhm   : ndarray — Q bin centres (Å⁻¹)
    g_hwhm   : ndarray — fitted HWHM Γ (meV)
    g_err    : ndarray — standard error on Γ (meV)
    eisf     : ndarray — EISF values
    save_dir : str     — output directory (created if it doesn't exist)

    Returns
    -------
    str
        Full path to the written CSV file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "hwhm_table.csv")

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # Header row
        w.writerow(["q_centre_ainv", "hwhm_mev", "hwhm_err_mev", "eisf"])
        # One data row per Q-bin
        for q, g, ge, ei in zip(q_hwhm, g_hwhm, g_err, eisf):
            w.writerow([f"{q:.4f}", f"{g:.6f}", f"{ge:.6f}", f"{ei:.4f}"])

    print(f"  HWHM table saved → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Bayesian posterior
# ═══════════════════════════════════════════════════════════════════════════════

def build_data_bins(d_inc: dict, cfg: Config | None = None) -> list:
    """
    Build the list of Q-binned spectra used as input to the Bayesian fit.

    Each bin averages ~15–20 detector spectra, raising per-bin peak counts
    from ~100–200 (single detector) to ~2000–4000, which is necessary to
    resolve the Lorentzian shape above the noise.

    Parameters
    ----------
    d_inc : dict
        Dataset dict for the incoherent warm-sample run.
    cfg : Config or None

    Returns
    -------
    list of tuples
        Each tuple is (e_grid, spec, errs, q_mean):
            e_grid : ndarray — energy axis in the fitting window (meV)
            spec   : ndarray — mean spectrum across detectors in this bin
            errs   : ndarray — quadrature-averaged errors with 5% floor
            q_mean : float   — mean Q of detectors in this bin (Å⁻¹)
    """
    if cfg is None:
        cfg = Config()

    # ── Apply Q-range and energy window cuts ──────────────────────────────────
    good  = d_inc["good"].copy()
    q_g   = d_inc["q"][good]
    e     = d_inc["e"]

    q_mask = (q_g >= cfg.q_min) & (q_g <= cfg.q_max)
    good   = good[q_mask]
    q_g    = q_g[q_mask]

    emask = (e >= -cfg.ewin_mcmc) & (e <= cfg.ewin_mcmc)
    ew    = e[emask]

    # Quantile bin edges — equal detector count per bin
    q_edges = np.percentile(q_g, np.linspace(0, 100, cfg.n_bins_mc + 1))

    bins = []
    for k in range(cfg.n_bins_mc):
        if k < cfg.n_bins_mc - 1:
            mask = (q_g >= q_edges[k]) & (q_g <  q_edges[k+1])
        else:
            mask = (q_g >= q_edges[k]) & (q_g <= q_edges[k+1])

        if mask.sum() < 2:
            continue  # not enough detectors in this bin

        idxs  = good[mask]
        specs = np.array([d_inc["data"][i][emask] for i in idxs])
        errs_ = np.array([d_inc["errs"][i][emask] for i in idxs])

        # Mean spectrum and quadrature error across detectors in this bin
        spec = np.nanmean(specs, axis=0)
        errs = np.sqrt(np.nanmean(errs_**2, axis=0))
        spec = np.where(np.isfinite(spec), spec, 0.0)

        # Error floor at 5% of peak — prevents zero-division in chi-squared
        err_floor = max(spec.max() * 0.05, 1e-12)
        errs      = np.where(errs > 0, errs, err_floor)

        bins.append((ew, spec, errs, float(q_g[mask].mean())))

    print(f"  prepared {len(bins)} Q bins for MCMC")
    return bins


def log_likelihood(d_val: float, l: float, data_bins: list, sr: float) -> float:
    """
    Gaussian log-likelihood for the CE model at parameters (D, l).

    At each (D, l) the spectral amplitudes (a_el, a_ql, bg) are solved
    analytically by NNLS for every Q-bin.  This is a profile likelihood —
    the amplitudes are marginalised out exactly, reducing the sampling
    problem from 5D to 2D.

    Why NNLS instead of ordinary least squares?
    --------------------------------------------
    Amplitudes represent physical quantities (elastic fraction, quasi-elastic
    fraction, background level) that cannot be negative.  OLS does not enforce
    this constraint and can return negative amplitudes when one component is
    small, producing an unphysical model.  NNLS enforces a_el, a_ql, bg ≥ 0
    at negligible extra cost (only 3 basis functions).

    Parameters
    ----------
    d_val : float
        Trial diffusion coefficient (Å²/ps).
    l : float
        Trial jump length (Å).  Absolute value is taken internally.
    data_bins : list
        Output of ``build_data_bins``.
    sr : float
        Instrument resolution sigma (meV).

    Returns
    -------
    float
        Log-likelihood value.  Returns -inf for unphysical parameters.
    """
    # Guard: both parameters must be strictly positive
    if d_val <= 0 or abs(l) <= 0:
        return -np.inf

    logl = 0.0
    for e_grid, spec, errs, q_val in data_bins:
        # Build the [elastic | quasi-elastic | background] basis matrix
        basis = make_basis(e_grid, q_val, d_val, abs(l), sr)

        # Solve for non-negative amplitudes: min ||W(B·α − s)||²  s.t. α ≥ 0
        # W = diag(1/σ) is the inverse-error weighting matrix
        try:
            amp, _ = nnls(basis / errs[:, None], spec / errs)
        except Exception:
            return -np.inf

        # Residuals and Gaussian log-likelihood contribution from this bin
        resid = spec - basis @ amp
        logl -= 0.5 * np.sum((resid / errs)**2)

    return logl


def log_prior(d_val: float, l: float) -> float:
    """
    Uniform log-prior over a physically motivated (D, l) box.

    The prior is zero (log = 0) inside the box and −∞ outside.

    Bounds
    ------
    D ∈ (0, 3.0) Å²/ps
        Upper limit: D > 3 Å²/ps would be faster than small gas molecules —
        implausible for a liquid-phase sample.

    l ∈ (0.5, 6.0) Å
        Lower limit: below 0.5 Å the jump is smaller than an atomic radius —
        unphysical.
        Upper limit: above 6 Å the molecule would jump several diameters in
        one step — implausible for a compact liquid.

    Why uniform rather than log-uniform for l?
    -------------------------------------------
    Log-uniform is appropriate when l could span orders of magnitude.  Here
    the physically plausible range is less than one decade (0.5–6 Å), so
    uniform is a reasonable approximation and simpler to implement.

    Parameters
    ----------
    d_val : float
        Diffusion coefficient (Å²/ps).
    l : float
        Jump length (Å).

    Returns
    -------
    float
        0.0 inside the prior box, -np.inf outside.
    """
    if 0 < d_val < 3.0 and 0.5 < abs(l) < 6.0:
        return 0.0
    return -np.inf


def log_posterior(d_val: float, l: float, data_bins: list, sr: float) -> float:
    """
    Log-posterior = log-prior + log-likelihood.

    Short-circuits to -inf if the prior rejects the parameters (avoiding
    an expensive likelihood evaluation for out-of-bound proposals).

    Parameters
    ----------
    d_val, l : float
        Diffusion coefficient (Å²/ps) and jump length (Å).
    data_bins : list
        Output of ``build_data_bins``.
    sr : float
        Instrument resolution sigma (meV).

    Returns
    -------
    float
        Log-posterior value.
    """
    lp = log_prior(d_val, l)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(d_val, l, data_bins, sr)


def find_map(data_bins: list, sr: float, cfg: Config | None = None) -> tuple:
    """
    Find the Maximum A Posteriori (MAP) estimate of (D, l) via Nelder-Mead.

    Runs 20 optimisations from random starting points and returns the best
    result.  Multiple starts guard against convergence to a local maximum.

    The MAP serves two purposes:
    1. A quick point estimate reportable without running full MCMC.
    2. An initialisation point for the MCMC walkers — starting near the
       posterior peak dramatically reduces warmup time.

    Parameters
    ----------
    data_bins : list
        Output of ``build_data_bins``.
    sr : float
        Instrument resolution sigma (meV).
    cfg : Config or None

    Returns
    -------
    d_map : float
        MAP diffusion coefficient (Å²/ps).
    l_map : float
        MAP jump length (Å).
    tau_map : float
        MAP residence time τ = l²/(6D) (ps).
    """
    if cfg is None:
        cfg = Config()

    rng = np.random.default_rng(cfg.random_seed)

    def neg_lp(params):
        """Objective for minimisation: negative log-posterior."""
        return -log_posterior(params[0], abs(params[1]), data_bins, sr)

    best_val = np.inf
    best_p   = np.array([0.3, 2.0])  # reasonable fallback starting point

    print("  finding MAP (20 random starts) ...")
    for _ in range(20):
        # Random starting point drawn from within the prior box
        d0 = rng.uniform(0.05, 1.5)
        l0 = rng.uniform(1.0, 4.0)

        res = minimize(
            neg_lp, [d0, l0],
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-7, "fatol": 1e-7},
        )

        # Keep the result only if it is the best finite value seen so far
        if res.fun < best_val and np.isfinite(res.fun):
            best_val = res.fun
            best_p   = res.x

    d_map   = float(best_p[0])
    l_map   = abs(float(best_p[1]))          # ensure positive
    tau_map = l_map**2 / (6 * d_map)        # ps

    print(
        f"  MAP: D={d_map:.5f} Å²/ps  "
        f"l={l_map:.5f} Å  "
        f"τ={tau_map:.5f} ps"
    )
    return d_map, l_map, tau_map
