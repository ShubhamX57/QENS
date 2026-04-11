"""
HWHM extraction and Bayesian posterior for diffusion models.
"""

from __future__ import annotations
import csv
import os
import numpy as np
from scipy.optimize import curve_fit, minimize, nnls
from scipy.signal import fftconvolve
from .models import ce, gnorm, lorentz, make_basis
from .config import Config



# HWHM extraction

def extract_hwhm(d: dict, cfg: Config | None = None):
    if cfg is None:
        cfg = Config()

    good  = d["good"].copy()
    q_arr = d["q"][good]
    e     = d["e"]
    sr    = d["sigma_res"]

    q_mask = (q_arr >= cfg.q_min) & (q_arr <= cfg.q_max)
    good   = good[q_mask]
    q_arr  = q_arr[q_mask]

    if len(good) < 4:
        print(f"  WARNING: only {len(good)} detectors in Q range")

    emask   = (e >= -cfg.ewin_hwhm) & (e <= cfg.ewin_hwhm)
    ew      = e[emask]
    q_edges = np.percentile(q_arr, np.linspace(0, 100, cfg.n_bins + 1))

    def _model(x, a_el, a_ql, gamma_val, bg):
        el = np.exp(-0.5 * (x / sr) ** 2)
        el /= el.max() if el.max() > 0 else 1.0

        gamma_safe = max(gamma_val, 1e-5)
        dt  = x[1] - x[0]
        lor = (1.0 / np.pi) * gamma_safe / (x**2 + gamma_safe**2)
        ql  = fftconvolve(lor, el / (el.sum() * dt + 1e-30), mode="same") * dt
        ql /= ql.max() if ql.max() > 0 else 1.0
        return a_el * el + a_ql * ql + bg

    q_out, g_out, ge_out, eisf_out = [], [], [], []
    for k in range(cfg.n_bins):
        if k < cfg.n_bins - 1:
            in_bin = np.where((q_arr >= q_edges[k]) & (q_arr < q_edges[k+1]))[0]
        else:
            in_bin = np.where((q_arr >= q_edges[k]) & (q_arr <= q_edges[k+1]))[0]

        if len(in_bin) < 2:
            continue

        specs = np.array([d["data"][good[j]][emask] for j in in_bin])
        errs_ = np.array([d["errs"][good[j]][emask] for j in in_bin])

        spec = np.nanmean(specs, axis=0)
        spec = np.where(np.isfinite(spec), spec, 0.0)

        err = np.sqrt(np.nanmean(errs_**2, axis=0))
        err_floor = max(spec.max() * 0.05, 1e-12)
        err = np.where(err > 0, err, err_floor)

        q_mid = q_arr[in_bin].mean()
        spec_peak = spec.max() if spec.max() > 0 else 1.0
        p0 = [spec_peak * 0.5, spec_peak * 0.5, max(sr, 0.05), max(spec.min(), 0.0)]

        try:
            popt, pcov = curve_fit(
                _model, ew, spec, p0=p0, sigma=err,
                bounds=([0, 0, sr*0.1, 0], [np.inf, np.inf, cfg.ewin_hwhm*0.9, np.inf]),
                maxfev=8000
            )
            gamma_val = abs(popt[2])
            gamma_err = np.sqrt(pcov[2,2]) if np.isfinite(pcov[2,2]) else gamma_val * 0.1
            denom = popt[0] + popt[1]
            eisf_val = popt[0] / denom if denom > 0 else 0.5

            q_out.append(q_mid)
            g_out.append(gamma_val)
            ge_out.append(gamma_err)
            eisf_out.append(eisf_val)

        except Exception as exc:
            print(f"  Q-bin {k} (Q≈{q_mid:.3f}) fit failed: {exc}")

    return (np.array(q_out), np.array(g_out),
            np.array(ge_out), np.array(eisf_out))

def save_hwhm_csv(q_hwhm, g_hwhm, g_err, eisf, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "hwhm_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q_centre_ainv", "hwhm_mev", "hwhm_err_mev", "eisf"])
        for q, g, ge, ei in zip(q_hwhm, g_hwhm, g_err, eisf):
            w.writerow([f"{q:.4f}", f"{g:.6f}", f"{ge:.6f}", f"{ei:.4f}"])
    print(f"  HWHM table saved → {path}")
    return path



# Bayesian posterior

def build_data_bins(d_inc: dict, cfg: Config | None = None):
    if cfg is None:
        cfg = Config()

    good  = d_inc["good"].copy()
    q_g   = d_inc["q"][good]
    e     = d_inc["e"]

    q_mask = (q_g >= cfg.q_min) & (q_g <= cfg.q_max)
    good   = good[q_mask]
    q_g    = q_g[q_mask]

    emask   = (e >= -cfg.ewin_mcmc) & (e <= cfg.ewin_mcmc)
    ew      = e[emask]
    q_edges = np.percentile(q_g, np.linspace(0, 100, cfg.n_bins_mc + 1))

    bins = []
    for k in range(cfg.n_bins_mc):
        if k < cfg.n_bins_mc - 1:
            mask = (q_g >= q_edges[k]) & (q_g < q_edges[k+1])
        else:
            mask = (q_g >= q_edges[k]) & (q_g <= q_edges[k+1])

        if mask.sum() < 2:
            continue

        idxs = good[mask]
        specs = np.array([d_inc["data"][i][emask] for i in idxs])
        errs_ = np.array([d_inc["errs"][i][emask] for i in idxs])

        spec = np.nanmean(specs, axis=0)
        errs = np.sqrt(np.nanmean(errs_**2, axis=0))
        spec = np.where(np.isfinite(spec), spec, 0.0)
        err_floor = max(spec.max() * 0.05, 1e-12)
        errs = np.where(errs > 0, errs, err_floor)

        bins.append((ew, spec, errs, float(q_g[mask].mean())))

    print(f"  prepared {len(bins)} Q bins for MCMC")
    return bins

def log_likelihood(d_val: float, l: float, data_bins: list, sr: float):
    if d_val <= 0 or abs(l) <= 0:
        return -np.inf

    logl = 0.0
    for e_grid, spec, errs, q_val in data_bins:
        basis = make_basis(e_grid, q_val, d_val, abs(l), sr)
        try:
            amp, _ = nnls(basis / errs[:, None], spec / errs)
        except Exception:
            return -np.inf
        resid = spec - basis @ amp
        logl -= 0.5 * np.sum((resid / errs)**2)
    return logl

def log_prior(d_val: float, l: float):
    if 0 < d_val < 3.0 and 0.5 < abs(l) < 6.0:
        return 0.0
    return -np.inf

def log_posterior(d_val: float, l: float, data_bins: list, sr: float):
    lp = log_prior(d_val, l)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(d_val, l, data_bins, sr)

def find_map(data_bins: list, sr: float, cfg: Config | None = None):
    if cfg is None:
        cfg = Config()

    rng = np.random.default_rng(cfg.random_seed)

    def neg_lp(params):
        return -log_posterior(params[0], abs(params[1]), data_bins, sr)

    best_val = np.inf
    best_p   = np.array([0.3, 2.0])

    print("  finding MAP (20 random starts) ...")
    for _ in range(20):
        d0 = rng.uniform(0.05, 1.5)
        l0 = rng.uniform(1.0, 4.0)
        res = minimize(neg_lp, [d0, l0], method="Nelder-Mead",
                       options={"maxiter": 10000, "xatol": 1e-7, "fatol": 1e-7})
        if res.fun < best_val and np.isfinite(res.fun):
            best_val = res.fun
            best_p   = res.x

    d_map   = float(best_p[0])
    l_map   = abs(float(best_p[1]))
    tau_map = l_map**2 / (6 * d_map)

    print(f"  MAP: D={d_map:.5f} Å²/ps  l={l_map:.5f} Å  τ={tau_map:.5f} ps")
    return d_map, l_map, tau_map
