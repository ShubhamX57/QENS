"""
Inference primitives: likelihood, prior, MAP search, plus utilities to bin
the data and build per-Q-bin resolution kernels.

"""
from __future__ import annotations

import csv
import os

import numpy as np
from scipy.optimize import curve_fit, minimize, nnls
from scipy.signal   import fftconvolve

from .config       import Config
from .models       import (
    lorentz, gnorm, predict_sqw,
    fickian_hwhm, ce_hwhm, ss_hwhm,
    bessel_weights, get_model,
)
from .models.forward import _make_resolution_kernel


__all__ = [
    "build_data_bins",
    "build_resolution_bins",
    "extract_hwhm",
    "save_hwhm_csv",
    "log_likelihood",
    "log_prior",
    "log_posterior",
    "find_map"
]




def _percentile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Equal occupancy bin edges via percentiles.
    
    """
    return np.percentile(values, np.linspace(0, 100, n_bins + 1))




def _average_bin(
    indices: np.ndarray,
    data_arr: np.ndarray,
    err_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean spectrum and combined error spectrum over a set of detector indices.

    """
    specs = np.array([data_arr[i] for i in indices])
    errs2 = np.array([err_arr[i]  for i in indices]) ** 2
    spec  = np.nanmean(specs, axis=0)
    errs  = np.sqrt(np.nanmean(errs2, axis=0))
    spec  = np.where(np.isfinite(spec), spec, 0.0)
    err_floor = max(spec.max() * 0.05, 1e-12)
    errs  = np.where(errs > 0, errs, err_floor)
    return spec, errs




def build_data_bins(
    d: dict,
    cfg: Config | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Bin a dataset into  "n_q_bins " equal occupancy Q-bins.

    Parameters
    ----------
    d : dict
        A processed dataset dict (must already have  "e ",  "good ",  "q ",
         "data ",  "errs ").

    cfg : Config

    Returns
    -------
    list of  "(omega, spec, errs, q_centre) " tuples — one per Q-bin —
    where  "omega " is restricted to ±cfg.energy_window and  "q_centre "
    is the mean Q of the detectors in the bin.

    """


    if cfg is None:
        cfg = Config()


    good  = d["good"].copy()
    q_arr = d["q"][good]
    e     = d["e"]


    q_mask = (q_arr >= cfg.q_min) & (q_arr <= cfg.q_max)
    good   = good[q_mask]
    q_arr  = q_arr[q_mask]


    emask = (e >= -cfg.energy_window) & (e <= cfg.energy_window)
    ew    = e[emask]
    edges = _percentile_edges(q_arr, cfg.n_q_bins)


    bins: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for k in range(cfg.n_q_bins):
        if k < cfg.n_q_bins - 1:
            mask = (q_arr >= edges[k]) & (q_arr < edges[k + 1])
        else:
            mask = (q_arr >= edges[k]) & (q_arr <= edges[k + 1])
        if mask.sum() < 2:
            continue
        idxs = good[mask]
        spec, errs = _average_bin(
            idxs,
            d["data"][:, emask],
            d["errs"][:, emask],
        )
        bins.append((ew, spec, errs, float(q_arr[mask].mean())))
    return bins




def build_resolution_bins(
    d_ref: dict,
    cfg: Config | None = None,
    q_centres: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    Build a measured per Q bin resolution kernel from a frozen sample.

    For each Q bin in the data, average the frozen sample spectra at
    detectors with similar Q to produce a measured resolution function.
    Use this kernel inside the forward model, it captures the Lorentzian
    wings of the real instrument that a pure-Gaussian fit misses.

    Parameters
    ----------
    d_ref : dict
        Resolution-reference dataset (typically the frozen low-T inc file).
        Must already have  "e ",  "good ",  "q ",  "data ".

    cfg : Config
    q_centres : array, optional
        If given, build kernels at these Q-centres instead of percentile bins.
        Useful when matching the binning of a different (target) dataset.

    Returns
    -------
    list of 1-D arrays, one per Q-bin, on the same ω grid as
    :func:`build_data_bins`.


    """


    if cfg is None:
        cfg = Config()
    good = d_ref["good"].copy()
    q_arr = d_ref["q"][good]
    e = d_ref["e"]
    q_mask = (q_arr >= cfg.q_min) & (q_arr <= cfg.q_max)
    good = good[q_mask]
    q_arr = q_arr[q_mask]
    emask = (e >= -cfg.energy_window) & (e <= cfg.energy_window)


    if q_centres is None:
        edges = _percentile_edges(q_arr, cfg.n_q_bins)
        n = cfg.n_q_bins
    else:
        # build edges centred on each Q-centre
        qc = np.asarray(q_centres, dtype=float)
        edges = np.empty(len(qc) + 1)
        edges[1:-1] = 0.5 * (qc[:-1] + qc[1:])
        edges[0]  = qc[0]  - 0.5 * (qc[1]  - qc[0])
        edges[-1] = qc[-1] + 0.5 * (qc[-1] - qc[-2])
        n = len(qc)


    out: list[np.ndarray] = []
    for k in range(n):
        if k < n - 1:
            mask = (q_arr >= edges[k]) & (q_arr < edges[k + 1])
        else:
            mask = (q_arr >= edges[k]) & (q_arr <= edges[k + 1])
        idxs = good[mask]
        if len(idxs) < 1:
            qc_k = 0.5 * (edges[k] + edges[k + 1])
            idxs = good[np.argsort(np.abs(q_arr - qc_k))[:5]]
        specs = np.array([d_ref["data"][i][emask] for i in idxs])
        specs = np.where(np.isfinite(specs), specs, 0.0)
        kernel = np.nanmean(specs, axis=0)
        kernel = np.where(kernel > 0, kernel, 0.0)
        out.append(kernel)
    return out



def extract_hwhm(
    d: dict,
    cfg: Config | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    
    Fit  "elastic + Lorentzian + bg " independently in each Q-bin.


    Parameters
    ----------
    d : dict
    cfg : Config

    Returns
    -------
    q_centres, hwhm, hwhm_err, eisf : 1-D arrays of length n_qbins (or fewer
    if some bins fail to fit).

    """
    if cfg is None:
        cfg = Config()


    good  = d["good"].copy()
    q_arr = d["q"][good]
    e     = d["e"]
    sr    = d["sigma_res"]


    q_mask = (q_arr >= cfg.q_min) & (q_arr <= cfg.q_max)
    good   = good[q_mask]
    q_arr  = q_arr[q_mask]


    emask = (e >= -cfg.energy_window) & (e <= cfg.energy_window)
    ew    = e[emask]
    edges = _percentile_edges(q_arr, cfg.n_q_bins)


    def _model(x, a_el, a_ql, gamma, bg):
        """
        elastic Gaussian + (Lorentz x Gauss) + bg, all peak normalised.
        
        """
        el = np.exp(-0.5 * (x / sr) ** 2)
        el = el / max(el.max(), 1e-30)
        g = max(gamma, 1e-5)
        dt = x[1] - x[0]
        ql = (1.0 / np.pi) * g / (x ** 2 + g ** 2)
        ql = fftconvolve(ql, el / max(el.sum() * dt, 1e-30),
                         mode="same") * dt
        ql = ql / max(ql.max(), 1e-30)
        return a_el * el + a_ql * ql + bg



    q_out, g_out, ge_out, eisf_out = [], [], [], []
    for k in range(cfg.n_q_bins):
        if k < cfg.n_q_bins - 1:
            sel = np.where((q_arr >= edges[k]) & (q_arr < edges[k + 1]))[0]
        else:
            sel = np.where((q_arr >= edges[k]) & (q_arr <= edges[k + 1]))[0]
        if len(sel) < 2:
            continue

        spec, errs = _average_bin(
            good[sel], d["data"][:, emask], d["errs"][:, emask]
        )
        peak = max(spec.max(), 1.0)
        p0 = [peak * 0.5, peak * 0.5, max(sr, 0.05), max(spec.min(), 0.0)]
        try:
            popt, pcov = curve_fit(
                _model, ew, spec, p0=p0, sigma=errs,
                bounds=([0, 0, sr * 0.1, 0],
                        [np.inf, np.inf, cfg.energy_window * 0.9, np.inf]),
                maxfev=8000,
            )
        except Exception:
            continue
        gamma = abs(popt[2])
        gerr = float(np.sqrt(pcov[2, 2])) if np.isfinite(pcov[2, 2]) else gamma * 0.1
        denom = popt[0] + popt[1]
        eisf = popt[0] / denom if denom > 0 else 0.5
        q_out.append(q_arr[sel].mean())
        g_out.append(gamma)
        ge_out.append(gerr)
        eisf_out.append(eisf)

    return (np.array(q_out), np.array(g_out),
            np.array(ge_out), np.array(eisf_out))



def save_hwhm_csv(
    q_centres: np.ndarray, hwhm: np.ndarray, hwhm_err: np.ndarray,
    eisf: np.ndarray, save_dir: str,
) -> str:
    """
    Write the HWHM table to "<save_dir>/hwhm_table.csv ".
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "hwhm_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q_centre_inv_angstrom",
                    "hwhm_mev", "hwhm_err_mev", "eisf"])
        for q, g, ge, ei in zip(q_centres, hwhm, hwhm_err, eisf):
            w.writerow([f"{q:.4f}", f"{g:.6f}", f"{ge:.6f}", f"{ei:.4f}"])
    return path




# Bayesian inference over registered ForwardModels

def _resolve_kernel_for_bin(sigma_res, k):
    """
    If sigma_res is a list/array of arrays, take the k-th one.
    
    """
    if isinstance(sigma_res, (list, tuple)):
        return sigma_res[k]
    if isinstance(sigma_res, np.ndarray) and sigma_res.ndim > 1:
        return sigma_res[k]
    return sigma_res




def log_prior(
    params: np.ndarray,
    model: str = "anisotropic_rotor",
) -> float:
    """
    
    Log-prior: 0 inside the registered model's prior box,  "-inf " outside.
    
    """
    fm = get_model(model)
    if fm.in_prior(np.asarray(params)):
        return 0.0
    return -np.inf




def log_likelihood(
    params: np.ndarray,
    data_bins: list,
    sigma_res,
    model: str = "anisotropic_rotor",
    **extras,
) -> float:
    """
    Joint X^2 log-likelihood across all Q-bins.

    For each bin a 2 column NNLS fits  "[overall_amp, flat_bg] ", so the
    physics determined Bessel weight ratios inside the forward model are
    preserved as a global Q-constraint.

    Parameters
    ----------
    params : array
        Parameter vector matching the registered model.

    data_bins : list of (omega, spec, errs, q)
        From :func:`build_data_bins`.

    sigma_res : float | array | list[array]
        Resolution: scalar Gaussian , single measured kernel, or one
        kernel per Q-bin (list of arrays).

    model : str
        Name of a registered forward model. Default  ""anisotropic_rotor" ".

    extras :
        Forwarded to the model's  "predict " callable.

    """

    fm = get_model(model)
    if not fm.in_prior(np.asarray(params)):
        return -np.inf
    if any(p <= 0 for p, lo in zip(params, fm.prior_lo) if lo > 0):
        return -np.inf


    extras_full = {**fm.extras, **extras}
    logl = 0.0
    for k, (omega, spec, errs, q) in enumerate(data_bins):
        sr_k = _resolve_kernel_for_bin(sigma_res, k)
        try:
            shape = fm.predict(omega, q, params, sr_k, **extras_full)
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(shape)):
            return -np.inf
        basis = np.column_stack([shape, np.ones_like(shape)])
        try:
            amp, _ = nnls(basis / errs[:, None], spec / errs)
        except Exception:
            return -np.inf
        resid = spec - basis @ amp
        logl -= 0.5 * np.sum((resid / errs) ** 2)
    return logl




def log_posterior(
    params, data_bins, sigma_res,
    model: str = "anisotropic_rotor",
    **extras,
) -> float:
    """
    Convenience wrapper: prior + likelihood.
    
    """
    lp = log_prior(params, model=model)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, data_bins, sigma_res,
                               model=model, **extras)




def find_map(
    data_bins: list,
    sigma_res,
    model: str = "anisotropic_rotor",
    cfg: Config | None = None,
    n_starts: int | None = None,
    verbose: bool = True,
    **extras,
) -> tuple[np.ndarray, float]:
    """
    Maximum- a posteriori search via Nelder Mead from random starts.

    Parameters
    ----------
    data_bins : list

    sigma_res : float | array | list[array]
    
    model : str
        Name of a registered forward model.

    cfg : Config

    n_starts : int, optional
        Random starts. If None uses  "cfg.n_map_starts ".

    verbose : bool

    extras :
        Forwarded to the model's  "predict " callable.

    Returns
    -------
    (params_map, neg_logp) : (ndarray, float)

    """

    if cfg is None:
        cfg = Config()
    fm = get_model(model)
    rng = np.random.default_rng(cfg.random_seed)
    n_starts = n_starts if n_starts is not None else cfg.n_map_starts


    def neg_lp(p):
        return -log_posterior(p, data_bins, sigma_res, model=model, **extras)


    best_val = np.inf
    best_p: np.ndarray | None = None
    for _ in range(n_starts):
        p0 = fm.random_in_prior(rng)
        res = minimize(neg_lp, p0, method="Nelder-Mead",
                       options={"maxiter": 30000,
                                "xatol": 1e-8, "fatol": 1e-8})
        if res.fun < best_val and np.isfinite(res.fun):
            best_val = float(res.fun)
            best_p   = np.asarray(res.x)


    if best_p is None:
        raise RuntimeError(
            f"MAP search failed for all {n_starts} starts — likelihood -inf "
            f"everywhere in the prior box. Check data, priors, and "
            f"resolution kernel."
        )


    if verbose:
        print(f"  MAP for model {model!r}:  -lnP = {best_val:.2f}")
        for n, v in zip(fm.param_names, best_p):
            print(f"      {n:<16} = {v:.5f}")


    return best_p, best_val
