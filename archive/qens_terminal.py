#!/usr/bin/env python3
"""
qens_terminal.py 

"""

from __future__ import annotations


import argparse
import csv
import datetime
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import curve_fit, minimize, nnls
    from scipy.signal import fftconvolve
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import h5py
    _HAS_H5 = True
except ImportError:
    _HAS_H5 = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        TextColumn, TimeElapsedColumn, TaskProgressColumn,
    )
    from rich.live import Live
    from rich.text import Text
    from rich.rule import Rule
    from rich.columns import Columns
    from rich import box
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    print("Install rich for a better experience:  pip install rich")

try:
    import emcee as _emcee_mod
    _HAS_EMCEE = True
except ImportError:
    _HAS_EMCEE = False





console = Console(highlight=False) if _HAS_RICH else None

def _print(msg, style=""):
    if _HAS_RICH:
        console.print(msg, style=style)
    else:
        print(msg)

def _rule(title=""):
    if _HAS_RICH:
        console.print(Rule(title, style="bold blue"))
    else:
        print(f"\n{'─'*60} {title}")

def _ok(msg):    _print(f"  ✓ {msg}", "bold green")
def _warn(msg):  _print(f"  ⚠ {msg}", "bold yellow")
def _err(msg):   _print(f"  ✗ {msg}", "bold red")
def _info(msg):  _print(f"  · {msg}", "dim")



MN         = 1.67493e-27   # neutron mass           [kg]
HBAR       = 1.05457e-34   # reduced Planck         [J·s]
MEV_J      = 1.60218e-22   # 1 meV in joules        [J]
HBAR_MEVPS = 0.65821       # ħ in meV·ps

_GAMMA_FLOOR = 1e-5        # meV — Lorentzian floor for numerical stability



# DIFFUSION MODELS

def ce(q, d, l):
    """Chudley-Elliott HWHM Γ(Q) in meV.  τ = l²/(6D)."""
    l   = abs(l)
    tau = l**2 / (6.0 * d)
    return (HBAR_MEVPS / tau) * (1.0 - np.sinc(np.asarray(q) * l / np.pi))


def fickian(q, d):
    """Fickian (Brownian) HWHM Γ(Q) = ħ D Q² in meV."""
    return HBAR_MEVPS * d * np.asarray(q)**2


def ss_model(q, d, tau_s):
    """Singwi-Sjolander HWHM Γ(Q) in meV."""
    q = np.asarray(q)
    return HBAR_MEVPS * d * q**2 / (1.0 + d * q**2 * tau_s)



# LINESHAPE PRIMITIVES

def lorentz(w, gamma):
    """Normalised Lorentzian, area = 1."""
    gamma = max(float(gamma), _GAMMA_FLOOR)
    return (1.0 / np.pi) * gamma / (np.asarray(w)**2 + gamma**2)


def gnorm(w, sigma):
    """Normalised Gaussian, area = 1 (resolution function)."""
    return np.exp(-0.5 * (np.asarray(w) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))


def make_basis(e_grid, q_val, d, l, sigma_res):
    """
    3-column spectral basis: [elastic, quasi-elastic, background].
    Both signal columns are peak-normalised to 1.
    """
    dt    = e_grid[1] - e_grid[0]
    gamma = max(float(ce(q_val, d, l)), _GAMMA_FLOOR)

    el = gnorm(e_grid, sigma_res)
    el = el / el.max() if el.max() > 0 else el

    ql = fftconvolve(lorentz(e_grid, gamma), gnorm(e_grid, sigma_res),
                     mode="same") * dt
    ql = ql / ql.max() if ql.max() > 0 else np.ones_like(ql) / len(ql)

    return np.column_stack([el, ql, np.ones(len(e_grid))])


# IO  —  ISIS HDF5 .nxspe reader

def inspect_nxspe(path: str) -> None:
    """Print the full HDF5 tree of a .nxspe file."""
    if not _HAS_H5:
        _err("h5py is required:  pip install h5py"); return
    _rule(f"HDF5 tree: {Path(path).name}")
    def _v(name, obj):
        depth = name.count("/")
        leaf  = name.split("/")[-1]
        if hasattr(obj, "shape"):
            _info(f"{'  '*depth}{leaf}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            _info(f"{'  '*depth}{leaf}/")
    with h5py.File(path, "r") as hf:
        hf.visititems(_v)


def _read_ei(entry, parts):
    for key in ("efixed", "fixed_energy"):
        try:
            return float(entry["NXSPE_info"][key][()])
        except (KeyError, TypeError):
            pass
    try:
        return int(parts[2]) / 100.0
    except (ValueError, IndexError):
        raise ValueError("Cannot determine Ei from HDF5 or filename.")


def _read_polar(entry, fname):
    candidates = [
        ("data",                  "polar"),
        ("instrument/detector",   "polar"),
        ("instrument/detector_1", "polar_angle"),
    ]
    for grp_path, ds in candidates:
        try:
            grp = entry
            for part in grp_path.split("/"):
                grp = grp[part]
            arr = np.asarray(grp[ds], dtype=float)
            if arr.ndim == 1 and arr.size > 0:
                return arr
        except (KeyError, TypeError):
            pass
    # last resort: azimuthal (wrong but better than crashing)
    try:
        arr = np.asarray(entry["data"]["azimuthal"], dtype=float)
        if arr.ndim == 1 and arr.size > 0:
            warnings.warn(f"Using azimuthal as polar proxy for '{fname}'",
                          UserWarning, stacklevel=3)
            return arr
    except (KeyError, TypeError):
        pass
    raise ValueError(
        f"Cannot locate polar angles in '{fname}'.\n"
        "Expected: <entry>/data/polar  (standard Mantid SaveNXSPE).\n"
        "Run with --inspect to see the actual HDF5 layout."
    )


def read_nxspe(path: str) -> dict:
    """Read a single ISIS Mantid-reduced .nxspe (HDF5) file."""
    if not _HAS_H5:
        raise ImportError("h5py is required:  pip install h5py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    name  = os.path.basename(path)
    parts = name.replace(".nxspe", "").split("_")
    if len(parts) < 4:
        raise ValueError(f"Filename '{name}' doesn't match <sample>_<T>_<Ei*100>_<kind>.nxspe")

    try:
        temp = int(parts[1])
    except ValueError:
        raise ValueError(f"Cannot parse temperature from '{name}'")
    kind = parts[3]

    with h5py.File(path, "r") as hf:
        entry_key = next(iter(hf.keys()))
        entry     = hf[entry_key]
        ei        = _read_ei(entry, parts)
        edges     = np.asarray(entry["data"]["energy"], dtype=float)
        e_raw     = 0.5 * (edges[:-1] + edges[1:])
        data      = np.asarray(entry["data"]["data"],  dtype=float)
        errs      = np.asarray(entry["data"]["error"], dtype=float)
        two_theta = _read_polar(entry, name)

    data = np.where(np.isfinite(data), data, 0.0)
    errs = np.where(np.isfinite(errs) & (errs > 0), errs, 0.0)

    good_mask = np.sum((data > 0) & np.isfinite(data), axis=1) > data.shape[1] // 2
    good = np.where(good_mask)[0]
    if good.size == 0:
        raise ValueError(f"No usable detectors in '{name}'. Check data reduction.")

    ki = np.sqrt(2.0 * MN * ei * MEV_J) / HBAR * 1e-10
    q  = 2.0 * ki * np.sin(np.radians(two_theta / 2.0))

    return dict(name=name, temp=temp, ei=ei, kind=kind,
                e_raw=e_raw, e=e_raw.copy(),
                data=data, errs=errs, good=good, q=q, format="hdf5")


def load_dataset(file_list, data_dir=".", critical_files=None):
    """Batch-load .nxspe files.  Returns {filename: dict}."""
    if critical_files is None:
        critical_files = []
    dataset = {}
    for fname in file_list:
        full = os.path.join(data_dir, fname)
        if not os.path.exists(full):
            if fname in critical_files:
                raise FileNotFoundError(f"Critical file not found: {full}")
            _warn(f"skipping (not found): {fname}"); continue
        try:
            d = read_nxspe(full)
            dataset[fname] = d
            _ok(f"[hdf5] {fname}  Ei={d['ei']:.2f} meV  "
                f"T={d['temp']} K  good={len(d['good'])} det")
        except Exception as exc:
            if fname in critical_files: raise
            _warn(f"failed: {fname} — {exc}")
    if not dataset:
        raise RuntimeError("No files loaded. Check --data_dir and filenames.")
    return dataset


# PREPROCESSING

def fit_elastic_peak(d: dict) -> tuple[float, float]:
    """Fit Gaussian to elastic peak; shift energy axis to ω = 0."""
    good  = d["good"]
    e     = d["e_raw"]
    n_low = max(3, len(good) // 7)
    avg   = np.nanmean([d["data"][i] for i in good[:n_low]], axis=0)
    avg   = np.where(np.isfinite(avg), avg, 0.0)

    def _gauss(x, a, mu, sigma, bg):
        return a * np.exp(-0.5 * ((x - mu) / sigma)**2) + bg

    pk = int(np.argmax(avg))
    try:
        popt, _ = curve_fit(
            _gauss, e, avg,
            p0=[avg[pk], e[pk], 0.05, max(avg.min(), 0.0)],
            bounds=([0, e[0], 1e-4, -np.inf], [np.inf, e[-1], 2.0, np.inf]),
            maxfev=8000,
        )
        e0, sig = float(popt[1]), abs(float(popt[2]))
    except Exception:
        e0, sig = float(e[pk]), 0.043

    d["e0"]      = e0
    d["sig_raw"] = sig
    d["e"]       = e - e0
    return e0, sig


def assign_resolution(dataset: dict, res_key: str | None = None) -> None:
    """Assign σ_res to every dataset entry (frozen → COH → raw INC)."""
    res_frozen: dict[float, float] = {}

    if res_key and res_key in dataset:
        d_ref = dataset[res_key]
        res_frozen[d_ref["ei"]] = d_ref["sig_raw"]
        _ok(f"Resolution reference: {res_key}  "
            f"FWHM={d_ref['sig_raw']*2355:.1f} µeV")
    else:
        for d in dataset.values():
            if d["kind"] == "inc" and d["temp"] <= 270:
                res_frozen[d["ei"]] = d["sig_raw"]
                _ok(f"Frozen reference (auto): {d['name']}  "
                    f"T={d['temp']} K  FWHM={d['sig_raw']*2355:.1f} µeV")

    coh_sigma = {d["ei"]: d["sig_raw"]
                 for d in dataset.values() if d["kind"] == "coh"}

    for fname, d in dataset.items():
        ei = d["ei"]
        if ei in res_frozen:
            d["sigma_res"], d["res_source"] = res_frozen[ei], "frozen sample"
        elif ei in coh_sigma:
            d["sigma_res"], d["res_source"] = coh_sigma[ei], "COH file"
        else:
            d["sigma_res"], d["res_source"] = d["sig_raw"], "raw INC (inflated)"
            _warn(f"No resolution ref for {fname} — using raw INC")
        d["fwhm_res"] = 2.355 * d["sigma_res"]


# HWHM EXTRACTION

def extract_hwhm(d: dict, cfg: dict) -> tuple:
    """Curve-fit each Q bin; return (q, gamma, gamma_err, eisf)."""
    good  = d["good"].copy()
    q_arr = d["q"][good]
    e     = d["e"]
    sr    = d["sigma_res"]
    ewin  = cfg["ewin_hwhm"]
    nbins = cfg["n_bins"]

    q_mask = (q_arr >= cfg["q_min"]) & (q_arr <= cfg["q_max"])
    good, q_arr = good[q_mask], q_arr[q_mask]

    emask   = (e >= -ewin) & (e <= ewin)
    ew      = e[emask]
    q_edges = np.percentile(q_arr, np.linspace(0, 100, nbins + 1))

    def _model(x, a_el, a_ql, gval, bg):
        el = np.exp(-0.5 * (x / sr)**2)
        if el.max() > 0: el /= el.max()
        g_s = max(gval, 1e-5)
        dt  = x[1] - x[0]
        lor = (1 / np.pi) * g_s / (x**2 + g_s**2)
        s   = el.sum() * dt
        ql  = fftconvolve(lor, el / (s if s > 0 else 1), mode="same") * dt
        if ql.max() > 0: ql /= ql.max()
        return a_el * el + a_ql * ql + bg

    q_out, g_out, ge_out, eisf_out = [], [], [], []

    for k in range(nbins):
        hi = q_edges[k+1]
        if k == nbins - 1:
            mask = (q_arr >= q_edges[k]) & (q_arr <= hi)
        else:
            mask = (q_arr >= q_edges[k]) & (q_arr < hi)
        if mask.sum() < 2: continue

        specs = np.array([d["data"][good[j]][emask] for j in np.where(mask)[0]])
        errs_ = np.array([d["errs"][good[j]][emask] for j in np.where(mask)[0]])
        spec  = np.nanmean(specs, axis=0)
        err   = np.sqrt(np.nanmean(errs_**2, axis=0))
        spec  = np.where(np.isfinite(spec), spec, 0.0)
        ef    = max(spec.max() * 0.05, 1e-12)
        err   = np.where(err > 0, err, ef)
        pk    = spec.max() if spec.max() > 0 else 1.0

        try:
            popt, pcov = curve_fit(
                _model, ew, spec,
                p0=[pk * 0.5, pk * 0.5, max(sr, 0.05), max(spec.min(), 0)],
                sigma=err,
                bounds=([0, 0, sr * 0.1, 0], [np.inf, np.inf, ewin * 0.9, np.inf]),
                maxfev=8000,
            )
            gamma_val = abs(popt[2])
            gamma_err = np.sqrt(pcov[2, 2]) if np.isfinite(pcov[2, 2]) else gamma_val * 0.1
            denom     = popt[0] + popt[1]
            eisf_val  = popt[0] / denom if denom > 0 else 0.5
            q_out.append(q_arr[mask].mean())
            g_out.append(gamma_val)
            ge_out.append(gamma_err)
            eisf_out.append(eisf_val)
        except Exception as exc:
            _warn(f"Q-bin {k} fit failed: {exc}")

    return (np.array(q_out), np.array(g_out),
            np.array(ge_out), np.array(eisf_out))


def fit_model_to_hwhm(q_arr, g_arr, model_name):
    """Least-squares fit of one diffusion model to Γ(Q)."""
    try:
        if model_name == "ce":
            p, pc = curve_fit(ce, q_arr, g_arr,
                              p0=[0.30, 2.5], bounds=([0, 0.5], [3, 6]))
            return {"D": p[0], "l": p[1], "tau": p[1]**2/(6*p[0]), "cov": pc.tolist()}
        elif model_name == "fickian":
            p, pc = curve_fit(fickian, q_arr, g_arr,
                              p0=[0.30], bounds=([0], [3]))
            return {"D": p[0], "cov": [[pc[0, 0]]]}
        elif model_name == "ss_model":
            p, pc = curve_fit(ss_model, q_arr, g_arr,
                              p0=[0.30, 1.0], bounds=([0, 0.01], [3, 20]))
            return {"D": p[0], "tau_s": p[1], "cov": pc.tolist()}
    except Exception as exc:
        return {"error": str(exc)}


# BAYESIAN POSTERIOR

def build_data_bins(d_inc: dict, cfg: dict) -> list:
    """Package Q-binned spectra for the MCMC likelihood."""
    good  = d_inc["good"].copy()
    q_g   = d_inc["q"][good]
    e     = d_inc["e"]
    ewin  = cfg["ewin_mcmc"]
    nbins = cfg["n_bins_mc"]

    q_mask = (q_g >= cfg["q_min"]) & (q_g <= cfg["q_max"])
    good, q_g = good[q_mask], q_g[q_mask]

    emask   = (e >= -ewin) & (e <= ewin)
    ew      = e[emask]
    q_edges = np.percentile(q_g, np.linspace(0, 100, nbins + 1))

    bins = []
    for k in range(nbins):
        if k < nbins - 1:
            mask = (q_g >= q_edges[k]) & (q_g < q_edges[k+1])
        else:
            mask = (q_g >= q_edges[k]) & (q_g <= q_edges[k+1])
        if mask.sum() < 2: continue

        idxs = good[mask]
        spec = np.nanmean([d_inc["data"][i][emask] for i in idxs], axis=0)
        errs = np.sqrt(np.nanmean([d_inc["errs"][i][emask]**2 for i in idxs], axis=0))
        spec = np.where(np.isfinite(spec), spec, 0.0)
        ef   = max(spec.max() * 0.05, 1e-12)
        errs = np.where(errs > 0, errs, ef)
        bins.append((ew, spec, errs, float(q_g[mask].mean())))
    return bins


def log_likelihood(d_val, l, data_bins, sr):
    if d_val <= 0 or abs(l) <= 0: return -np.inf
    logl = 0.0
    for e_grid, spec, errs, q_val in data_bins:
        basis = make_basis(e_grid, q_val, d_val, abs(l), sr)
        try:
            amp, _ = nnls(basis / errs[:, None], spec / errs)
        except Exception:
            return -np.inf
        resid  = spec - basis @ amp
        logl  -= 0.5 * np.sum((resid / errs)**2)
    return logl


def log_prior(d_val, l):
    return 0.0 if (0 < d_val < 3 and 0.5 < abs(l) < 6) else -np.inf


def log_posterior(d_val, l, data_bins, sr):
    lp = log_prior(d_val, l)
    return -np.inf if not np.isfinite(lp) else lp + log_likelihood(d_val, l, data_bins, sr)


def find_map(data_bins, sr, cfg):
    """Multi-start Nelder-Mead MAP estimation."""
    rng = np.random.default_rng(cfg["random_seed"])

    def neg_lp(p): return -log_posterior(p[0], abs(p[1]), data_bins, sr)

    best_val, best_p = np.inf, np.array([0.3, 2.0])
    for _ in range(20):
        d0, l0 = rng.uniform(0.05, 1.5), rng.uniform(1.0, 4.0)
        res = minimize(neg_lp, [d0, l0], method="Nelder-Mead",
                       options={"maxiter": 10_000, "xatol": 1e-7, "fatol": 1e-7})
        if res.fun < best_val and np.isfinite(res.fun):
            best_val, best_p = res.fun, res.x

    d_map, l_map = float(best_p[0]), abs(float(best_p[1]))
    tau_map = l_map**2 / (6.0 * d_map)
    return d_map, l_map, tau_map


# SAMPLING

def gelman_rubin(chains):
    m, n = len(chains), min(len(c) for c in chains)
    chains = [np.asarray(c[:n]) for c in chains]
    w = np.mean([c.var(ddof=1) for c in chains])
    if w == 0: return float("nan")
    b = n * np.array([c.mean() for c in chains]).var(ddof=1)
    return float(np.sqrt(((1 - 1/n) * w + b/n) / w))


def run_mcmc(data_bins, sr, d_map, l_map, cfg, progress=None, task_id=None):
    """Run MCMC; returns (samples, rhat_d, rhat_l)."""
    ndim    = 2
    rng     = np.random.default_rng(cfg["random_seed"])

    def log_prob(params):
        return log_posterior(params[0], abs(params[1]), data_bins, sr)

    if _HAS_EMCEE:
        p0 = [np.array([d_map, l_map]) * (1 + rng.normal(0, 0.05, ndim))
              for _ in range(cfg["n_walkers"])]
        p0 = [np.clip(p, [1e-3, 0.6], [2.9, 5.9]) for p in p0]
        sampler = _emcee_mod.EnsembleSampler(cfg["n_walkers"], ndim, log_prob)
        total   = cfg["n_warmup"] + cfg["n_keep"]

        for i, _ in enumerate(sampler.sample(p0, iterations=total)):
            if progress and task_id is not None:
                progress.update(task_id, completed=i + 1, total=total)

        samples = sampler.get_chain(discard=cfg["n_warmup"],
                                    thin=cfg["thin"], flat=True)
        acc     = float(np.mean(sampler.acceptance_fraction))
        _info(f"emcee acceptance fraction: {acc:.3f}  (target 0.2–0.5)")

        # per-walker chains for Gelman-Rubin
        chain_arr = sampler.get_chain(discard=cfg["n_warmup"], thin=cfg["thin"])
        rhat_d = gelman_rubin([chain_arr[:, w, 0] for w in range(cfg["n_walkers"])])
        rhat_l = gelman_rubin([chain_arr[:, w, 1] for w in range(cfg["n_walkers"])])

    else:
        # Metropolis-Hastings fallback (4 chains)
        _warn("emcee not found — using MH fallback (pip install emcee)")
        step    = np.array([d_map * 0.1, 0.1])
        n_total = cfg["n_warmup"] + cfg["n_keep"]
        chains  = []

        for cid in range(4):
            start  = np.clip(
                np.array([d_map, l_map]) * (1 + rng.normal(0, 0.05, 2)),
                [1e-3, 0.6], [2.9, 5.9]
            )
            d_c, l_c = start
            cur_lp = log_posterior(d_c, l_c, data_bins, sr)
            samps  = [(d_c, l_c)]
            n_acc  = 0
            for i in range(n_total):
                d_n = d_c + rng.normal(0, step[0])
                l_n = np.exp(np.log(l_c) + rng.normal(0, step[1]))
                new_lp = log_posterior(d_n, l_n, data_bins, sr)
                log_a  = new_lp - cur_lp + (np.log(l_n) - np.log(l_c))
                if np.log(rng.random() + 1e-300) < log_a:
                    d_c, l_c = d_n, l_n; cur_lp = new_lp; n_acc += 1
                samps.append((d_c, l_c))
                if progress and task_id is not None and cid == 0:
                    progress.update(task_id, completed=i+1,
                                    total=n_total * 4)
            chains.append(np.array(samps[cfg["n_warmup"]::cfg["thin"]]))

        samples = np.vstack(chains)
        rhat_d  = gelman_rubin([c[:, 0] for c in chains])
        rhat_l  = gelman_rubin([c[:, 1] for c in chains])

    samples[:, 1] = np.abs(samples[:, 1])
    return samples, rhat_d, rhat_l


# FIGURES

_CMAP = LinearSegmentedColormap.from_list(
    "qens",
    ["#0a0e1a", "#0c2d6b", "#1565c0", "#42a5f5",
     "#e3f2fd", "#ff8f00", "#e65100"],
    N=512,
) if _HAS_MPL else None

_MODEL_COLORS = {
    "ce":       "#c0392b",
    "fickian":  "#2471a3",
    "ss_model": "#1e8449",
}
_MODEL_LABELS = {
    "ce":       "Chudley-Elliott",
    "fickian":  "Fickian",
    "ss_model": "Singwi-Sjolander",
}


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, path, dpi=200):
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def fig_data_preview(d, save_path=None):
    """Two-panel data quality overview: S(Q,ω) map + elastic peak + detector strip."""
    good  = d["good"]
    q_arr = d["q"][good]
    e     = d["e"]
    ei    = d["ei"]
    ewin  = min(ei * 0.45, 1.2)

    emask = (e >= -ewin) & (e <= ewin)
    qs    = np.argsort(q_arr)
    img   = d["data"][np.ix_(good, emask)]
    img   = np.where(np.isfinite(img) & (img > 0), img, np.nan)
    ism   = gaussian_filter(np.where(np.isfinite(img[qs]), img[qs], 0.0),
                            sigma=[1.5, 0.8])
    ism[ism <= 0] = np.nan
    vmin  = max(np.nanpercentile(ism, 2), 1e-8)
    vmax  = np.nanpercentile(ism, 99)

    n_lo  = max(3, len(good) // 3)
    avg   = np.nanmean([d["data"][good[j]] for j in range(n_lo)], axis=0)
    avg   = np.where(np.isfinite(avg), avg, 0.0)
    em2   = (e >= -ewin * 0.6) & (e <= ewin * 0.6)

    emask_el = (e >= -0.15) & (e <= 0.15)
    ph = np.array([d["data"][i][emask_el].max()
                   if emask_el.any() else 0.0 for i in good])
    ph = np.where(np.isfinite(ph), ph, 0.0)

    fig = plt.figure(figsize=(14, 5.2))
    gs  = fig.add_gridspec(2, 2, width_ratios=[1, 1],
                            height_ratios=[5, 1], hspace=0.08, wspace=0.34)
    ax_map   = fig.add_subplot(gs[:, 0])
    ax_peak  = fig.add_subplot(gs[0, 1])
    ax_strip = fig.add_subplot(gs[1, 1])

    # S(Q,w) map
    im = ax_map.pcolormesh(e[emask], q_arr[qs], ism, cmap=_CMAP,
                           norm=LogNorm(vmin=vmin, vmax=vmax),
                           shading="auto", rasterized=True)
    ax_map.axvline(0, color="#aaa", lw=0.8, ls="--", alpha=0.6)
    ax_map.set_xlabel(r"$\hbar\omega$ (meV)", fontsize=11)
    ax_map.set_ylabel(r"$Q$ (Å$^{-1}$)", fontsize=11)
    ax_map.set_title(rf"$S(Q,\omega)$ — {d['name']}", fontsize=10, pad=7)
    cb = fig.colorbar(im, ax=ax_map, pad=0.02, fraction=0.038)
    cb.set_label("intensity", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _despine(ax_map)

    # elastic peak
    norm_pk = avg[em2].max() if avg[em2].max() > 0 else 1.0
    ax_peak.fill_between(e[em2], 0, avg[em2] / norm_pk, alpha=0.3, color="#2471a3")
    ax_peak.plot(e[em2], avg[em2] / norm_pk, color="#2471a3", lw=1.8)
    ax_peak.axvline(0, color="#aaa", lw=0.8, ls="--", alpha=0.6)
    ax_peak.set_ylabel("normalised intensity", fontsize=9)
    fwhm_str  = f"{d.get('fwhm_res', 0)*1000:.0f} µeV" if "fwhm_res" in d else "—"
    res_src   = d.get("res_source", "—")
    ax_peak.set_title(f"Elastic peak (low-Q avg)  FWHM≈{fwhm_str}  [{res_src}]",
                      fontsize=8.5, pad=6)
    ax_peak.set_xlim(e[em2][0], e[em2][-1])
    ax_peak.set_ylim(bottom=-0.04)
    ax_peak.grid(True, alpha=0.18, lw=0.6)
    plt.setp(ax_peak.get_xticklabels(), visible=False)
    _despine(ax_peak)

    # detector strip
    strip = ph[np.argsort(q_arr)][np.newaxis, :]
    ax_strip.imshow(strip, aspect="auto", cmap="plasma",
                    extent=[q_arr.min(), q_arr.max(), 0, 1],
                    vmin=0, vmax=max(ph.max(), 1e-8))
    ax_strip.set_xlabel(r"$Q$ (Å$^{-1}$)", fontsize=9)
    ax_strip.set_yticks([])
    ax_strip.set_title("detector peak intensity  (dark = weak/masked)", fontsize=8, pad=4)
    fig.tight_layout(pad=1.2)

    if save_path: _save(plt.gcf(), save_path)
    return fig


def fig_hwhm(q_hwhm, g_hwhm, g_err, model_results, d_inc,
             samples=None, save_path=None):
    """Γ(Q²) with data, all fitted model curves, and optional posterior fan."""
    q_f  = np.linspace(max(q_hwhm.min() * 0.85, 0.05),
                        q_hwhm.max() * 1.10, 350)
    q2f  = q_f**2
    q2d  = q_hwhm**2

    fig, ax = plt.subplots(figsize=(9, 6))

    # posterior fan (CE)
    if samples is not None:
        d_s, l_s = samples[:, 0], np.abs(samples[:, 1])
        g_fan = ce(q_f[None, :], d_s[:, None], l_s[:, None]) * 1000
        ax.fill_between(q2f, np.percentile(g_fan, 2.5, axis=0),
                        np.percentile(g_fan, 97.5, axis=0),
                        alpha=0.2, color=_MODEL_COLORS["ce"],
                        label=f"CE 95% posterior (n={len(d_s)})")

    for model, res in model_results.items():
        if "error" in res: continue
        col = _MODEL_COLORS.get(model, "#333")
        lbl = _MODEL_LABELS.get(model, model)
        D   = res.get("D", 0.3)
        L   = res.get("l", 2.5)
        ts  = res.get("tau_s", 1.0)
        c2  = res.get("_chi2r")
        suf = rf"  ($\chi^2_r={c2:.3f}$)" if c2 is not None else ""
        if   model == "ce":       y = ce(q_f, D, L) * 1000
        elif model == "fickian":  y = fickian(q_f, D) * 1000
        elif model == "ss_model": y = ss_model(q_f, D, ts) * 1000
        else: continue
        ax.plot(q2f, y, "-", color=col, lw=2.2, zorder=5,
                label=f"{lbl}  D={D:.4f}{suf}")

    ax.errorbar(q2d, g_hwhm * 1000, yerr=2 * g_err * 1000,
                fmt="o", color="#111", ms=6, capsize=3.5, elinewidth=1.5,
                label=r"Data ±2σ", zorder=6)

    res_hwhm = d_inc["fwhm_res"] / 2 * 1000
    ax.axhline(res_hwhm, color="#888", ls=":", lw=1.3,
               label=rf"Resolution HWHM = {res_hwhm:.0f} µeV")
    ax.axhspan(0, res_hwhm * 1.1, alpha=0.04, color="#888")

    ax.set_xlabel(r"$Q^2$ (Å$^{-2}$)", fontsize=12)
    ax.set_ylabel(r"$\Gamma(Q)$ (µeV)", fontsize=12)
    ax.set_title(r"$\Gamma(Q)$ vs $Q^2$ — model comparison", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(left=-0.04)
    ax.set_ylim(bottom=-10)
    ax.grid(True, alpha=0.18)
    _despine(ax)
    plt.tight_layout()

    if save_path: _save(fig, save_path)
    return fig


def fig_spectrum(d_inc, d_map, l_map, q_target=1.06, ewin=0.8, save_path=None):
    """Single-Q spectrum vs CE MAP model with residuals."""
    gp    = d_inc["good"]
    qg    = d_inc["q"][gp]
    sr    = d_inc["sigma_res"]
    emask = (d_inc["e"] >= -ewin) & (d_inc["e"] <= ewin)
    ew    = d_inc["e"][emask]

    near = np.where(np.abs(qg - q_target) < 0.10)[0]
    if len(near) == 0: near = np.argsort(np.abs(qg - q_target))[:4]

    spec = np.nanmean([d_inc["data"][gp[j]][emask] for j in near], axis=0)
    errs = np.sqrt(np.nanmean([d_inc["errs"][gp[j]][emask]**2 for j in near], axis=0))
    spec = np.where(np.isfinite(spec), spec, 0.0)
    errs = np.where(errs > 0, errs, spec.max() * 0.05)
    sn, en = spec / spec.max(), errs / spec.max()

    wf  = np.linspace(-ewin, ewin, 1000)
    dt  = wf[1] - wf[0]
    gamma = float(ce(q_target, d_map, l_map))
    el  = gnorm(wf, sr); el /= el.max()
    ql_ = fftconvolve(lorentz(wf, gamma), gnorm(wf, sr), mode="same") * dt
    ql  = ql_ / ql_.max() if ql_.max() > 0 else ql_
    amp, _ = nnls(np.column_stack([el, ql, np.ones(len(wf))]),
                  np.interp(wf, ew, sn))
    fit    = amp[0]*el + amp[1]*ql + amp[2]
    fit_d  = np.interp(ew, wf, fit)
    resid  = (sn - fit_d) / en
    chi2r  = float(np.sum(resid**2) / max(len(resid) - 4, 1))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5),
                              gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    axes[0].fill_between(wf, amp[2], amp[0]*el + amp[2], alpha=0.22,
                         color="#2471a3", label="Elastic")
    axes[0].fill_between(wf, amp[2], amp[1]*ql + amp[2], alpha=0.22,
                         color="#e65100", label="Quasi-elastic")
    axes[0].errorbar(ew, sn, yerr=en, fmt=".", color="#333", ms=3.5,
                     elinewidth=0.7, alpha=0.8, label=f"Data  Q≈{q_target:.2f} Å⁻¹")
    axes[0].plot(wf, fit, "-", color="#c0392b", lw=2.2,
                 label=rf"CE MAP  $\chi^2_r={chi2r:.2f}$")
    axes[0].annotate("", xy=(gamma, 0.5), xytext=(0, 0.5),
                     arrowprops=dict(arrowstyle="<->", color="#e67e22", lw=1.8))
    axes[0].text(gamma/2, 0.56, f"HWHM={gamma*1000:.0f} µeV",
                 ha="center", color="#e67e22", fontsize=9.5, fontweight="bold")
    axes[0].set_ylabel(r"$S(Q,\omega)$ normalised", fontsize=11)
    axes[0].set_title(rf"Single spectrum  $Q\approx{q_target:.2f}$ Å$^{{-1}}$", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.18); _despine(axes[0])

    axes[1].axhline(0, color="#555", lw=0.9)
    axes[1].axhline(2, color="#999", ls="--", lw=0.7)
    axes[1].axhline(-2, color="#999", ls="--", lw=0.7)
    axes[1].plot(ew, resid, ".", color="#c0392b", ms=3.5, alpha=0.85)
    axes[1].set_xlabel(r"$\hbar\omega$ (meV)", fontsize=11)
    axes[1].set_ylabel(r"Residual ($\sigma$)", fontsize=10)
    axes[1].set_ylim(-4.8, 4.8)
    axes[1].grid(True, alpha=0.18); _despine(axes[1])

    fig.tight_layout(h_pad=0.4)
    if save_path: _save(fig, save_path)
    return fig, chi2r


def fig_posteriors(samples, d_map, l_map, save_path=None):
    """Marginal posterior histograms for D, l, τ."""
    d_s   = samples[:, 0]
    l_s   = np.abs(samples[:, 1])
    tau_s = l_s**2 / (6 * d_s)
    tau_map = l_map**2 / (6 * d_map)

    params = [
        (d_s,   d_map,   "D (Å²/ps)",  "#c0392b"),
        (l_s,   l_map,   "ℓ (Å)",       "#1e8449"),
        (tau_s, tau_map, "τ (ps)",       "#e67e22"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (arr, mp, xlabel, col) in zip(axes, params):
        med = float(np.median(arr))
        lo, hi = np.percentile(arr, [2.5, 97.5])
        cnt, _, _ = ax.hist(arr, bins=80, density=True,
                            color=col, alpha=0.8, edgecolor="white", lw=0.25)
        pk = cnt.max()
        ax.axvspan(lo, hi, alpha=0.18, color=col)
        ax.axvline(med, color="#111", lw=2.5, label=f"median={med:.4f}")
        ax.axvline(mp,  color=col, lw=1.8, ls="--", label=f"MAP={mp:.4f}")
        ax.annotate("", xy=(hi, pk*1.08), xytext=(lo, pk*1.08),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.8))
        ax.text((lo+hi)/2, pk*1.15, f"95% CI [{lo:.4f}, {hi:.4f}]",
                ha="center", fontsize=8, color=col, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("density", fontsize=10)
        ax.set_title(xlabel, fontsize=12, color=col, fontweight="bold")
        ax.set_ylim(0, pk*1.30)
        ax.legend(fontsize=8, framealpha=0.92)
        ax.grid(True, alpha=0.18); _despine(ax)
    fig.suptitle("Bayesian posteriors — CE model", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path: _save(fig, save_path)
    return fig


def fig_joint_posterior(samples, d_map, l_map, save_path=None):
    """2-D joint posterior (D, l) with KDE contours."""
    d_s = samples[:, 0]
    l_s = np.abs(samples[:, 1])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d_s), min(3000, len(d_s)), replace=False)
    ax.scatter(d_s[idx], l_s[idx], c="#2471a3", alpha=0.12, s=4, rasterized=True)
    ax.axvline(d_map, color="#c0392b", lw=1.8, ls="--",
               label=f"D MAP={d_map:.4f}")
    ax.axhline(l_map, color="#1e8449", lw=1.8, ls="--",
               label=f"ℓ MAP={l_map:.4f}")
    try:
        from scipy.stats import gaussian_kde
        xy  = np.vstack([d_s, l_s])
        kde = gaussian_kde(xy)
        dg  = np.linspace(d_s.min(), d_s.max(), 80)
        lg  = np.linspace(l_s.min(), l_s.max(), 80)
        D, L = np.meshgrid(dg, lg)
        Z    = kde(np.vstack([D.ravel(), L.ravel()])).reshape(D.shape)
        zf   = np.sort(Z.ravel())[::-1]
        cdf  = np.cumsum(zf) / zf.sum()
        l68  = zf[np.searchsorted(cdf, 0.68)]
        l95  = zf[np.searchsorted(cdf, 0.95)]
        ax.contour(D, L, Z, levels=[l95, l68],
                   colors=[_MODEL_COLORS["ce"], "#c0392b"],
                   linewidths=[1.2, 2.0])
    except Exception:
        pass
    ax.set_xlabel(r"$D$ (Å²/ps)", fontsize=12)
    ax.set_ylabel(r"$\ell$ (Å)", fontsize=12)
    ax.set_title("Joint posterior (D, ℓ)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.18); _despine(ax)
    plt.tight_layout()
    if save_path: _save(fig, save_path)
    return fig


# RESULTS TABLE 

def print_results_table(d_inc, model_results, samples, d_map, l_map, method):
    if not _HAS_RICH: return
    t = Table(title="QENS Analysis Results", box=box.ROUNDED,
              header_style="bold white on dark_blue", show_lines=True)
    t.add_column("Parameter", style="bold", min_width=22)
    t.add_column("Value",     min_width=18)
    t.add_column("Unit",      min_width=14)
    t.add_column("Notes",     min_width=28, style="dim")

    def _add(p, v, u="", n=""):
        t.add_row(p, str(v), u, n)
    def _sec(label):
        t.add_section()
        t.add_row(f"[bold cyan]{label}[/]", "", "", "")
        t.add_section()

    _sec("Dataset")
    _add("File",        d_inc["name"])
    _add("Temperature", f"{d_inc['temp']} K")
    _add("Ei",          f"{d_inc['ei']:.2f}", "meV")
    _add("Resolution",  f"{d_inc['fwhm_res']*1000:.1f}", "µeV FWHM",
         d_inc["res_source"])
    _add("Method",      "Bayesian MCMC" if method == "bayes" else "Least Squares")

    for model, res in model_results.items():
        lbl = _MODEL_LABELS.get(model, model)
        _sec(lbl)
        if "error" in res:
            _add("Status", f"[red]Failed: {res['error']}[/]")
            continue

        if method == "bayes" and samples is not None and model == "ce":
            d_s   = samples[:, 0]
            l_s_  = np.abs(samples[:, 1])
            tau_s = l_s_**2 / (6 * d_s)
            for arr, nm, unit in [
                (d_s,   "D",  "Å² ps⁻¹"),
                (l_s_,  "ℓ",  "Å"),
                (tau_s, "τ",  "ps"),
            ]:
                med = np.median(arr)
                lo, hi = np.percentile(arr, [2.5, 97.5])
                _add(nm, f"{med:.5f}", unit, f"95% CI [{lo:.5f}, {hi:.5f}]")
        else:
            D = res.get("D")
            L = res.get("l")
            if D is not None: _add("D",   f"{D:.5f}", "Å² ps⁻¹")
            if L is not None:
                _add("ℓ",   f"{L:.5f}", "Å")
                tau = L**2 / (6*D) if D else 0
                _add("τ",   f"{tau:.5f}", "ps")
            ts = res.get("tau_s")
            if ts is not None: _add("τₛ", f"{ts:.5f}", "ps")
        c2 = res.get("_chi2r")
        if c2 is not None: _add("χ²ᵣ", f"{c2:.4f}", "", "ideal ≈ 1")

    console.print(t)



def save_hwhm_csv(q, g, ge, eisf, out_dir):
    path = os.path.join(out_dir, "hwhm_table.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["q_centre_ainv", "hwhm_mev", "hwhm_err_mev", "eisf"])
        for row in zip(q, g, ge, eisf):
            w.writerow([f"{v:.6f}" for v in row])
    return path


def save_json(d_inc, model_results, samples, method, chi2r, cfg, out_dir):
    def _ci(a):
        lo, hi = np.percentile(a, [2.5, 97.5])
        return {"median": float(np.median(a)), "lo95": float(lo), "hi95": float(hi)}

    out = {
        "timestamp":    datetime.datetime.now().isoformat(),
        "dataset":      d_inc["name"],
        "temperature_K": int(d_inc["temp"]),
        "Ei_meV":       float(d_inc["ei"]),
        "method":       method,
        "res_fwhm_meV": float(d_inc["fwhm_res"]),
        "res_source":   d_inc["res_source"],
        "spectrum_chi2r": float(chi2r),
        "config":       cfg,
        "models": {},
    }
    for model, res in model_results.items():
        entry = {"ls": {k: float(v) for k, v in res.items()
                        if k not in ("cov", "error", "_chi2r") and
                        not isinstance(v, list)}}
        if "_chi2r" in res:
            entry["gamma_chi2r"] = float(res["_chi2r"])
        if samples is not None and model == "ce":
            d_s  = samples[:, 0]
            l_s_ = np.abs(samples[:, 1])
            ts   = l_s_**2 / (6 * d_s)
            entry["bayesian"] = {
                "n_samples": int(len(d_s)),
                "D": _ci(d_s), "l": _ci(l_s_), "tau": _ci(ts),
            }
        out["models"][model] = entry

    path = os.path.join(out_dir, "results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    return path


def save_config(cfg, out_dir):
    path = os.path.join(out_dir, "run_config.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    return path


# DEFAULT CONFIG

DEFAULT_CONFIG = {
    "files_to_fit":  [],           # set from --files
    "primary_file":  None,         # set from --primary or auto-detected
    "q_min":         0.30,
    "q_max":         2.50,
    "ewin_hwhm":     0.80,
    "ewin_mcmc":     0.80,
    "n_bins":        13,
    "n_bins_mc":     10,
    "n_walkers":     32,
    "n_warmup":      500,
    "n_keep":        2000,
    "thin":          5,
    "random_seed":   42,
    "models":        ["ce", "fickian"],
    "q_target":      1.06,
    "save_dir":      "results",
}


# PIPELINE

def run(cfg: dict, no_mcmc: bool = False) -> None:
    run_ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg["save_dir"], run_ts)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    t_start = time.perf_counter()

    if _HAS_RICH:
        console.print(Panel.fit(
            "[bold white]QENS Analysis Pipeline[/]\n"
            "[dim]ISIS HDF5 · Bayesian CE inference · terminal UI[/]",
            border_style="bold blue", padding=(0, 4)
        ))
    else:
        print("=" * 60)
        print("  QENS Analysis Pipeline")
        print("=" * 60)

    save_config(cfg, out_dir)

    _rule("1 · Loading data")
    data_dir = cfg.get("data_dir", ".")
    files    = cfg["files_to_fit"]

    # auto-detect frozen reference
    res_candidates = []
    for f in files:
        parts = f.replace(".nxspe", "").split("_")
        if len(parts) >= 4:
            try:
                if int(parts[1]) <= 270:
                    res_candidates.append(f)
            except ValueError:
                pass
    # add guesses for frozen references
    for f in list(files):
        p = f.replace("_290_", "_260_")
        if p not in files and p not in res_candidates:
            res_candidates.append(p)

    all_files = list(dict.fromkeys(files + res_candidates))
    dataset   = load_dataset(all_files, data_dir=data_dir, critical_files=files)

    _rule("2 · Preprocessing")
    for fname, d in dataset.items():
        e0, sig = fit_elastic_peak(d)
        _info(f"{fname:<44}  e0={e0:+.3f} meV  σ={sig*1000:.1f} µeV")
    assign_resolution(dataset)

    # primary file
    prim = cfg.get("primary_file") or next(
        (f for f in files if "inc" in f.lower()), files[0])
    if prim not in dataset:
        prim = next(iter(dataset))
        _warn(f"Primary not in dataset — using {prim}")
    d_inc = dataset[prim]
    _ok(f"Primary: {d_inc['name']}")

    _rule("3 · Data preview")
    if _HAS_MPL:
        fig_data_preview(d_inc,
                         save_path=os.path.join(fig_dir, "data_preview.pdf"))
        _ok(f"Data preview → {fig_dir}/data_preview.pdf")
    else:
        _warn("matplotlib not available — skipping figures")

    _rule("4 · HWHM extraction")
    q_hwhm, g_hwhm, g_err, eisf = extract_hwhm(d_inc, cfg)
    if len(q_hwhm) == 0:
        _err("No HWHM bins converged — widen Q range or energy window."); return
    csv_path = save_hwhm_csv(q_hwhm, g_hwhm, g_err, eisf, out_dir)
    _ok(f"{len(q_hwhm)} Q bins  |  "
        f"Γ = {g_hwhm.min()*1000:.0f}–{g_hwhm.max()*1000:.0f} µeV  "
        f"→ {csv_path}")

    _rule("5 · Least-squares model fitting")
    model_results: dict = {}
    best_d, best_l, best_chi2r = 0.30, 2.50, np.inf

    for model in cfg["models"]:
        lbl = _MODEL_LABELS.get(model, model)
        res = fit_model_to_hwhm(q_hwhm, g_hwhm, model)
        model_results[model] = res
        if "error" in res:
            _warn(f"{lbl}: {res['error']}"); continue
        D  = res.get("D", 0.3)
        L  = res.get("l", 2.5)
        ts = res.get("tau_s", 1.0)
        _ok(f"{lbl:<20}  D={D:.5f}  ℓ={L:.5f} Å"
            + (f"  τₛ={ts:.4f} ps" if model == "ss_model" else ""))
        # compute chi²ᵣ against HWHM
        try:
            if   model == "ce":       pred = ce(q_hwhm, D, L)
            elif model == "fickian":  pred = fickian(q_hwhm, D)
            elif model == "ss_model": pred = ss_model(q_hwhm, D, ts)
            else: continue
            c2r = float(np.sum((g_hwhm - pred)**2
                               / np.where(pred > 0, pred**2, 1e-30))
                        / max(len(q_hwhm) - 2, 1))
            res["_chi2r"] = c2r
            if c2r < best_chi2r:
                best_chi2r, best_d, best_l = c2r, D, L
        except Exception:
            pass

    samples     = None
    rhat_d      = rhat_l = float("nan")

    if "ce" in cfg["models"] and not no_mcmc:
        _rule("6 · MAP estimation")
        bins  = build_data_bins(d_inc, cfg)
        sr    = d_inc["sigma_res"]
        if _HAS_RICH:
            with console.status("[cyan]Finding MAP (20 random starts)…[/]"):
                d_map, l_map, tau_map = find_map(bins, sr, cfg)
        else:
            _info("Finding MAP …")
            d_map, l_map, tau_map = find_map(bins, sr, cfg)

        _ok(f"MAP:  D={d_map:.5f} Å²/ps  ℓ={l_map:.5f} Å  τ={tau_map:.5f} ps")

        _rule("7 · Bayesian MCMC")
        total_steps = cfg["n_warmup"] + cfg["n_keep"]
        sampler_lbl = "emcee" if _HAS_EMCEE else "MH fallback"
        _info(f"Sampler: {sampler_lbl}  |  "
              f"{cfg['n_walkers']} walkers × {total_steps} steps")

        if _HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as prog:
                tid = prog.add_task("MCMC sampling…", total=total_steps)
                t0  = time.perf_counter()
                samples, rhat_d, rhat_l = run_mcmc(
                    bins, sr, d_map, l_map, cfg, progress=prog, task_id=tid)
        else:
            t0 = time.perf_counter()
            samples, rhat_d, rhat_l = run_mcmc(bins, sr, d_map, l_map, cfg)

        elapsed = time.perf_counter() - t0
        _ok(f"{len(samples)} posterior samples in {elapsed:.1f} s")
        _info(f"Gelman-Rubin R̂:  D={rhat_d:.4f}  ℓ={rhat_l:.4f}  (< 1.01 = good)")
        if max(rhat_d, rhat_l) > 1.05:
            _warn("R̂ > 1.05 — chains may not have converged; try more steps")

        # MAP may be better starting point for the spectrum plot
        best_d, best_l = d_map, l_map
        method = "bayes"
    else:
        d_map = l_map = best_d
        method = "ls"
        if no_mcmc:
            _info("--no_mcmc: skipping MAP & MCMC")

    _rule("8 · Spectrum fit + residuals")
    chi2r = 1.0
    if _HAS_MPL:
        fig_sp, chi2r = fig_spectrum(
            d_inc, best_d, best_l,
            q_target=cfg["q_target"],
            save_path=os.path.join(fig_dir, "spectrum_fit.pdf"),
        )
        _ok(f"Spectrum fit  χ²ᵣ={chi2r:.3f}  "
            f"(Q={cfg['q_target']:.2f} Å⁻¹)  "
            f"→ {fig_dir}/spectrum_fit.pdf")

    _rule("9 · Γ(Q²) comparison figure")
    if _HAS_MPL:
        fig_hwhm(q_hwhm, g_hwhm, g_err, model_results, d_inc,
                 samples=samples,
                 save_path=os.path.join(fig_dir, "hwhm_vs_q2.pdf"))
        _ok(f"Γ(Q²) figure → {fig_dir}/hwhm_vs_q2.pdf")

    if samples is not None and _HAS_MPL:
        _rule("10 · Posterior figures")
        fig_posteriors(samples, d_map, l_map,
                       save_path=os.path.join(fig_dir, "posteriors.pdf"))
        _ok(f"Posterior histograms → {fig_dir}/posteriors.pdf")
        fig_joint_posterior(samples, d_map, l_map,
                            save_path=os.path.join(fig_dir, "joint_posterior.pdf"))
        _ok(f"Joint posterior → {fig_dir}/joint_posterior.pdf")
        np.savez(os.path.join(out_dir, "posterior_samples.npz"),
                 D=samples[:, 0], l=np.abs(samples[:, 1]))

    _rule("11 · Results summary")
    print_results_table(d_inc, model_results, samples, best_d, best_l, method)
    json_path = save_json(
        d_inc, model_results, samples, method, chi2r, cfg, out_dir)

    total = time.perf_counter() - t_start
    _rule()

    if _HAS_RICH:
        console.print(Panel(
            f"[bold green]Analysis complete in {total:.1f} s[/]\n\n"
            f"Output directory:  [cyan]{os.path.abspath(out_dir)}[/]\n"
            f"  ├── run_config.json\n"
            f"  ├── hwhm_table.csv\n"
            f"  ├── results.json\n"
            + ("  ├── posterior_samples.npz\n" if samples is not None else "")
            + f"  └── figures/",
            border_style="green", padding=(0, 4)
        ))
    else:
        print(f"\nDone in {total:.1f} s.  Output: {os.path.abspath(out_dir)}")


# CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qens_terminal.py",
        description="QENS analysis — ISIS HDF5 · Bayesian CE model · rich terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python qens_terminal.py --data_dir ./data
  python qens_terminal.py --data_dir ./data --files "benzene_290_360_inc.nxspe benzene_290_197_inc.nxspe"
  python qens_terminal.py --data_dir ./data --models ce fickian ss_model
  python qens_terminal.py --data_dir ./data --no_mcmc --q_min 0.5 --q_max 1.8
  python qens_terminal.py --cfg results/20250401_120000/run_config.json
  python qens_terminal.py --inspect /data/benzene_290_360_inc.nxspe
""",
    )

    p.add_argument(
        "--inspect", metavar="FILE",
        help="Print HDF5 tree of FILE and exit (diagnostic)",
    )
    p.add_argument(
        "--data_dir", default="data", metavar="DIR",
        help="Directory containing .nxspe files  [default: ./data]",
    )
    p.add_argument(
        "--files", nargs="+", metavar="FILE",
        help="Space-separated list of filenames to analyse  "
             "(if omitted, all .nxspe files in data_dir are used)",
    )
    p.add_argument(
        "--primary", metavar="FILE",
        help="Primary INC file for spectrum / preview plots  "
             "(auto-detected if omitted)",
    )
    p.add_argument(
        "--cfg", metavar="JSON",
        help="Reload a previously saved run_config.json",
    )
    p.add_argument(
        "--models", nargs="+",
        choices=["ce", "fickian", "ss_model"],
        default=["ce", "fickian"],
        help="Diffusion models to fit  [default: ce fickian]",
    )
    p.add_argument("--q_min",  type=float, default=0.30,  metavar="Å⁻¹")
    p.add_argument("--q_max",  type=float, default=2.50,  metavar="Å⁻¹")
    p.add_argument("--ewin",   type=float, default=0.80,  metavar="meV",
                   help="Energy window half-width  [default: 0.80 meV]")
    p.add_argument("--n_bins", type=int,   default=13,
                   help="Q bins for HWHM extraction  [default: 13]")
    p.add_argument("--q_target", type=float, default=1.06, metavar="Å⁻¹",
                   help="Q value for single-spectrum plot  [default: 1.06]")

    mc = p.add_argument_group("MCMC options")
    mc.add_argument("--no_mcmc", action="store_true",
                    help="Skip MAP + MCMC — fast LS-only run")
    mc.add_argument("--n_walkers", type=int, default=32)
    mc.add_argument("--n_warmup",  type=int, default=500)
    mc.add_argument("--n_keep",    type=int, default=2000)
    mc.add_argument("--thin",      type=int, default=5)
    mc.add_argument("--seed",      type=int, default=42,
                    help="Random seed  [default: 42]")

    p.add_argument("--save_dir", default="results", metavar="DIR",
                   help="Parent output directory  [default: results]")

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.inspect:
        inspect_nxspe(args.inspect)
        return

    if args.cfg:
        with open(args.cfg) as fh:
            cfg = json.load(fh)
        _ok(f"Config reloaded from {args.cfg}")
    else:
        cfg = dict(DEFAULT_CONFIG)

    # CLI overrides
    cfg["data_dir"] = args.data_dir

    if args.files:
        cfg["files_to_fit"] = args.files
    elif not cfg.get("files_to_fit"):
        # auto-discover all .nxspe in data_dir
        nxspe = sorted(Path(args.data_dir).glob("*.nxspe"))
        if not nxspe:
            _err(f"No .nxspe files found in {args.data_dir}")
            sys.exit(1)
        cfg["files_to_fit"] = [f.name for f in nxspe]
        _info(f"Auto-discovered {len(nxspe)} .nxspe file(s)")

    if args.primary:
        cfg["primary_file"] = args.primary

    cfg.update(dict(
        models     = args.models,
        q_min      = args.q_min,
        q_max      = args.q_max,
        ewin_hwhm  = args.ewin,
        ewin_mcmc  = args.ewin,
        n_bins     = args.n_bins,
        n_bins_mc  = max(4, args.n_bins - 3),
        q_target   = args.q_target,
        n_walkers  = args.n_walkers,
        n_warmup   = args.n_warmup,
        n_keep     = args.n_keep,
        thin       = args.thin,
        random_seed = args.seed,
        save_dir   = args.save_dir,
    ))

    # sanity
    if cfg["q_min"] >= cfg["q_max"]:
        _err("--q_min must be less than --q_max"); sys.exit(1)
    if cfg["n_walkers"] % 2 != 0:
        _err("--n_walkers must be even"); sys.exit(1)

    if _HAS_RICH:
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        t.add_column("k", style="dim")
        t.add_column("v", style="cyan")
        for k, v in [
            ("data_dir",    cfg["data_dir"]),
            ("files",       ", ".join(cfg["files_to_fit"])),
            ("models",      " + ".join(cfg["models"])),
            ("Q range",     f"{cfg['q_min']}–{cfg['q_max']} Å⁻¹"),
            ("energy win",  f"±{cfg['ewin_hwhm']} meV"),
            ("MCMC",        "disabled (--no_mcmc)" if args.no_mcmc
                            else f"{cfg['n_walkers']}×{cfg['n_warmup']+cfg['n_keep']} steps"),
        ]:
            t.add_row(k, str(v))
        console.print(t)

    try:
        run(cfg, no_mcmc=args.no_mcmc)
    except Exception as exc:
        _err(f"Pipeline failed: {exc}")
        if _HAS_RICH:
            console.print_exception()
        else:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
