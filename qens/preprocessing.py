"""
Preprocessing: align the elastic peak to ω = 0 and assign each measurement a
resolution function.

elastic-peak alignment, resolution assignment

"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit

from .config import Config
from .constants import GAUSSIAN_FWHM_FACTOR

__all__ = ["fit_elastic_peak", "assign_resolution"]





# elastic peak fit

def _gauss(x, a, mu, sigma, bg):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + bg


def fit_elastic_peak(d: dict) -> tuple[float, float]:
    """Fit a Gaussian to the elastic peak of a low-Q detector average.

    Updates ``d["e"]`` to be ``d["e_raw"] - e0``, and stores
    ``d["e0"]``, ``d["sig_raw"]`` (the elastic-peak Gaussian sigma).

    Parameters
    ----------
    d : dict
        Output of :func:`qens.io.read_nxspe`.

        

    Returns
    -------
    (e0, sig_raw) : (float, float)
        Fitted peak centre and width in meV.


    """
    good = d["good"]
    e = d["e_raw"]


    # average the lowest-Q ~14% of detectors (most elastic-dominated)
    n_low = max(3, len(good) // 7)
    avg = np.nanmean([d["data"][i] for i in good[:n_low]], axis=0)
    avg = np.where(np.isfinite(avg), avg, 0.0)


    pk = int(np.argmax(avg))
    try:
        popt, _ = curve_fit(
            _gauss, e, avg,
            p0=[avg[pk], e[pk], 0.05, max(avg.min(), 0.0)],
            bounds=([0, e[0], 1e-4, -np.inf], [np.inf, e[-1], 2.0, np.inf]),
            maxfev=8000,
        )
        e0 = float(popt[1])
        sig = abs(float(popt[2]))
    except Exception as exc:
        warnings.warn(
            f"Elastic-peak fit failed for {d.get('name','?')}: {exc}; "
            f"falling back to argmax",
            RuntimeWarning, stacklevel=2,
        )
        e0 = float(e[pk])
        sig = 0.043  # sensible default ~ IRIS resolution


    d["e0"] = e0
    d["sig_raw"] = sig
    d["e"] = e - e0
    return e0, sig




# resolution assignment

def assign_resolution(
    dataset: dict[str, dict],
    cfg: Config | None = None,
    verbose: bool = True,
) -> None:
    """
    
    Decide which file is the resolution function for each loaded file.

    Modifies each dict in place, adding:

        ``sigma_res``   : Gaussian sigma in meV (Gaussian-fit fallback)
        ``fwhm_res``    : 2.355 x sigma_res, in meV
        ``res_source``  : one of "frozen_sample", "coh_file",
                          "raw_inc_inflated", "explicit_override"
        ``res_file``    : filename of the resolution reference, or None
                          if the file's own peak fit is used

                          
    Parameters
    ----------
    dataset : dict[str, dict]
        Output of :func:`qens.io.load_dataset`. Each entry must already have
        had :func:`fit_elastic_peak` called on it.
    cfg : Config, optional
        If given, ``cfg.resolution_file`` and ``cfg.frozen_temp_threshold``
        are honoured.
    verbose : bool
        If True, print a per-file summary.


    """



    if cfg is None:
        cfg = Config()


    # Build the lookup tables
    frozen_by_ei: dict[float, dict] = {}
    coh_by_ei: dict[float, dict] = {}


    if cfg.resolution_file and cfg.resolution_file in dataset:
        ref = dataset[cfg.resolution_file]
        frozen_by_ei[ref["ei"]] = ref
        if verbose:
            print(f"  using explicit resolution file: {cfg.resolution_file}  "
                  f"Ei={ref['ei']:.2f} meV  "
                  f"FWHM={ref['sig_raw']*GAUSSIAN_FWHM_FACTOR*1000:.1f} µeV")
    elif cfg.resolution_file:
        warnings.warn(f"resolution_file '{cfg.resolution_file}' not found in "
                      f"loaded dataset; falling back to auto-pick",
                      RuntimeWarning, stacklevel=2)


    # auto-pick: every frozen INC file is a resolution ref for its E_i, does this by ensuring inc spectrum is below threshold temp
    for fname, d in dataset.items():
        if (d["kind"] == "inc"
                and d["temp"] <= cfg.frozen_temp_threshold
                and d["ei"] not in frozen_by_ei):
            frozen_by_ei[d["ei"]] = d


    # also note coherent files as backup
    for fname, d in dataset.items():
        if d["kind"] == "coh" and d["ei"] not in coh_by_ei:
            coh_by_ei[d["ei"]] = d


    # Now assign each file its resolution
    for fname, d in dataset.items():
        ei = d["ei"]
        if ei in frozen_by_ei:
            ref = frozen_by_ei[ei]
            d["sigma_res"] = ref["sig_raw"]
            d["res_source"] = ("explicit_override"
                               if fname == cfg.resolution_file
                               else "frozen_sample")
            d["res_file"] = ref["name"]
        elif ei in coh_by_ei:
            ref = coh_by_ei[ei]
            d["sigma_res"] = ref["sig_raw"]
            d["res_source"] = "coh_file"
            d["res_file"] = ref["name"]
        else:
            d["sigma_res"] = d["sig_raw"]
            d["res_source"] = "raw_inc_inflated"
            d["res_file"] = None
            warnings.warn(
                f"No resolution reference for {fname} (Eᵢ={ei:.2f} meV). "
                f"Using its own elastic-peak fit — sigma may be inflated by "
                f"QENS broadening.",
                RuntimeWarning, stacklevel=2,
            )
        d["fwhm_res"] = GAUSSIAN_FWHM_FACTOR * d["sigma_res"]


    if verbose:
        print("\n  resolution assignment:")
        for fname, d in dataset.items():
            ref = d.get("res_file") or "(self)"
            print(f"    {fname:<36}  FWHM={d['fwhm_res']*1000:6.1f} µeV  "
                  f"[{d['res_source']:<18}]  ref={ref}")
