"""
preprocessing.py — Elastic peak alignment and resolution assignment
====================================================================

Two steps must be completed before any fitting:

1. ``fit_elastic_peak``  — shift the energy axis so that ω = 0 coincides with
   the true elastic line, and measure the raw elastic peak width.

2. ``assign_resolution`` — decide which sigma_res value to use as the
   instrument resolution for each dataset.  Priority order:
       (a) frozen-sample reference (same sample, T ≤ 270 K, dynamics frozen)
       (b) coherent scattering reference (e.g. vanadium) at the same Ei
       (c) the warm sample's own sig_raw — with a warning, as this is inflated
           by quasi-elastic broadening.

Why resolution matters
----------------------
sigma_res enters the spectral model directly as the width of the Gaussian
that is convolved with the quasi-elastic Lorentzian in ``make_basis``.  If
sigma_res is too large (e.g. 229 µeV from a warm benzene elastic peak instead
of the true ~52 µeV instrument resolution), the Lorentzian must be narrower
to reproduce the observed peak width, giving a systematically wrong Γ and
therefore a wrong D.  The entire analysis propagates this error.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def fit_elastic_peak(d: dict) -> tuple[float, float]:
    """
    Fit a Gaussian to the elastic line and shift the energy axis to ω = 0.

    The function averages spectra from the lowest-Q detectors, fits a Gaussian
    plus flat background, extracts the peak centre e0 and sigma sig_raw, then
    shifts the energy axis so that the elastic peak sits exactly at ω = 0.

    Why use only the lowest-Q detectors?
    -------------------------------------
    At high Q the quasi-elastic component is already broad, pulling the fitted
    Gaussian centre away from the true elastic peak and inflating sig_raw.
    The lowest-Q detectors (bottom 1/7 of the good list) have the smallest
    quasi-elastic broadening, giving the cleanest Gaussian shape to fit.

    Modifies the dataset dict in place
    -----------------------------------
    d["e0"]      : float — fitted peak position offset (meV)
    d["sig_raw"] : float — fitted Gaussian sigma (meV); includes instrument
                   resolution and any QENS broadening from the sample.
    d["e"]       : ndarray — energy axis shifted so elastic peak is at ω = 0.

    Parameters
    ----------
    d : dict
        Dataset dictionary as returned by ``read_nxspe``.  Must contain
        keys: "e_raw", "data", "good", "name".

    Returns
    -------
    e0 : float
        Elastic peak position (meV) — the calibration offset.
    sig_raw : float
        Gaussian sigma of the elastic peak (meV).

    Notes
    -----
    If curve_fit fails (e.g. very noisy data), e0 falls back to the index of
    the maximum of the averaged spectrum and sig_raw to 0.043 meV (~100 µeV
    FWHM), which is a reasonable fallback for typical ISIS direct-geometry
    spectrometers.
    """
    good = d["good"]   # index array of usable detectors
    e    = d["e_raw"]  # raw energy axis (bin centres, meV)

    # Use only the lowest-Q detectors to avoid quasi-elastic contamination.
    # max(3, ...) ensures at least 3 detectors even for small datasets.
    n_low = max(3, len(good) // 7)
    avg = np.nanmean([d["data"][i] for i in good[:n_low]], axis=0)
    avg = np.where(np.isfinite(avg), avg, 0.0)  # replace NaN with 0

    def _gauss(x, a, mu, sigma, bg):
        """Gaussian + flat background model for elastic peak fitting."""
        return a * np.exp(-0.5 * ((x - mu) / sigma)**2) + bg

    # Initial guess: peak at the index of the maximum
    pk = np.argmax(avg)

    try:
        popt, _ = curve_fit(
            _gauss, e, avg,
            p0=[avg[pk], e[pk], 0.05, max(avg.min(), 0)],
            # Physical bounds: amplitude ≥ 0, sigma ∈ (0.0001, 2) meV
            bounds=([0, e[0], 1e-4, -np.inf], [np.inf, e[-1], 2.0, np.inf]),
            maxfev=8000
        )
        e0      = float(popt[1])   # fitted peak centre
        sig_raw = abs(float(popt[2]))  # fitted sigma
    except Exception as exc:
        # Fallback: use the raw maximum position and a typical ISIS resolution
        print(f"  elastic peak fit failed for {d['name']}: {exc}")
        e0      = float(e[pk])
        sig_raw = 0.043  # ~100 µeV FWHM — reasonable fallback

    # Store results in the dataset dict and shift the energy axis
    d["e0"]      = e0
    d["sig_raw"] = sig_raw
    d["e"]       = e - e0  # elastic line now sits at ω = 0

    return e0, sig_raw


def assign_resolution(dataset: dict, res_key: str | None = None) -> None:
    """
    Assign instrument resolution sigma to every dataset in place.

    Determines the best available estimate of sigma_res for each dataset and
    writes it (along with fwhm_res and res_source) into each dataset dict.

    Resolution source priority
    --------------------------
    1. Frozen-sample file (res_key or auto-detected):
       Same sample measured at T ≤ 270 K so molecular dynamics are frozen.
       The elastic peak then reflects only instrument broadening.
       Matched by incident energy Ei.

    2. Coherent scattering file (e.g. vanadium):
       Detected by d["kind"] == "coh".  Matched by Ei.

    3. Raw sig_raw from the warm sample (fallback, with warning):
       The elastic peak of a liquid sample at 290 K is quasi-elastically
       broadened.  For benzene at Ei = 3.6 meV this gives ~229 µeV instead
       of the true ~52 µeV instrument resolution.  All fits using this value
       will return a wrong Γ and therefore a wrong D.

    Modifies each dataset dict in place
    ------------------------------------
    d["sigma_res"] : float — instrument resolution sigma (meV)
    d["fwhm_res"]  : float — 2.355 × sigma_res (meV) — the FWHM
    d["res_source"] : str  — one of:
        "frozen sample" | "COH file" | "raw INC (may be inflated)"

    Parameters
    ----------
    dataset : dict
        Dict of dataset dicts as returned by ``load_dataset``.
    res_key : str or None
        If provided, the key of a specific file in dataset to use as the
        frozen resolution reference.  If None, the function auto-detects any
        file with kind="inc" and temp ≤ 270 K.

    Notes
    -----
    The 2.355 factor converts sigma to FWHM: FWHM = 2√(2 ln 2) · σ ≈ 2.355σ.
    This is the standard conversion for a Gaussian.
    """
    # ── Step 1: build frozen-reference lookup  {Ei: sigma_res} ───────────────
    res_frozen: dict[float, float] = {}

    if res_key is not None:
        # User explicitly nominated a reference file
        if res_key in dataset:
            d_ref = dataset[res_key]
            res_frozen[d_ref["ei"]] = d_ref["sig_raw"]
            print(
                f"  frozen reference: {res_key}  "
                f"Ei={d_ref['ei']:.2f} meV  "
                f"FWHM={d_ref['sig_raw']*2355:.1f} µeV"
            )
        else:
            print(f"  WARNING: res_key '{res_key}' not found in dataset")
    else:
        # Auto-detect frozen files: same sample kind "inc" at low temperature
        frozen_files = [
            d for d in dataset.values()
            if d["kind"] == "inc" and d["temp"] <= 270
        ]
        for d_ref in frozen_files:
            res_frozen[d_ref["ei"]] = d_ref["sig_raw"]
            print(
                f"  frozen reference (auto): {d_ref['name']}  "
                f"T={d_ref['temp']} K  "
                f"FWHM={d_ref['sig_raw']*2355:.1f} µeV"
            )

    # ── Step 2: build coherent-reference lookup  {Ei: sigma_res} ─────────────
    # Coherent files (vanadium etc.) provide an alternative resolution estimate
    coh_sigma: dict[float, float] = {
        d["ei"]: d["sig_raw"]
        for d in dataset.values()
        if d["kind"] == "coh"
    }

    # ── Step 3: assign sigma_res to every file ────────────────────────────────
    for fname, d in dataset.items():
        ei = d["ei"]

        if ei in res_frozen:
            # Best case: frozen-sample measurement at same Ei
            d["sigma_res"]  = res_frozen[ei]
            d["res_source"] = "frozen sample"

        elif ei in coh_sigma:
            # Second best: coherent reference at same Ei
            d["sigma_res"]  = coh_sigma[ei]
            d["res_source"] = "COH file"

        else:
            # Fallback: use the warm sample's own elastic width — inflated!
            d["sigma_res"]  = d["sig_raw"]
            d["res_source"] = "raw INC (may be inflated)"
            print(
                f"  WARNING: no resolution reference for {fname} "
                f"(Ei={ei:.2f}) — using raw sig_raw which may be "
                f"broadened by quasi-elastic scattering"
            )

        # FWHM = 2√(2 ln 2) · sigma ≈ 2.355 · sigma
        d["fwhm_res"] = 2.355 * d["sigma_res"]

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n  resolution summary:")
    for fname, d in dataset.items():
        print(
            f"    {fname:<44}  "
            f"{d['fwhm_res']*1000:6.1f} µeV  "
            f"[{d['res_source']}]"
        )
