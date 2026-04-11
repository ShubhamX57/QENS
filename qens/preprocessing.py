"""
Elastic peak alignment and resolution assignment.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit


def fit_elastic_peak(d: dict):
    """
    Fit Gaussian to elastic line and shift e axis to ω=0.
    """
    good = d["good"]
    e    = d["e_raw"]


    n_low = max(3, len(good) // 7)
    avg = np.nanmean([d["data"][i] for i in good[:n_low]], axis=0)
    avg = np.where(np.isfinite(avg), avg, 0.0)


    def _gauss(x, a, mu, sigma, bg):
        return a * np.exp(-0.5 * ((x - mu) / sigma)**2) + bg


    pk = np.argmax(avg)
    try:
        popt, _ = curve_fit(_gauss, e, avg,
                            p0=[avg[pk], e[pk], 0.05, max(avg.min(), 0)],
                            bounds=([0, e[0], 1e-4, -np.inf], [np.inf, e[-1], 2.0, np.inf]),
                            maxfev=8000)
        e0 = float(popt[1])
        sig_raw = abs(float(popt[2]))
    except Exception as exc:
        print(f"  elastic peak fit failed for {d['name']}: {exc}")
        e0 = float(e[pk])
        sig_raw = 0.043

    d["e0"] = e0
    d["sig_raw"] = sig_raw
    d["e"] = e - e0
    return e0, sig_raw




def assign_resolution(dataset: dict, res_key: str | None = None):
    """
    Assign instrument sigma to each file. Modifies dicts in place.
    """
    res_frozen = {}
    if res_key is not None:
        if res_key in dataset:
            d_ref = dataset[res_key]
            res_frozen[d_ref["ei"]] = d_ref["sig_raw"]
            print(f"  frozen reference: {res_key} Ei={d_ref['ei']:.2f} meV FWHM={d_ref['sig_raw']*2355:.1f} µeV")
        else:
            print(f"  WARNING: res_key '{res_key}' not found")
    else:
        frozen_files = [d for d in dataset.values() if d["kind"] == "inc" and d["temp"] <= 270]
        for d_ref in frozen_files:
            res_frozen[d_ref["ei"]] = d_ref["sig_raw"]
            print(f"  frozen reference (auto): {d_ref['name']} T={d_ref['temp']} K FWHM={d_ref['sig_raw']*2355:.1f} µeV")

    coh_sigma = {d["ei"]: d["sig_raw"] for d in dataset.values() if d["kind"] == "coh"}

    for fname, d in dataset.items():
        ei = d["ei"]
        if ei in res_frozen:
            d["sigma_res"] = res_frozen[ei]
            d["res_source"] = "frozen sample"
        elif ei in coh_sigma:
            d["sigma_res"] = coh_sigma[ei]
            d["res_source"] = "COH file"
        else:
            d["sigma_res"] = d["sig_raw"]
            d["res_source"] = "raw INC (may be inflated)"
            print(f"  WARNING: no resolution reference for {fname} (Ei={ei:.2f})")
        d["fwhm_res"] = 2.355 * d["sigma_res"]

    print("\n  resolution summary:")
    for fname, d in dataset.items():
        print(f"    {fname:<44}  {d['fwhm_res']*1000:6.1f} µeV  [{d['res_source']}]")
