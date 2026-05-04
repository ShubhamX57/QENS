"""
Reader for ISIS Mantid '.nxspe' (NeXus / HDF5) files.


Filename convention (lowercase, underscore-separated)
-----------------------------------------------------
    <sample>_<temperature_K>_<Ei_x_100>_<kind>.nxspe

    
Example: "benzene_290_360_inc.nxspe" >>> benzene, 290 K, Ei = 3.60 meV,
incoherent measurement.



"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from .constants import NEUTRON_MASS_KG, HBAR_JS, MEV_TO_J



__all__ = ["inspect_nxspe",
           "read_nxspe",
           "read_nxspe_with_overrides",
           "load_dataset",
           "compute_q_from_2theta"]




# kinematics
def compute_q_from_2theta(two_theta_deg: np.ndarray, ei_mev: float) -> np.ndarray:
    """
    Q(2θ) at the elastic line from the incident wavevector.

    Parameters
    ----------
    two_theta_deg : array
        Scattering angle in degrees.

    ei_mev : float
        Incident energy in meV.

    Returns
    -------
    Q in Å⁻¹.

    """
    ki = np.sqrt(2 * NEUTRON_MASS_KG * ei_mev * MEV_TO_J) / HBAR_JS * 1e-10
    return 2 * ki * np.sin(np.radians(two_theta_deg / 2))



# inspection helper
def inspect_nxspe(path: str) -> None:
    """
    Print the HDF5 tree of a .nxspe file (debugging tool).
    
    """

    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py required for .nxspe IO. pip install h5py") from e


    print(f"HDF5 tree: {path}")
    print("-" * 60)


    def _visitor(name, obj):
        indent = "  " * name.count("/")
        leaf = name.split("/")[-1]
        if hasattr(obj, "shape"):
            print(f"{indent}{leaf}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            print(f"{indent}{leaf}/")


    with h5py.File(path, "r") as hf:
        hf.visititems(_visitor)
    print("-" * 60)



# core reader
_NXSPE_POLAR_PATHS = (("data", "polar"),
                      ("instrument/detector", "polar"),
                      ("instrument/detector_1", "polar_angle"))




def _read_ei(entry, parts: list[str]) -> float:
    """
    Read fixed incident energy from NXSPE_info, or fall back to filename.
    
    """
    for key in ("efixed", "fixed_energy"):
        try:
            return float(entry["NXSPE_info"][key][()])
        except (KeyError, TypeError):
            pass
    try:
        return int(parts[2]) / 100.0
    except (ValueError, IndexError) as e:
        raise ValueError(
            "Cannot determine incident energy: not in NXSPE_info and "
            "filename does not contain Ei x 100 in the third underscore field"
        ) from e




def _read_polar(entry, fname: str) -> np.ndarray:
    """
    Locate the polar-angle array (per-detector 2θ) in the HDF5 tree.
    
    """
    for grp_path, ds_name in _NXSPE_POLAR_PATHS:
        try:
            grp = entry
            for part in grp_path.split("/"):
                grp = grp[part]
            arr = np.asarray(grp[ds_name], dtype=float)
            if arr.ndim == 1 and arr.size > 0:
                return arr
        except (KeyError, TypeError):
            pass
    # last resort
    try:
        arr = np.asarray(entry["data"]["azimuthal"], dtype=float)
        if arr.ndim == 1 and arr.size > 0:
            import warnings
            warnings.warn(f"Using azimuthal as proxy for polar in {fname}",
                          UserWarning, stacklevel=3)
            return arr
    except (KeyError, TypeError):
        pass
    raise ValueError(f"Cannot locate polar angles in '{fname}'")




def _parse_filename(name: str) -> tuple[str, int, str]:
    """
    Parse <sample>_<temp>_<Eix100>_<kind>.nxspe  >> (sample, temp, kind).

    """

    base = name.replace(".nxspe", "")
    parts = base.split("_")
    if len(parts) < 4:
        raise ValueError(f"Filename '{name}' does not match "
                         "<sample>_<temperature>_<Eix100>_<kind>")

    sample = parts[0]
    try:
        temp = int(parts[1])
    except ValueError as e:
        raise ValueError(f"Cannot parse temperature from '{name}'") from e
    kind = parts[3]
    return sample, temp, kind, parts





def read_nxspe(path: str) -> dict:
    """
    
    Load a single '.nxspe' file.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    dict with keys:
        "name" : filename
        "sample", "temp", "ei", "kind" : metadata
        "e_raw" : energy axis (meV) **before** elastic-peak alignment
        "e"     : energy axis (meV) — initially copy of e_raw, gets shifted
                    by :func:'qens.preprocessing.fit_elastic_peak'
        "data", "errs" : (n_det, n_E) arrays of intensity and 1σ errors
        "q" : (n_det,) momentum transfer at elastic line, Å⁻¹
        "good" : (n_good,) integer indices of detectors with usable data

    """


    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py required: pip install h5py") from e



    name = os.path.basename(path)
    sample, temp, kind, parts = _parse_filename(name)



    with h5py.File(path, "r") as hf:
        entry_key = next(iter(hf.keys()))
        entry = hf[entry_key]
        ei = _read_ei(entry, parts)
        energy_edges = np.asarray(entry["data"]["energy"], dtype=float)
        e_raw = 0.5 * (energy_edges[:-1] + energy_edges[1:])
        data = np.asarray(entry["data"]["data"], dtype=float)
        errs = np.asarray(entry["data"]["error"], dtype=float)
        two_theta = _read_polar(entry, name)



    data = np.where(np.isfinite(data), data, 0.0)
    errs = np.where(np.isfinite(errs) & (errs > 0), errs, 0.0)



    good_mask = np.sum((data > 0) & np.isfinite(data), axis=1) > data.shape[1] // 2
    good = np.where(good_mask)[0]
    if good.size == 0:
        raise ValueError(f"No usable detectors in '{name}'")



    q = compute_q_from_2theta(two_theta, ei)

    return dict(name=name, sample=sample, temp=temp, ei=ei, kind=kind,
                e_raw=e_raw, e=e_raw.copy(),
                data=data, errs=errs, good=good, q=q,
                format="hdf5")




def read_nxspe_with_overrides(
    path: str,
    *,
    sample: str | None = None,
    temp: int | None = None,
    ei: float | None = None,
    kind: str | None = None,
) -> dict:
    """
    
    Like :func:'read_nxspe' but lets you override metadata.

    Useful when filenames don't follow the convention.

    """
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py required") from e
    name = os.path.basename(path)


    with h5py.File(path, "r") as hf:
        entry_key = next(iter(hf.keys()))
        entry = hf[entry_key]

        if ei is None:
            for key in ("efixed", "fixed_energy"):
                try:
                    ei = float(entry["NXSPE_info"][key][()])
                    break
                except (KeyError, TypeError):
                    pass
            if ei is None:
                raise ValueError("Could not determine Ei; pass ei=...")


        energy_edges = np.asarray(entry["data"]["energy"], dtype=float)
        e_raw = 0.5 * (energy_edges[:-1] + energy_edges[1:])
        data = np.asarray(entry["data"]["data"], dtype=float)
        errs = np.asarray(entry["data"]["error"], dtype=float)
        two_theta = _read_polar(entry, name)




    data = np.where(np.isfinite(data), data, 0.0)
    errs = np.where(np.isfinite(errs) & (errs > 0), errs, 0.0)
    good_mask = np.sum((data > 0) & np.isfinite(data), axis=1) > data.shape[1] // 2
    good = np.where(good_mask)[0]


    if good.size == 0:
        raise ValueError(f"No usable detectors in '{name}'")


    return dict(name=name, sample=sample or "?", temp=temp if temp is not None else -1,
                ei=ei, kind=kind or "inc",
                e_raw=e_raw, e=e_raw.copy(),
                data=data, errs=errs, good=good,
                q=compute_q_from_2theta(two_theta, ei),
                format="hdf5")





# multi file loader

def load_dataset(file_list: Iterable[str],
                 data_dir: str = ".",
                 critical_files: Iterable[str] | None = None,
                 verbose: bool = True,
) -> dict[str, dict]:
    """
    Load several .nxspe files into a dict keyed by filename.

    Parameters
    ----------
    file_list : iterable of str
        Filenames to load (relative to "data_dir").
    data_dir : str
        Directory containing the files.
    critical_files : iterable of str, optional
        Files that *must* load — raises if any are missing/unreadable.
        Other failures are warnings.
    verbose : bool
        If True, print a one-line summary per file.

    Returns
    -------
    dict[str, dict]
        Keys are filenames, values are the dicts returned by :func:'read_nxspe'.


    """


    critical = set(critical_files or [])
    
    out: dict[str, dict] = {}
    for fname in file_list:
        full = os.path.join(data_dir, fname)
        if not os.path.exists(full):
            msg = f"  skipping (not found): {fname}"
            if fname in critical:
                raise FileNotFoundError(f"Critical file missing: {fname}")
            if verbose:
                print(msg)
            continue
        try:
            d = read_nxspe(full)
            out[fname] = d
            if verbose:
                print(f"  loaded: {fname:<36}  Ei={d['ei']:.2f} meV  "
                      f"T={d['temp']} K  good={len(d['good'])} det")
        except Exception as exc:
            if fname in critical:
                raise
            if verbose:
                print(f"  failed: {fname}: {exc}")
    if not out:
        raise RuntimeError("No files loaded successfully")
    return out
