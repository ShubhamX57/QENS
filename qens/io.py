"""
Read ISIS Mantid .nxspe HDF5 files.
"""

from __future__ import annotations
import os
import numpy as np
from .constants import mn, hbar, mev_j


def inspect_nxspe(path: str) -> None:
    """
    Print the HDF5 tree of a .nxspe file.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required: pip install h5py")

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




def read_nxspe(path: str) -> dict:
    """
    Load a single .nxspe file. Returns dict with e, data, errs, q, etc.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required")

    name  = os.path.basename(path)
    parts = name.replace(".nxspe", "").split("_")
    if len(parts) < 4:
        raise ValueError(f"Filename '{name}' does not match <sample>_<temp>_<Ei×100>_<kind>")
    try:
        temp = int(parts[1])
    except ValueError:
        raise ValueError(f"Cannot parse temperature from '{name}'")
    kind = parts[3]

    with h5py.File(path, "r") as hf:
        entry_key = next(iter(hf.keys()))
        entry = hf[entry_key]
        ei = _read_ei(entry, parts)
        energy_edges = np.asarray(entry["data"]["energy"], dtype=float)
        e_raw = 0.5 * (energy_edges[:-1] + energy_edges[1:])
        data = np.asarray(entry["data"]["data"], dtype=float)
        errs = np.asarray(entry["data"]["error"], dtype=float)
        two_theta_det = _read_polar(entry, path)

    data = np.where(np.isfinite(data), data, 0.0)
    errs = np.where(np.isfinite(errs) & (errs > 0), errs, 0.0)

    good_mask = np.sum((data > 0) & np.isfinite(data), axis=1) > data.shape[1] // 2
    good = np.where(good_mask)[0]
    if good.size == 0:
        raise ValueError(f"No usable detectors in '{name}'")

    ki = np.sqrt(2 * mn * ei * mev_j) / hbar * 1e-10
    q  = 2 * ki * np.sin(np.radians(two_theta_det / 2))

    return dict(name=name, temp=temp, ei=ei, kind=kind, e_raw=e_raw, e=e_raw.copy(),
                data=data, errs=errs, good=good, q=q, format="hdf5")





read_nxspe_hdf5 = read_nxspe

def _read_ei(entry, filename_parts):
    for key in ("efixed", "fixed_energy"):
        try:
            return float(entry["NXSPE_info"][key][()])
        except (KeyError, TypeError):
            pass
    try:
        return int(filename_parts[2]) / 100.0
    except (ValueError, IndexError):
        raise ValueError("Cannot determine Ei")





def _read_polar(entry, path):
    candidates = [("data", "polar"),
                  ("instrument/detector", "polar"),
                  ("instrument/detector_1", "polar_angle"),]
    
    for grp_path, ds_name in candidates:
        try:
            grp = entry
            for part in grp_path.split("/"):
                grp = grp[part]
            arr = np.asarray(grp[ds_name], dtype=float)
            if arr.ndim == 1 and arr.size > 0:
                return arr
        except (KeyError, TypeError):
            pass
    # last resort: azimuthal
    try:
        arr = np.asarray(entry["data"]["azimuthal"], dtype=float)
        if arr.ndim == 1 and arr.size > 0:
            import warnings
            warnings.warn("Using azimuthal as proxy for polar", UserWarning)
            return arr
    except (KeyError, TypeError):
        pass
    raise ValueError(f"Cannot locate polar angles in '{os.path.basename(path)}'")




def load_dataset(file_list: list[str], data_dir: str = ".", critical_files: list[str] | None = None):
    if critical_files is None:
        critical_files = []
    dataset = {}
    for fname in file_list:
        full_path = os.path.join(data_dir, fname)
        if not os.path.exists(full_path):
            if fname in critical_files:
                raise FileNotFoundError(f"Critical file missing: {fname}")
            print(f"  skipping (not found): {fname}")
            continue
        try:
            d = read_nxspe(full_path)
            dataset[fname] = d
            print(f"  loaded [hdf5]: {fname}  Ei={d['ei']:.2f} meV  T={d['temp']} K  good={len(d['good'])} det")
        except Exception as exc:
            if fname in critical_files:
                raise
            print(f"  failed to load {fname}: {exc}")
    if not dataset:
        raise RuntimeError("No files loaded successfully")
    return dataset
