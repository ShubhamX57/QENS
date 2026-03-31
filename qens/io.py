"""
io.py

Everything to do with loading data from disk.
The Pelican .nxspe format is a fairly simple binary file — the data
is just raw arrays at fixed byte offsets. We verify the offsets look
sane before reading, which catches the most common failure mode of
a corrupt or truncated file.

The main things you'd call from outside are:
    read_nxspe(path)                   — load a single file
    load_dataset(files, data_dir, ...)  — load a bunch at once
"""

import os
import numpy as np

from .constants import (
    n_det, n_bin, two_theta,
    mn, hbar, mev_j,
    off_e_default, off_d_default, off_err_default,
)


def _find_offsets(raw):
    """
    Check that the default byte offsets give us a sensible energy value.
    If they do, return them. If not, raise a ValueError.

    This is admittedly a bit of a hack — we just poke the file at the
    expected location and see if the number we get looks like it could
    be an energy in meV. If someone sends a weird non-Pelican file,
    this is the first thing that'll catch it.
    """
    try:
        test = np.frombuffer(
            raw[off_e_default : off_e_default + 8], dtype="<f8"
        )
        # a valid energy value should be finite and in roughly meV range
        if np.isfinite(test[0]) and -20 < test[0] < 20:
            return off_e_default, off_d_default, off_err_default
    except Exception:
        pass

    raise ValueError(
        "Can't locate data offsets in this file — "
        "it might be corrupt, truncated, or not a Pelican .nxspe file."
    )


def read_nxspe(path):
    """
    Read a single Pelican .nxspe binary file and return everything
    you need for analysis in a plain dict.

    The filename is expected to follow the convention:
        <sample>_<temperature>_<Ei*100>_<kind>.nxspe
    e.g. benzene_290_360_inc.nxspe → 290 K, Ei=3.60 meV, incoherent

    Returns a dict with keys:
        name      — basename of the file
        temp      — temperature in K (parsed from filename)
        ei        — incident energy in meV
        kind      — "inc" or "coh"
        e_raw     — energy bin centres before elastic peak correction (meV)
        e         — corrected energy axis (set by fit_elastic_peak later)
        data      — S(Q,w) array, shape (n_det, n_bin)
        errs      — uncertainties, same shape
        good      — indices of detectors that passed basic sanity checks
        q         — momentum transfer for each detector (Å⁻¹)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    name   = os.path.basename(path)
    parts  = name.replace(".nxspe", "").split("_")

    if len(parts) < 4:
        raise ValueError(
            f"Filename '{name}' doesn't match the expected pattern "
            f"<sample>_<temp>_<Ei*100>_<kind>.nxspe"
        )

    temp = int(parts[1])
    ei   = int(parts[2]) / 100.0
    kind = parts[3]

    with open(path, "rb") as fh:
        raw = fh.read()

    off_e, off_d, off_err = _find_offsets(raw)

    # read the energy bin edges and compute bin centres
    edges = np.frombuffer(raw[off_e : off_e + (n_bin + 1) * 8], dtype="<f8")
    e_raw = 0.5 * (edges[:-1] + edges[1:])

    # read the data and error arrays
    n_vals = n_det * n_bin
    data = (
        np.frombuffer(raw[off_d   : off_d   + n_vals * 8], dtype="<f8")
        .reshape(n_det, n_bin)
        .copy()
    )
    errs = (
        np.frombuffer(raw[off_err : off_err + n_vals * 8], dtype="<f8")
        .reshape(n_det, n_bin)
        .copy()
    )

    # a "good" detector has finite positive counts in at least half its bins.
    # this filters out broken detectors, masked regions, etc.
    good_mask = np.array([
        np.sum(np.isfinite(data[i]) & (data[i] > 0)) > n_bin // 2
        for i in range(n_det)
    ])

    # compute Q for each detector group
    ki = np.sqrt(2 * mn * ei * mev_j) / hbar * 1e-10
    q  = 2 * ki * np.sin(np.radians(two_theta / 2))

    return dict(
        name  = name,
        temp  = temp,
        ei    = ei,
        kind  = kind,
        e_raw = e_raw,
        e     = e_raw.copy(),   # will be updated by fit_elastic_peak
        data  = data,
        errs  = errs,
        good  = np.where(good_mask)[0],
        q     = q,
    )


def load_dataset(file_list, data_dir=".", critical_files=None):
    """
    Load a list of .nxspe files from data_dir. Files that are missing or
    fail to load are skipped with a warning, unless they appear in
    critical_files — those raise an error immediately.

    Returns a dict mapping filename → data dict.

    Example
    -------
        ds = load_dataset(
            ["benzene_290_360_inc.nxspe", "benzene_290_360_coh.nxspe"],
            data_dir="/data/run42",
            critical_files=["benzene_290_360_inc.nxspe"],
        )
    """
    if critical_files is None:
        critical_files = []

    dataset = {}

    for fname in file_list:
        full_path = os.path.join(data_dir, fname)

        if not os.path.exists(full_path):
            if fname in critical_files:
                raise FileNotFoundError(
                    f"Critical file not found: {fname}\n"
                    f"Expected at: {full_path}"
                )
            print(f"  skipping (not found): {fname}")
            continue

        try:
            d = read_nxspe(full_path)
            dataset[fname] = d
            print(f"  loaded: {fname}  (Ei={d['ei']:.2f} meV, T={d['temp']} K)")
        except Exception as exc:
            if fname in critical_files:
                raise
            print(f"  failed to load {fname}: {exc}")

    return dataset
