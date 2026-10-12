"""Reading LET (ISIS) .nxspe files that hold S(|Q|, ω).

These reductions store 249 constant-|Q| rows but not the Q axis itself.  The grid,
recovered from the kinematic coverage of the data, is

    Q_row(n) = k_i (EDGE0 + DQ (n + ½)),    k_i = sqrt(E_i / 2.0721)

It matches 99.4–99.99% of populated cells and puts the aluminium (111) Bragg peak
of the sample environment at 2.687 Å⁻¹, so it is checked against the data rather
than assumed (see ``tests/test_let.py``).  If the reduction changes, re-check it.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field

import h5py
import numpy as np

from .constants import C_E2K

EDGE0, DQ = 0.0399, 0.00873  # Q grid of these reductions, in units of k_i
BRAGG = {"Al(111)": 2.687, "Cu(111)": 3.010, "Al(200)": 3.103}  # sample environment, Å⁻¹

# Resolution prior.  Elastic FWHM from PyChop (LET, high-flux mode, 200 Hz) gives the
# ratio between energies; the absolute scale comes from the Gaussian sd fitted at
# 3.60 meV on the low-Q frozen-sample data.  The fit treats it as a prior, not a fact.
PYCHOP_FWHM = {1.24: 19.5e-3, 1.97: 39.1e-3, 3.60: 89.3e-3, 8.61: 304.6e-3}  # meV
SIGMA_360 = 41.7e-3  # meV


def ki_of(ei):
    """Incident wavevector (Å⁻¹) for incident energy ``ei`` (meV)."""
    return np.sqrt(ei / C_E2K)


def resolution_sigma(ei):
    """Prior centre for the Gaussian resolution sd (meV) at incident energy ``ei``."""
    k = min(PYCHOP_FWHM, key=lambda e: abs(e - ei))
    if abs(k - ei) > 0.05:
        raise ValueError(f"no LET resolution width tabulated near Ei = {ei} meV")
    return SIGMA_360 * PYCHOP_FWHM[k] / PYCHOP_FWHM[3.60]


def read_raw(path):
    """The raw arrays of one .nxspe file, with the recovered constant-Q axis."""
    with h5py.File(path, "r") as h:
        e = h[list(h.keys())[0]]
        data = e["data/data"][()]
        error = e["data/error"][()]
        edges = e["data/energy"][()]
        ei = float(e["NXSPE_info/fixed_energy"][0])
    w = 0.5 * (edges[1:] + edges[:-1])
    q = ki_of(ei) * (EDGE0 + DQ * (np.arange(data.shape[0]) + 0.5))
    return {"D": data, "Er": error, "w": w, "dw": np.diff(edges), "ei": ei, "q_rows": q, "name": os.path.basename(path)}


@dataclass
class Dataset:
    """One incident energy, binned in Q, ready to fit.  One entry per (Q, ω) point."""

    name: str
    ei: float
    omega: np.ndarray  # bin centre, meV (energy loss positive)
    dw: np.ndarray  # bin width, meV
    q: np.ndarray  # Q of each point (mean of contributing rows), Å⁻¹
    y: np.ndarray
    err: np.ndarray
    bin_id: np.ndarray  # Q-bin index of each point
    q_bins: np.ndarray  # mean Q of each bin
    n_bins: int
    meta: dict = field(default_factory=dict)


def noise_ratio(ds, sigma_res, wing=6.0):
    """Do the error bars describe the scatter?

    In the wings (|ω| > ``wing`` × resolution) the true spectrum is smooth, so the
    second difference of neighbouring points is noise.  Returns the mean of
    (second difference)² / predicted variance (1.0 = error bars are right) and the
    number of triples used.  Neighbouring Q rows share counts, so after rows are
    combined this is usually above 1.
    """
    vals = []
    for b in range(ds.n_bins):
        m = np.where(ds.bin_id == b)[0]
        y, e, w = ds.y[m], ds.err[m], ds.omega[m]
        for i in range(1, len(y) - 1):
            if abs(w[i]) > wing * sigma_res and abs((w[i + 1] - w[i]) - (w[i] - w[i - 1])) < 1e-9:
                dd = y[i - 1] - 2 * y[i] + y[i + 1]
                vals.append(dd * dd / (e[i - 1] ** 2 + 4 * e[i] ** 2 + e[i + 1] ** 2))
    return float(np.mean(vals)), len(vals)


def prepare(path, qmin=0.3, qmax=2.6, wmax=1.25, dq_target=0.10, min_row_frac=0.75, bragg_halfwidth=0.06):
    """Read one file and bin it for fitting.

    Keeps |ω| ≤ ``wmax``, drops cells next to the kinematic edge and Q rows within
    ``bragg_halfwidth`` of a sample-environment Bragg peak, then averages groups of
    consecutive rows into bins of about ``dq_target``.  A point is kept when at least
    ``min_row_frac`` of the rows in its bin are valid there.  Cells with zero counts
    are treated as missing here; background and resolution runs binned to match
    (``subtract``, ``attach_resolution``) count them as measured zeros.
    """
    r = read_raw(path)
    data, error, w, dwr, ei, qr = r["D"], r["Er"], r["w"], r["dw"], r["ei"], r["q_rows"]
    valid = np.isfinite(data) & np.isfinite(error) & (error > 0) & (data != 0)
    inner = valid.copy()  # drop cells next to the kinematic edge (partly filled)
    inner[1:, :] &= valid[:-1, :]
    inner[:-1, :] &= valid[1:, :]
    inner[:, 1:] &= valid[:, :-1]
    inner[:, :-1] &= valid[:, 1:]
    rows_ok = (qr >= qmin) & (qr <= qmax)
    for qb in BRAGG.values():
        rows_ok &= np.abs(qr - qb) > bragg_halfwidth
    wsel = np.where(np.abs(w) <= wmax)[0]
    per = max(1, int(round(dq_target / (qr[1] - qr[0]))))
    rows = np.where(rows_ok)[0]
    groups, cur = [], [rows[0]]
    for rr in rows[1:]:
        if rr == cur[-1] + 1 and len(cur) < per:
            cur.append(rr)
        else:
            if len(cur) >= max(2, per // 2):
                groups.append(np.array(cur))
            cur = [rr]
    if len(cur) >= max(2, per // 2):
        groups.append(np.array(cur))
    om, dws, qq, yy, ee, bid, qb_mean, kept = [], [], [], [], [], [], [], []
    b = 0
    for g in groups:
        ok = inner[np.ix_(g, wsel)]
        cnt = ok.sum(0)
        keep = cnt >= np.ceil(min_row_frac * len(g))
        if keep.sum() < 10:
            continue
        kept.append(g.tolist())
        c = np.maximum(cnt, 1)
        y = np.where(ok, data[np.ix_(g, wsel)], 0.0).sum(0) / c
        e = np.sqrt((np.where(ok, error[np.ix_(g, wsel)], 0.0) ** 2).sum(0)) / c  # error of the mean
        qv = np.where(ok, qr[g][:, None], 0.0).sum(0) / c
        idx = np.where(keep)[0]
        om.append(w[wsel][idx])
        dws.append(dwr[wsel][idx])
        yy.append(y[idx])
        ee.append(e[idx])
        qq.append(qv[idx])
        bid.append(np.full(len(idx), b))
        qb_mean.append(qv[idx].mean())
        b += 1
    return Dataset(
        name=r["name"],
        ei=ei,
        omega=np.concatenate(om),
        dw=np.concatenate(dws),
        q=np.concatenate(qq),
        y=np.concatenate(yy),
        err=np.concatenate(ee),
        bin_id=np.concatenate(bid),
        q_bins=np.array(qb_mean),
        n_bins=b,
        meta={
            "rows_per_bin": per,
            "qmin": qmin,
            "qmax": qmax,
            "wmax": wmax,
            "dq_target": dq_target,
            "min_row_frac": min_row_frac,
            "bragg_halfwidth": bragg_halfwidth,
            "groups": kept,  # the rows averaged into each Q-bin
        },
    )


_REQUIRED = object()


def _for_energy(value, ei, default=_REQUIRED):
    """A setting given either as one value or as {incident energy: value}."""
    if isinstance(value, dict):
        for k, v in value.items():
            if abs(k - ei) < 0.05:
                return v
        if default is _REQUIRED:
            raise KeyError(f"no setting for Ei = {ei} meV in {value}")
        return default
    return value


def _binned_like(ds, path):
    """Another run averaged over exactly the rows and energies of ``ds``'s points.

    Returns (values, errors, file name) aligned with ``ds``; NaN where the other run
    has too few valid cells.  Zero-count cells count as measured zeros.
    """
    r = read_raw(path)
    if abs(r["ei"] - ds.ei) > 0.05:
        raise ValueError(f"{r['name']} was measured at Ei = {r['ei']:.2f} meV, not {ds.ei:.2f} meV")
    data, error, w = r["D"], r["Er"], r["w"]
    valid = np.isfinite(data) & np.isfinite(error) & (error >= 0)
    inner = valid.copy()
    inner[1:, :] &= valid[:-1, :]
    inner[:-1, :] &= valid[1:, :]
    inner[:, 1:] &= valid[:, :-1]
    inner[:, :-1] &= valid[:, 1:]
    wsel = np.where(np.abs(w) <= ds.meta["wmax"] + 1e-9)[0]
    column = {round(float(v), 6): j for j, v in enumerate(w[wsel])}
    y, e = np.full(len(ds.y), np.nan), np.full(len(ds.y), np.nan)
    for b, rows in enumerate(ds.meta["groups"]):
        g = np.array(rows)
        ok = inner[np.ix_(g, wsel)]
        cnt = ok.sum(0)
        yb = np.where(ok, data[np.ix_(g, wsel)], 0.0).sum(0) / np.maximum(cnt, 1)
        eb = np.sqrt((np.where(ok, error[np.ix_(g, wsel)], 0.0) ** 2).sum(0)) / np.maximum(cnt, 1)
        need = np.ceil(ds.meta["min_row_frac"] * len(g))
        for i in np.where(ds.bin_id == b)[0]:
            j = column.get(round(float(ds.omega[i]), 6))
            if j is not None and cnt[j] >= need:
                y[i], e[i] = yb[j], eb[j]
    return y, e, r["name"]


def subtract(ds, path, scale=1.0):
    """``ds`` minus ``scale`` × another run (an empty can, say) at the same E_i.

    The other run is averaged over the same rows and energies.  Points it cannot
    cover are dropped (the number is kept in ``meta``); errors add in quadrature.
    """
    y_bg, e_bg, name = _binned_like(ds, path)
    keep = np.isfinite(y_bg)
    return dataclasses.replace(
        ds,
        omega=ds.omega[keep],
        dw=ds.dw[keep],
        q=ds.q[keep],
        y=ds.y[keep] - scale * y_bg[keep],
        err=np.sqrt(ds.err[keep] ** 2 + (scale * e_bg[keep]) ** 2),
        bin_id=ds.bin_id[keep],
        meta={**ds.meta, "subtracted": f"{scale:g} x {name}", "dropped_in_subtraction": int(np.sum(~keep))},
    )


def load(paths, qmin=0.3, qmax=2.6, window=1.25, dq_target=0.10, background=None, scale_errors=True, verbose=False):
    """Load several incident energies of one run, sorted by energy.

    ``qmin``, ``qmax`` and ``dq_target`` may be single values or dicts keyed by
    incident energy.  The energy window is ``window`` meV, or 0.7 E_i if smaller.
    ``background``: {incident energy: path} of runs to subtract (empty can);
    energies without an entry are left as they are.
    With ``scale_errors`` the error bars are multiplied by sqrt(noise ratio) when
    the audit finds more scatter than they predict; the factor is kept in ``meta``.
    """
    out = []
    for p in paths:
        ei = read_raw(p)["ei"]
        ds = prepare(
            p,
            qmin=_for_energy(qmin, ei),
            qmax=_for_energy(qmax, ei),
            wmax=min(window, round(0.7 * ei, 2)),
            dq_target=_for_energy(dq_target, ei),
        )
        bg = None if background is None else _for_energy(background, ei, default=None)
        if bg is not None:
            ds = subtract(ds, bg)
        s0 = resolution_sigma(ei)
        ratio, n = noise_ratio(ds, s0)
        factor = float(np.sqrt(max(ratio, 1.0))) if scale_errors else 1.0
        ds.err = ds.err * factor
        ds.meta.update(sigma0=s0, noise_ratio=ratio, noise_triples=n, error_scale=factor)
        if verbose:
            print(
                f"{ds.name}: Ei {ei:.2f} meV, {len(ds.y)} points in {ds.n_bins} Q-bins, "
                f"noise ratio {ratio:.2f} -> errors x{factor:.2f}"
            )
        out.append(ds)
    return sorted(out, key=lambda d: d.ei)


def attach_resolution(ds, path, background=None):
    """Use a measured spectrum (vanadium, or the sample frozen) as the resolution of ``ds``.

    The run is averaged over the same rows and energies as ``ds`` (minus
    ``background``, an empty-can run, if given) and each Q-bin's spectrum,
    normalised to unit area, becomes that bin's convolution kernel (``Problem``
    uses it when the model's ``resolution`` is 'measured').  A frozen sample that
    still moves, like benzene at 260 K, makes the kernel too broad.
    """
    y, _, name = _binned_like(ds, path)
    if background is not None:
        y = y - _binned_like(ds, background)[0]
    offsets, energies, weights, widths = [0], [], [], []
    for b in range(ds.n_bins):
        sel = (ds.bin_id == b) & np.isfinite(y)
        area = float(np.sum(y[sel] * ds.dw[sel]))
        if area <= 0:
            raise ValueError(f"resolution spectrum has no positive area in Q-bin {b}")
        energies.append(ds.omega[sel])
        weights.append(y[sel] * ds.dw[sel] / area)
        widths.append(ds.dw[sel])
        offsets.append(offsets[-1] + int(sel.sum()))
    ds.meta["kernel"] = {
        "offsets": np.array(offsets, np.int64),
        "energies": np.concatenate(energies),
        "weights": np.concatenate(weights),
        "widths": np.concatenate(widths),
        "source": name,
    }
    return ds
