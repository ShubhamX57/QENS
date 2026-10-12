"""The reader: Q axis, binning and the error-bar audit."""

import numpy as np

from uqens import let


def test_q_axis_puts_aluminium_bragg_peak_in_place(data_dir):
    """The recovered Q grid must put the Al (111) peak of the sample can at 2.687 Å⁻¹."""
    r = let.read_raw(data_dir / "benzene_290_861_coh.nxspe")
    elastic = np.abs(r["w"]) < 0.4
    ok = (np.isfinite(r["D"]) & (r["Er"] > 0))[:, elastic]
    counts = ok.sum(1)
    sums = np.where(ok, r["D"][:, elastic], 0.0).sum(1)
    profile = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    q = r["q_rows"]
    win = (q > 2.55) & (q < 2.82)
    base = np.nanmedian(profile[win])
    peak = int(np.nanargmax(np.where(win, profile, -np.inf)))
    near = slice(peak - 2, peak + 3)
    w = np.clip(profile[near] - base, 0, None)
    centroid = float(np.sum(w * q[near]) / np.sum(w))
    assert abs(centroid - let.BRAGG["Al(111)"]) < 0.01


def test_binned_data_are_clean(benzene):
    assert [round(d.ei, 2) for d in benzene] == [1.24, 1.97, 3.60, 8.61]
    for d in benzene:
        n = len(d.y)
        assert n > 1000 and d.n_bins >= 10
        for a in (d.omega, d.dw, d.q, d.y, d.err):
            assert len(a) == n and np.all(np.isfinite(a))
        assert np.all(d.err > 0)
        assert set(np.unique(d.bin_id)) == set(range(d.n_bins))
        assert np.all(np.abs(d.omega) <= d.meta["wmax"])
        assert d.q.min() >= d.meta["qmin"] - 1e-9 and d.q.max() <= d.meta["qmax"] + 1e-9


def test_error_bars_are_audited(benzene):
    """Neighbouring Q rows share counts, so the propagated errors understate the scatter."""
    for d in benzene:
        assert d.meta["noise_triples"] > 200
        assert 1.0 < d.meta["noise_ratio"] < 2.0
        assert d.meta["error_scale"] == np.sqrt(d.meta["noise_ratio"])


def test_resolution_prior_scales_with_energy():
    widths = [let.resolution_sigma(e) for e in (1.24, 1.97, 3.60, 8.61)]
    assert widths == sorted(widths)
    assert abs(let.resolution_sigma(3.60) - let.SIGMA_360) < 1e-12
