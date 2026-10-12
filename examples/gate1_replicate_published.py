"""Gate 1: rerun the published LET fit at 3.60 meV, with and without two fixes.

The model, data handling and constants below are copied from the paper's own code,
github.com/scams-research/fitting-qens (commit 8f9fcd3; files plet_data.py,
experiment_nested_sampling/utility/{data,model,bounds}.py and
experiment_nested_sampling/aniso/runner.py), restricted to 3.60 meV because this
repository has no 1.97 meV empty-can or 260 K run.  Instead of nested sampling, each
variant is fitted by optimisation from a few starts, which is enough to compare D.

Variants:
  published    the code as it is
  grid         the model evaluated on the data's own energy spacing, not linspace(-2, 2, 2N-1)
  grid+window  also bin centres as (omega[:-1] + omega[1:]) / 2, so the window is really ±1.25 meV

Needs scipp:  pip install scipp
    python examples/gate1_replicate_published.py            # all three variants, a few minutes each

Portions copied from fitting-qens are under this licence:

    MIT License

    Copyright (c) 2026 @scams-research

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import scipp as sc
from scipy import constants
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import spherical_jn

ROOT = Path(__file__).resolve().parents[1]
STEP = 0.018  # LET energy bin at 3.60 meV
Q_EDGES = sc.arange("q", 0.05409, 2.91309 + (0.0115 / 2), 0.0115, unit=sc.Unit("1/angstrom"))
OMEGA_EDGES = sc.arange("omega", -2.88, 2.88 + (0.018 / 2), 0.018, unit=sc.Unit("meV"))
BOUNDS = [(0.01, 0.2), (0.06, 1), (0.0001, 0.15), (0.00001, 0.18), (0.01, 30)]  # D_fick, D_parr, D_perp, u2, norm
BKG = (0.001, 10)


class PletData:
    """plet_data.PletData; ``true_centres`` switches on the bin-centre fix."""

    def __init__(self, filename, omega_lims, q_lims, true_centres):
        with h5py.File(filename, "r") as file:
            entry = file[list(file.keys())[0]]
            values, errors = entry["data"]["data"][:], entry["data"]["error"][:]
        self.q, self.omega = Q_EDGES, OMEGA_EDGES
        self.data = sc.DataArray(
            data=sc.array(dims=["q", "omega"], values=values, variances=errors**2),
            coords={"q": self.q, "omega": self.omega},
        )
        self.q_mid = (self.q[:-1] + self.q[1:]) / 2
        if true_centres:
            self.omega_mid = (self.omega[:-1] + self.omega[1:]) / 2
        else:
            self.omega_mid = self.omega[:-1] + self.omega[1:] / 2  # as published
        self.data.masks["omega"] = (self.omega_mid < omega_lims[0] * sc.units.meV) | (
            self.omega_mid > omega_lims[1] * sc.units.meV
        )
        self.q_lims = q_lims
        self.data.masks["q"] = (self.q_mid < q_lims[0] * sc.Unit("1/angstrom")) | (
            self.q_mid > q_lims[1] * sc.Unit("1/angstrom")
        )

    def bin_q(self, q_bins):
        binning_factor = np.histogram(self.q.values, bins=q_bins.values)[0]
        self.data = sc.rebin(self.data, q=q_bins) / float(binning_factor[1])
        self.q = self.data.coords["q"]
        self.q_mid = (self.q[:-1] + self.q[1:]) / 2
        self.data.masks["q"] = (self.q_mid < self.q_lims[0] * sc.Unit("1/angstrom")) | (
            self.q_mid > self.q_lims[1] * sc.Unit("1/angstrom")
        )

    def _select(self, a):
        return a[np.invert(self.data.masks["q"].values)][:, np.invert(self.data.masks["omega"].values)]

    @property
    def masked(self):
        return self._select(self.data.data.values)

    @property
    def errors(self):
        return np.sqrt(self._select(self.data.data.variances))


def lorentzian(x, gamma, amplitude):
    return (amplitude / np.pi) * (gamma / (x**2 + gamma**2))


def high_level_model(omega, q, d_fick, d_parr, d_perp, r1, u2):
    lorentzian_fick = lorentzian(omega, d_fick * q**2, np.exp(-(q**2) * u2 / 3))
    amp1, amp2 = spherical_jn(1, q * r1) ** 2, spherical_jn(2, q * r1) ** 2
    l1 = lorentzian(omega, d_parr + d_perp, amp1)
    l2 = lorentzian(omega, 6 * d_perp, amp2 / 4) + lorentzian(omega, 2 * d_perp + 4 * d_parr, 3 * amp2 / 4)
    delta_fn = np.zeros_like(omega)
    delta_fn[np.argmin(np.abs(omega))] = spherical_jn(0, q * r1) ** 2 / (omega[1] - omega[0])
    return fftconvolve(3 * l1 + 5 * l2 + delta_fn, lorentzian_fick, mode="same")


def run(data_dir, variant, n_starts):
    fix_grid, fix_window = variant in ("grid", "grid+window"), variant == "grid+window"
    kw = {"omega_lims": [-1.25, 1.25], "q_lims": [0.6, 1.8], "true_centres": fix_window}
    data = PletData(data_dir / "benzene_290_360_inc.nxspe", **kw)
    empty = PletData(data_dir / "empty_360_inc.nxspe", **kw)
    res = PletData(data_dir / "benzene_260_360_inc.nxspe", **kw)
    q_bins = sc.linspace("q", data.q.min().values, data.q.max().values, 31, unit=sc.Unit("1/angstrom"))
    for obj in (data, empty, res):
        obj.bin_q(q_bins)
    q_mid = data.q_mid.values[np.invert(data.data.masks["q"].values)]
    data.data -= empty.data
    res.data -= empty.data
    data.data /= 2
    y, err, kernel = data.masked, data.errors, res.masked
    n = y.shape[1]
    if fix_grid:
        omega_conv = STEP * (np.arange(2 * n - 1) - (n - 1))
    else:
        omega_conv = np.linspace(-2, 2, (n * 2) - 1)  # as published

    def chi2(p):
        d_fick, d_parr, d_perp, u2, norm = p[:5]
        total = 0.0
        for k in range(q_mid.size):
            unconv = high_level_model(omega_conv, q_mid[k], d_fick, d_parr, d_perp, 2.48, u2) / norm + p[5 + k]
            model = fftconvolve(kernel[k] / kernel[k].sum(), unconv, mode="valid")
            total += np.sum(((model - y[k]) / err[k]) ** 2)
        return float(total)

    bounds = BOUNDS + [BKG] * q_mid.size
    lo, hi = np.array(bounds).T
    rng = np.random.default_rng(1)
    best = None
    for s in range(n_starts):
        x0 = rng.uniform(lo, hi)
        if s == 0:
            x0[:5] = [0.1, 0.2, 0.02, 0.05, 1.0]
        r = minimize(chi2, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000})
        r = minimize(
            chi2, r.x, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8, "adaptive": True}
        )
        r = minimize(chi2, np.clip(r.x, lo, hi), method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    hbar = constants.hbar / constants.e * 1e15  # meV ps
    centres = data.omega_mid.values[np.invert(data.data.masks["omega"].values)]
    true_halfwidth = (centres[-1] - centres[0]) / (1.0 if fix_window else 1.5) / 2
    print(
        f"{variant:12s} window ±{true_halfwidth:.2f} meV, model spacing {omega_conv[1] - omega_conv[0]:.4f} meV "
        f"(data {STEP}): D = {x[0] / hbar:.3f} Å²/ps, D_s = {x[1] / hbar:.3f}, D_t = {x[2] / hbar:.4f} ps⁻¹, "
        f"ratio {x[1] / x[2]:.0f}, chi2/N {best.fun / y.size:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--variant", choices=["published", "grid", "grid+window", "all"], default="all")
    parser.add_argument("--starts", type=int, default=3)
    args = parser.parse_args()
    for v in ["published", "grid", "grid+window"] if args.variant == "all" else [args.variant]:
        run(args.data, v, args.starts)
