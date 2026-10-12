"""Model specification, parameters, priors and the joint likelihood.

One ``Problem`` = one physical model fitted to several datasets (incident
energies) at once.  Physics parameters are shared; each energy has its own
resolution width and elastic-line offset; the resolution line shape is shared.
By default the per-Q-bin amplitude, can term and background are solved exactly
inside the likelihood (``forward.chi2_nnls``), so they never appear as parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import KB
from .forward import FICKIAN, JUMP, chi2_nnls, spectrum, spectrum_kernel
from .rotation import channel_weights

KAPPA_MAX = 3.0
PHYSICS = ("lnD", "lntau0", "lnDr", "lnDt", "lnDs")


@dataclass(frozen=True)
class ModelSpec:
    """Which physics to fit, and how.

    translation: 'fickian' or 'jump' (Singwi–Sjölander, residence time τ₀).
    rotation: 'none', 'iso' (one rate D_r) or 'aniso' (symmetric top: tumbling
    D_t, spinning D_s).  lmax=None expands rotation until the sum rule is met.
    asym: fit a two-component, asymmetric resolution line shape.
    use_el: a resolution-shaped elastic term per Q-bin (can scattering etc.).
    amplitudes: 'per_bin' (free scale per Q-bin, solved exactly) or 'global' (one
    scale per energy times exp(−Q²⟨u²⟩/3), with ⟨u²⟩ shared).
    resolution: 'analytic' (fitted Voigt-based line) or 'measured' (convolve with
    the kernel attached by ``let.attach_resolution``; energies without one stay analytic).
    detailed_balance: multiply by exp(ħω/2kT).
    """

    name: str
    translation: str = "jump"
    rotation: str = "iso"
    lmax: int | None = None
    R: float = 2.48  # Å, radius of the scatterer's circle (benzene H)
    theta0: float = math.pi / 2  # polar angle of the scatterer from the symmetry axis
    T: float = 290.0  # K
    asym: bool = True
    use_el: bool = True
    amplitudes: str = "per_bin"
    resolution: str = "analytic"
    detailed_balance: bool = True


# The model ladder used in the benzene analysis, simplest physics last.
LADDER = {
    "jump+iso": ModelSpec("jump+iso", "jump", "iso"),
    "jump+aniso": ModelSpec("jump+aniso", "jump", "aniso"),
    "fickian+aniso": ModelSpec("fickian+aniso", "fickian", "aniso"),
    "fickian+iso": ModelSpec("fickian+iso", "fickian", "iso"),
    "fickian+aniso_l2": ModelSpec("fickian+aniso_l2", "fickian", "aniso", lmax=2),
    "jump_only": ModelSpec("jump_only", "jump", "none"),
}


class Problem:
    """Parameters, priors and the joint posterior of one model on several datasets.

    Parameter vector (all on the scales below):
      lnD; lntau0 (jump only); lnDr, or lnDt and lnDs;
      per dataset with an analytic resolution: lnsig_<Ei> (normal prior around the
      instrument's resolution width, sd 0.3 in log) and e0_<Ei> (normal prior N(0, σ₀/4));
      kappa (Lorentzian part of the resolution, in units of sigma), and with asym
      f2, d2, s2 (second resolution component), if any resolution is analytic;
      global amplitudes only: lnC_<Ei> per dataset and the shared u2 (⟨u²⟩, Å²).
    """

    def __init__(self, spec: ModelSpec, datasets):
        if spec.translation not in ("fickian", "jump"):
            raise ValueError("translation must be 'fickian' or 'jump'")
        if spec.amplitudes not in ("per_bin", "global"):
            raise ValueError("amplitudes must be 'per_bin' or 'global'")
        if spec.resolution not in ("analytic", "measured"):
            raise ValueError("resolution must be 'analytic' or 'measured'")
        self.spec = spec
        self.datasets = list(datasets)
        self.measured = [spec.resolution == "measured" and "kernel" in d.meta for d in self.datasets]
        if spec.resolution == "measured" and not any(self.measured):
            raise ValueError("resolution='measured' needs a kernel: call let.attach_resolution first")
        self.names, lo, hi, self.kind = [], [], [], []

        def add(n, a, b, kind="u"):
            self.names.append(n)
            lo.append(a)
            hi.append(b)
            self.kind.append(kind)

        add("lnD", math.log(0.01), math.log(2.0))
        if spec.translation == "jump":
            add("lntau0", math.log(0.01), math.log(20.0))
        if spec.rotation == "iso":
            add("lnDr", math.log(0.003), math.log(5.0))
        elif spec.rotation == "aniso":
            add("lnDt", math.log(0.003), math.log(5.0))
            add("lnDs", math.log(0.003), math.log(5.0))
        for d, meas in zip(self.datasets, self.measured, strict=True):
            if meas:
                continue
            s0 = d.meta["sigma0"]
            add(f"lnsig_{d.ei:.2f}", math.log(s0) - 1.5, math.log(s0) + 1.5, ("n", math.log(s0), 0.3))
            add(f"e0_{d.ei:.2f}", -s0, s0, ("n", 0.0, 0.25 * s0))
        if not all(self.measured):
            add("kappa", 0.0, KAPPA_MAX)
            if spec.asym:
                add("f2", 0.0, 0.6)
                add("d2", -4.0, 4.0)
                add("s2", 0.7, 4.0)
        if spec.amplitudes == "global":
            for d in self.datasets:
                add(f"lnC_{d.ei:.2f}", math.log(1e-4), math.log(1e4))
            add("u2", 0.0, 0.5)
        self.lo, self.hi = np.array(lo), np.array(hi)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self._W = [channel_weights(d.q, spec.rotation, spec.R, spec.lmax, spec.theta0) for d in self.datasets]
        self._buf = [(np.empty(len(d.y)), np.empty(len(d.y)), np.zeros((d.n_bins, 3))) for d in self.datasets]
        self._zeros = [np.zeros(len(d.y)) for d in self.datasets]

    @property
    def physics_names(self):
        return [n for n in self.names if n in PHYSICS]

    def physics(self, th):
        """Physical parameters in natural units: D (Å²/ps), tau0 (ps), Dt, Ds (ps⁻¹)."""
        g = self.idx
        out = {"D": math.exp(th[g["lnD"]]), "tau0": math.exp(th[g["lntau0"]]) if "lntau0" in g else 0.0}
        if self.spec.rotation == "iso":
            out["Dt"] = out["Ds"] = math.exp(th[g["lnDr"]])
        elif self.spec.rotation == "aniso":
            out["Dt"], out["Ds"] = math.exp(th[g["lnDt"]]), math.exp(th[g["lnDs"]])
        else:
            out["Dt"] = out["Ds"] = 0.0
        return out

    def predict(self, th, k):
        """Model line shape and elastic term for dataset k (before any amplitude)."""
        d = self.datasets[k]
        p = self.physics(th)
        W, lch, mch = self._W[k]
        shape, el, _ = self._buf[k]
        transl = JUMP if self.spec.translation == "jump" else FICKIAN
        kT = KB * self.spec.T if self.spec.detailed_balance else math.inf
        if self.measured[k]:
            kern = d.meta["kernel"]
            spectrum_kernel(
                d.omega,
                d.q,
                d.bin_id,
                W,
                lch,
                mch,
                p["D"],
                p["tau0"],
                transl,
                p["Dt"],
                p["Ds"],
                kT,
                kern["offsets"],
                kern["energies"],
                kern["weights"],
                kern["widths"],
                shape,
                el,
            )
            return shape, el
        sig = math.exp(th[self.idx[f"lnsig_{d.ei:.2f}"]])
        e0 = th[self.idx[f"e0_{d.ei:.2f}"]]
        kap = th[self.idx["kappa"]]
        if self.spec.asym:
            f2, d2, s2 = th[self.idx["f2"]], th[self.idx["d2"]], th[self.idx["s2"]]
        else:
            f2, d2, s2 = 0.0, 0.0, 1.0
        spectrum(
            d.omega,
            d.dw,
            d.q,
            W,
            lch,
            mch,
            p["D"],
            p["tau0"],
            transl,
            p["Dt"],
            p["Ds"],
            sig,
            kap * sig,
            e0,
            kT,
            shape,
            el,
            f2,
            d2,
            s2,
        )
        return shape, el

    def _global_scale(self, th, k):
        d = self.datasets[k]
        return math.exp(th[self.idx[f"lnC_{d.ei:.2f}"]]) * np.exp(-(d.q**2) * th[self.idx["u2"]] / 3.0)

    def _chi2(self, th, k, amps):
        d = self.datasets[k]
        shape, el = self.predict(th, k)
        if self.spec.amplitudes == "global":
            m = self._global_scale(th, k) * shape
            # only the elastic term and the background are left to solve per bin
            return chi2_nnls(d.y - m, d.err, self._zeros[k], el, d.bin_id, d.n_bins, self.spec.use_el, amps), m
        return chi2_nnls(d.y, d.err, shape, el, d.bin_id, d.n_bins, self.spec.use_el, amps), None

    def model_curve(self, th, k):
        """The fitted model for dataset k, with its exactly solved terms applied."""
        d = self.datasets[k]
        amps = np.zeros((d.n_bins, 3))
        _, m = self._chi2(th, k, amps)
        shape, el, _ = self._buf[k]
        a = amps[d.bin_id]
        if m is None:
            return a[:, 0] * shape + a[:, 1] * el + a[:, 2]
        return m + a[:, 1] * el + a[:, 2]

    def chi2_parts(self, th):
        """χ² of each dataset at parameters ``th``."""
        return np.array([self._chi2(th, k, self._buf[k][2])[0] for k in range(len(self.datasets))])

    def scale_amplitudes(self, th):
        """Set the global scales (lnC_<Ei>) of ``th`` by least squares; no-op for per-bin amplitudes."""
        if self.spec.amplitudes != "global":
            return th
        for k, d in enumerate(self.datasets):
            shape, _ = self.predict(th, k)
            m = self._global_scale(th, k) / math.exp(th[self.idx[f"lnC_{d.ei:.2f}"]]) * shape
            w = 1.0 / d.err**2
            c = float(np.sum(w * m * d.y) / max(np.sum(w * m * m), 1e-300))
            i = self.idx[f"lnC_{d.ei:.2f}"]
            th[i] = np.clip(math.log(max(c, 1e-300)), self.lo[i] + 1e-6, self.hi[i] - 1e-6)
        return th

    def logprior(self, th):
        if np.any(th < self.lo) or np.any(th > self.hi):
            return -np.inf
        lp = 0.0
        for v, kd in zip(th, self.kind, strict=True):
            if kd != "u":
                lp += -0.5 * ((v - kd[1]) / kd[2]) ** 2
        return lp

    def logpost(self, th):
        lp = self.logprior(th)
        if not np.isfinite(lp):
            return -np.inf
        return lp - 0.5 * self.chi2_parts(th).sum()

    def sample_prior(self, rng):
        th = np.empty(len(self.names))
        for i, kd in enumerate(self.kind):
            if kd == "u":
                th[i] = rng.uniform(self.lo[i], self.hi[i])
            else:
                th[i] = np.clip(rng.normal(kd[1], kd[2]), self.lo[i], self.hi[i])
        return th

    def centre(self):
        """A neutral starting point: prior centres for the instrument parameters."""
        th = 0.5 * (self.lo + self.hi)
        for i, kd in enumerate(self.kind):
            if kd != "u":
                th[i] = kd[1]
        if "kappa" in self.idx:
            th[self.idx["kappa"]] = 0.5
        if "f2" in self.idx:
            th[self.idx["f2"]], th[self.idx["d2"]], th[self.idx["s2"]] = 0.2, -1.0, 1.5
        if "u2" in self.idx:
            th[self.idx["u2"]] = 0.05
        return th
