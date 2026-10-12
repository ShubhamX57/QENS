"""Model specification, parameters, priors and the joint likelihood.

One ``Problem`` = one physical model fitted to several datasets (incident
energies) at once.  Physics parameters are shared; each energy has its own
resolution width and elastic-line offset; the resolution line shape is shared.
Per-Q-bin amplitudes, can term and background are solved exactly inside the
likelihood (``forward.chi2_nnls``), so they never appear as parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import KB
from .forward import FICKIAN, JUMP, chi2_nnls, spectrum
from .rotation import channel_weights

KAPPA_MAX = 3.0
PHYSICS = ("lnD", "lntau0", "lnDr", "lnDt", "lnDs")


@dataclass(frozen=True)
class ModelSpec:
    """Which physics to fit.

    translation: 'fickian' or 'jump' (Singwi–Sjölander, residence time τ₀).
    rotation: 'none', 'iso' (one rate D_r) or 'aniso' (symmetric top: tumbling
    D_t, spinning D_s).  lmax=None expands rotation until the sum rule is met.
    asym: fit a two-component, asymmetric resolution line shape.
    """

    name: str
    translation: str = "jump"
    rotation: str = "iso"
    lmax: int | None = None
    R: float = 2.48  # Å, radius of the scatterer's circle (benzene H)
    theta0: float = math.pi / 2  # polar angle of the scatterer from the symmetry axis
    T: float = 290.0  # K
    asym: bool = True
    use_el: bool = True  # resolution-shaped elastic term per Q-bin (can etc.)


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
      per dataset: lnsig_<Ei> (normal prior around the instrument's resolution width,
      sd 0.3 in log) and e0_<Ei> (normal prior N(0, σ₀/4));
      kappa (Lorentzian part of the resolution, in units of sigma);
      asym only: f2, d2, s2 (second resolution component).
    """

    def __init__(self, spec: ModelSpec, datasets):
        if spec.translation not in ("fickian", "jump"):
            raise ValueError("translation must be 'fickian' or 'jump'")
        self.spec = spec
        self.datasets = list(datasets)
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
        for d in self.datasets:
            s0 = d.meta["sigma0"]
            add(f"lnsig_{d.ei:.2f}", math.log(s0) - 1.5, math.log(s0) + 1.5, ("n", math.log(s0), 0.3))
            add(f"e0_{d.ei:.2f}", -s0, s0, ("n", 0.0, 0.25 * s0))
        add("kappa", 0.0, KAPPA_MAX)
        if spec.asym:
            add("f2", 0.0, 0.6)
            add("d2", -4.0, 4.0)
            add("s2", 0.7, 4.0)
        self.lo, self.hi = np.array(lo), np.array(hi)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self._W = [channel_weights(d.q, spec.rotation, spec.R, spec.lmax, spec.theta0) for d in self.datasets]
        self._buf = [(np.empty(len(d.y)), np.empty(len(d.y)), np.zeros((d.n_bins, 3))) for d in self.datasets]

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
        """Model line shape and elastic term for dataset k (before the per-bin amplitudes)."""
        d = self.datasets[k]
        p = self.physics(th)
        W, lch, mch = self._W[k]
        sig = math.exp(th[self.idx[f"lnsig_{d.ei:.2f}"]])
        e0 = th[self.idx[f"e0_{d.ei:.2f}"]]
        kap = th[self.idx["kappa"]]
        if self.spec.asym:
            f2, d2, s2 = th[self.idx["f2"]], th[self.idx["d2"]], th[self.idx["s2"]]
        else:
            f2, d2, s2 = 0.0, 0.0, 1.0
        shape, el, _ = self._buf[k]
        transl = JUMP if self.spec.translation == "jump" else FICKIAN
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
            KB * self.spec.T,
            shape,
            el,
            f2,
            d2,
            s2,
        )
        return shape, el

    def model_curve(self, th, k):
        """The fitted model for dataset k, with its exactly solved amplitudes applied."""
        d = self.datasets[k]
        shape, el = self.predict(th, k)
        amps = np.zeros((d.n_bins, 3))
        chi2_nnls(d.y, d.err, shape, el, d.bin_id, d.n_bins, self.spec.use_el, amps)
        a = amps[d.bin_id]
        return a[:, 0] * shape + a[:, 1] * el + a[:, 2]

    def chi2_parts(self, th):
        """χ² of each dataset at parameters ``th``."""
        out = []
        for k, d in enumerate(self.datasets):
            shape, el = self.predict(th, k)
            amps = self._buf[k][2]
            out.append(chi2_nnls(d.y, d.err, shape, el, d.bin_id, d.n_bins, self.spec.use_el, amps))
        return np.array(out)

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
        th[self.idx["kappa"]] = 0.5
        if self.spec.asym:
            th[self.idx["f2"]], th[self.idx["d2"]], th[self.idx["s2"]] = 0.2, -1.0, 1.5
        return th
