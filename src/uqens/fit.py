"""Finding the best fit: many starts, every distinct optimum kept.

The posterior can have several modes (for example spinning faster than tumbling,
or the reverse).  A single start can stop in the wrong one without any warning,
so ``fit`` runs several and reports every distinct optimum it finds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .model import PHYSICS, Problem

# Physics starting points spanning both anisotropy orderings and the isotropic case.
DEFAULT_SEEDS = (
    {"lnD": math.log(0.1), "lntau0": 0.0, "lnDt": math.log(0.05), "lnDs": math.log(0.2)},
    {"lnD": math.log(0.1), "lntau0": 0.0, "lnDt": math.log(0.2), "lnDs": math.log(0.05)},
    {"lnD": math.log(0.1), "lntau0": 0.0, "lnDt": math.log(0.1), "lnDs": math.log(0.1)},
)


NELDER_MEAD = {"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-6, "adaptive": True}


def starts(problem: Problem, n_random=6, seed=0, seeds=DEFAULT_SEEDS):
    """Seeded starts (physics from ``seeds``, instrument at prior centres) plus random prior draws."""
    rng = np.random.default_rng(seed)
    out = []
    for s in seeds:
        th = problem.centre()
        for n, v in s.items():
            if n in problem.idx:
                i = problem.idx[n]
                th[i] = np.clip(v, problem.lo[i] + 1e-6, problem.hi[i] - 1e-6)
        if "lnDr" in problem.idx:
            th[problem.idx["lnDr"]] = 0.5 * (s["lnDt"] + s["lnDs"])
        out.append(th)
    out += [problem.sample_prior(rng) for _ in range(n_random)]
    return [problem.scale_amplitudes(th) for th in out]


def optimise(problem: Problem, x0):
    """Bounded quasi-Newton, then Nelder–Mead to polish across the solver's kinks."""
    f = lambda th: -problem.logpost(th)  # noqa: E731
    bounds = list(zip(problem.lo + 1e-9, problem.hi - 1e-9, strict=True))
    r = minimize(f, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000, "eps": 1e-6})
    r = minimize(f, r.x, method="Nelder-Mead", options=NELDER_MEAD)
    x = np.clip(r.x, problem.lo, problem.hi)
    return f(x), x


def distinct_optima(results, problem: Problem, tol=0.05, dchi=50.0):
    """Group end points into modes whose physics differs by more than ``tol`` (in log)."""
    best = results[0][0]
    phys = [problem.idx[n] for n in problem.names if n in PHYSICS]
    modes = []
    for v, x in results:
        if v - best > dchi:
            continue
        if all(np.max(np.abs(x[phys] - m[1][phys])) > tol for m in modes):
            modes.append((v, x))
    return modes


@dataclass
class FitResult:
    problem: Problem
    x: np.ndarray  # best parameter vector
    neglogpost: float
    modes: list = field(default_factory=list)  # (neglogpost, x) of each distinct optimum
    runs: list = field(default_factory=list)  # every start's end point, best first

    def physics(self):
        return self.problem.physics(self.x)

    def chi2(self):
        """χ² per dataset at the best fit."""
        return dict(zip([d.ei for d in self.problem.datasets], self.problem.chi2_parts(self.x), strict=True))

    def summary(self):
        p = self.physics()
        chi = self.chi2()
        n = {d.ei: len(d.y) for d in self.problem.datasets}
        lines = [
            f"{self.problem.spec.name}: -log posterior {self.neglogpost:.1f}, "
            f"{len(self.modes)} distinct optimum(s) from {len(self.runs)} starts",
            "  " + ", ".join(f"{k} = {v:.4g}" for k, v in p.items() if v),
            "  chi2/N per energy: " + ", ".join(f"{e:.2f} meV {c / n[e]:.2f}" for e, c in chi.items()),
        ]
        for v, x in self.modes[1:]:
            q = ", ".join(f"{k} = {w:.4g}" for k, w in self.problem.physics(x).items() if w)
            lines.append(f"  other optimum (+{v - self.neglogpost:.1f}): {q}")
        return "\n".join(lines)


def fit(problem: Problem, n_random=6, seed=0, seeds=DEFAULT_SEEDS, x0=None, verbose=False):
    """Best fit from seeded and random starts (or from the given ``x0`` list)."""
    inits = x0 if x0 is not None else starts(problem, n_random, seed, seeds)
    runs = []
    for k, th in enumerate(inits):
        runs.append(optimise(problem, th))
        if verbose:
            print(f"  start {k + 1}/{len(inits)}: -log posterior {runs[-1][0]:.1f}", flush=True)
    runs.sort(key=lambda t: t[0])
    return FitResult(problem, runs[0][1], runs[0][0], distinct_optima(runs, problem), runs)
