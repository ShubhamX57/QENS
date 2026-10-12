"""Tests a result must pass before it is reported.

``holdout``: fit the physics on all incident energies but one and predict the one
left out.  A model that only fits does not count; it has to predict.

``coverage``: simulate datasets from a fitted model with the real grids and error
bars, refit each from a start away from the truth, and check how often the error
bars contain the true value.  Starting at the truth flatters the error bars
(the optimiser stops early), so starts are offset by default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .fit import NELDER_MEAD, optimise
from .model import PHYSICS, ModelSpec, Problem


def transfer(src: Problem, x_src, dst: Problem):
    """Parameter vector for ``dst`` taking values from ``src`` where names match."""
    th = dst.centre()
    for n, v in zip(src.names, x_src, strict=True):
        if n in dst.idx:
            i = dst.idx[n]
            th[i] = np.clip(v, dst.lo[i] + 1e-6, dst.hi[i] - 1e-6)
    return th


def fit_with_fixed(problem: Problem, x0, fixed: dict):
    """Best fit over all parameters except those in ``fixed`` (name -> value)."""
    fix = {problem.idx[n]: v for n, v in fixed.items() if n in problem.idx}
    free = [i for i in range(len(problem.names)) if i not in fix]

    def full(z):
        th = np.array(x0, float)
        th[free] = z
        for i, v in fix.items():
            th[i] = v
        return th

    f = lambda z: -problem.logpost(full(z))  # noqa: E731
    bounds = list(zip(problem.lo[free] + 1e-9, problem.hi[free] - 1e-9, strict=True))
    r = minimize(
        f, np.asarray(x0, float)[free], method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000, "eps": 1e-6}
    )
    r = minimize(f, r.x, method="Nelder-Mead", options=NELDER_MEAD)
    th = full(np.clip(r.x, problem.lo[free], problem.hi[free]))
    return th, float(problem.chi2_parts(th).sum())


def holdout(spec: ModelSpec, datasets, best: Problem, x_best, verbose=False):
    """Leave-one-energy-out prediction test.

    For each energy: the physics is fitted on the others; on the held-out energy only
    the instrument parameters and the per-bin amplitudes are refitted.  ``penalty``
    is that χ² minus the χ² with everything refitted (0 = perfect transfer).
    """
    out = []
    for h, held in enumerate(datasets):
        train = Problem(spec, [d for k, d in enumerate(datasets) if k != h])
        _, xt = optimise(train, transfer(best, x_best, train))
        phys = {n: xt[train.idx[n]] for n in train.names if n in PHYSICS}
        one = Problem(spec, [held])
        _, xo = optimise(one, transfer(best, x_best, one))
        chi_free = float(one.chi2_parts(xo).sum())
        x0 = xo.copy()
        for n, v in phys.items():
            x0[one.idx[n]] = v
        _, chi_transfer = fit_with_fixed(one, x0, phys)
        row = {
            "ei": held.ei,
            "n": len(held.y),
            "chi2_transfer": chi_transfer,
            "chi2_free": chi_free,
            "penalty": chi_transfer - chi_free,
            "train_physics": train.physics(xt),
        }
        out.append(row)
        if verbose:
            print(
                f"  {held.ei:.2f} meV held out: chi2 {chi_transfer:.1f} (all free {chi_free:.1f}, "
                f"penalty {row['penalty']:+.1f}, N = {row['n']})",
                flush=True,
            )
    return out


@dataclass
class Coverage:
    names: list  # physics parameters (log scale)
    deviations: np.ndarray  # replicas × parameters: estimate − truth
    widths: dict  # name -> 1σ width used for the intervals (log scale)

    def inside(self, name, z=1.0):
        j = self.names.index(name)
        return float(np.mean(np.abs(self.deviations[:, j]) < z * self.widths[name]))

    def report(self):
        lines = [f"{len(self.deviations)} synthetic replicas"]
        for j, n in enumerate(self.names):
            d = self.deviations[:, j]
            lines.append(
                f"  {n[2:]:6s} scatter {d.std(ddof=1) * 100:.2f}% vs width {self.widths[n] * 100:.2f}%, "
                f"bias {d.mean() * 100:+.2f}%; inside 68%: {self.inside(n, 1.0) * 100:.0f}%, "
                f"inside 95%: {self.inside(n, 1.96) * 100:.0f}%"
            )
        return "\n".join(lines)


def coverage(problem: Problem, x_true, widths: dict, n_rep=50, offset=3.0, seed=0, verbose=False):
    """Synthetic coverage test of the 1σ ``widths`` (log scale, e.g. sample sds).

    Each replica adds Gaussian noise with the real error bars to the model at
    ``x_true``, then refits from a start ``offset`` widths away from the truth in
    every physics parameter (random signs).  Uses the widths for every replica,
    which assumes the posterior width does not change between them.
    """
    clean = [problem.model_curve(x_true, k) for k in range(len(problem.datasets))]
    names = problem.physics_names
    rng = np.random.default_rng(seed)
    dev = []
    for r in range(n_rep):
        sims = [
            dataclasses.replace(d, y=c + rng.normal(size=len(c)) * d.err)
            for d, c in zip(problem.datasets, clean, strict=True)
        ]
        rep = Problem(problem.spec, sims)
        x0 = np.array(x_true, float)
        for n, s in zip(names, rng.choice([-1.0, 1.0], size=len(names)), strict=True):
            x0[rep.idx[n]] += offset * s * widths[n]
        _, x = optimise(rep, np.clip(x0, rep.lo + 1e-6, rep.hi - 1e-6))
        dev.append([x[rep.idx[n]] - x_true[problem.idx[n]] for n in names])
        if verbose:
            shifts = ", ".join(f"{n[2:]} {d * 100:+.1f}%" for n, d in zip(names, dev[-1], strict=True))
            print(f"  replica {r + 1}/{n_rep}: {shifts}", flush=True)
    return Coverage(names, np.array(dev), dict(widths))
