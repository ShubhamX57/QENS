"""Posterior sampling with diagnostics that are returned, not just printed.

On the benzene data, error bars from the curvature at the best fit were unstable
(1.2–7.6% for D between statistically identical synthetic datasets) and too small
on average, while sampled widths matched the real scatter.  So uncertainties come
from samples.  Check ``Posterior.converged`` before quoting them.
"""

from __future__ import annotations

from dataclasses import dataclass

import emcee
import numpy as np

from .model import Problem


@dataclass
class Posterior:
    names: list
    chain: np.ndarray  # steps × walkers × parameters, burn-in included
    burn: int
    acceptance: float
    tau: np.ndarray  # integrated autocorrelation time per parameter, in steps

    @property
    def samples(self):
        return self.chain[self.burn :].reshape(-1, self.chain.shape[2])

    def values(self, name):
        """Samples of one parameter, in natural units (log parameters exponentiated)."""
        v = self.samples[:, self.names.index(name)]
        return np.exp(v) if name.startswith("ln") else v

    def interval(self, name, level=0.95):
        lo, hi = np.percentile(self.values(name), [50 * (1 - level), 50 * (1 + level)])
        return float(lo), float(hi)

    def median(self, name):
        return float(np.median(self.values(name)))

    def rhat(self, name):
        """Gelman–Rubin R̂ treating walkers as chains (approximate for ensemble samplers)."""
        c = self.chain[self.burn :, :, self.names.index(name)]
        n = c.shape[0]
        within = c.var(0, ddof=1).mean()
        between = n * c.mean(0).var(ddof=1)
        return float(np.sqrt(((n - 1) / n * within + between / n) / within))

    @property
    def n_effective(self):
        """Approximate number of independent samples, per parameter."""
        kept = self.chain.shape[0] - self.burn
        return kept * self.chain.shape[1] / self.tau

    @property
    def converged(self):
        """Chain at least 50 τ long after burn-in and R̂ < 1.1 for every parameter."""
        kept = self.chain.shape[0] - self.burn
        return bool(np.all(kept > 50 * self.tau) and all(self.rhat(n) < 1.1 for n in self.names))

    def summary(self, names=None):
        names = names or [n for n in self.names if n.startswith("ln")]
        lines = [
            f"{self.chain.shape[1]} walkers × {self.chain.shape[0]} steps (burn-in {self.burn}), "
            f"acceptance {self.acceptance:.2f}, converged: {self.converged}"
        ]
        for n in names:
            lo, hi = self.interval(n)
            j = self.names.index(n)
            label = n[2:] if n.startswith("ln") else n
            lines.append(
                f"  {label:6s} {self.median(n):.4g}  95% [{lo:.4g}, {hi:.4g}]  "
                f"tau {self.tau[j]:.0f}  R-hat {self.rhat(n):.3f}"
            )
        return "\n".join(lines)


def sample(problem: Problem, x_best, n_walkers=36, n_steps=3000, burn=None, seed=0, progress=False):
    """Run emcee from a small ball around ``x_best`` (1% of each prior range).

    ``x_best`` should be the best fit; walkers spread out from there.  The default
    run is a reasonable first look; for numbers to publish, run until
    ``Posterior.converged`` is true (about 50 autocorrelation times).
    """
    rng = np.random.default_rng(seed)
    scale = 0.01 * (problem.hi - problem.lo)
    p0 = np.array(
        [
            np.clip(x_best + scale * rng.normal(size=len(x_best)), problem.lo + 1e-9, problem.hi - 1e-9)
            for _ in range(n_walkers)
        ]
    )
    sampler = emcee.EnsembleSampler(n_walkers, len(x_best), problem.logpost)
    sampler.run_mcmc(p0, n_steps, progress=progress)
    burn = n_steps // 3 if burn is None else burn
    tau = sampler.get_autocorr_time(discard=burn, quiet=True)
    return Posterior(list(problem.names), sampler.get_chain(), burn, float(sampler.acceptance_fraction.mean()), tau)
