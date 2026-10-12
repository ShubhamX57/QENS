"""Slow tests: real fits.  Run with ``pytest -m slow`` (several minutes)."""

import numpy as np
import pytest

from uqens import LADDER, Problem, fit, validate

from .test_model import BENCHMARK_NEGLOGPOST, BENCHMARK_X

pytestmark = pytest.mark.slow


def test_default_starts_find_the_benchmark(benzene):
    """From generic starting points the fit must reach the benchmark optimum."""
    p = Problem(LADDER["jump+iso"], benzene)
    best = fit(p, n_random=2)
    assert best.neglogpost == pytest.approx(BENCHMARK_NEGLOGPOST, abs=0.5)
    assert best.physics()["D"] == pytest.approx(0.1724, rel=0.01)


def test_synthetic_recovery(benzene):
    """Data simulated from the benchmark are fitted back within the posterior width."""
    p = Problem(LADDER["jump+iso"], benzene)
    widths = {"lnD": 0.029, "lntau0": 0.132, "lnDr": 0.0093}  # sampled posterior sds (log scale)
    cov = validate.coverage(p, np.array(BENCHMARK_X), widths, n_rep=3, seed=11)
    assert np.all(np.abs(cov.deviations) < 3.5 * np.array([widths[n] for n in cov.names]))
