"""The report plots draw without errors (needs matplotlib)."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from uqens import LADDER, Problem, plot, validate  # noqa: E402

from .test_model import BENCHMARK_X  # noqa: E402


@pytest.fixture(scope="module")
def best(benzene):
    return Problem(LADDER["jump+iso"], benzene), np.array(BENCHMARK_X)


def test_spectra(best):
    fig = plot.spectra(*best, energy=3.60, q=[0.6, 1.2, 1.8])
    assert len(fig.axes) == 6


def test_misfit_is_scaled_noise(best):
    problem, x = best
    for _, rows in plot.misfit(problem, x):
        z = np.concatenate([zz for _, zz in rows])
        assert 0.5 < z.std() < 2.5


def test_misfit_map(best):
    fig = plot.misfit_map({"jump+iso": best})
    assert len(fig.axes) == 4 + 1  # four energies and the colour bar


def test_holdout(best, benzene):
    problem, x = best
    one = Problem(problem.spec, [benzene[0]])
    rows = [{"ei": benzene[0].ei, "problem": one, "x": validate.transfer(problem, x, one)}]
    fig = plot.holdout(rows)
    assert len(fig.axes) == 6


def test_timescales(best):
    fig = plot.timescales(*best)
    assert len(fig.axes) == 1
