"""Parameters, priors and the benzene benchmark likelihood."""

import math

import numpy as np
import pytest

from uqens import LADDER, Problem

# Best fit of jump diffusion + isotropic rotation to the four benzene energies, found
# by the prototype that produced the numbers in the proposal (D = 0.172 Å²/ps).
BENCHMARK_X = [
    -1.7577586842012738,
    -0.05027941675944619,
    -1.7950369787405251,
    -4.907074865364085,
    2.300449297629199e-05,
    -4.382951759987795,
    -0.0015831478750753774,
    -3.3967932962129677,
    -0.0061599784690922615,
    -1.907642796627984,
    0.032290823186889764,
    0.40422602918698014,
    0.27477168053377593,
    -1.4619589404183215,
    1.387864092289836,
]
BENCHMARK_NEGLOGPOST = 6088.380953425189


def test_parameter_names(benzene):
    p = Problem(LADDER["jump+iso"], benzene)
    assert p.names[:3] == ["lnD", "lntau0", "lnDr"]
    assert p.physics_names == ["lnD", "lntau0", "lnDr"]
    q = Problem(LADDER["fickian+aniso"], benzene)
    assert q.physics_names == ["lnD", "lnDt", "lnDs"]
    assert len(q.names) == len(p.names)


def test_prior_bounds(benzene):
    p = Problem(LADDER["jump+iso"], benzene)
    x = p.centre()
    assert np.isfinite(p.logpost(x))
    x[p.idx["lnD"]] = math.log(5.0)  # outside the prior
    assert p.logpost(x) == -np.inf


def test_benchmark_likelihood_is_unchanged(benzene):
    """Guards the whole chain (reader, binning, error audit, forward model, solver)
    against silent changes: the benchmark best fit must give the same posterior."""
    p = Problem(LADDER["jump+iso"], benzene)
    x = np.array(BENCHMARK_X)
    assert -p.logpost(x) == pytest.approx(BENCHMARK_NEGLOGPOST, rel=1e-9)
    phys = p.physics(x)
    assert phys["D"] == pytest.approx(0.1724, abs=5e-4)
    assert phys["tau0"] == pytest.approx(0.951, abs=5e-3)
    assert phys["Dt"] == pytest.approx(0.1661, abs=5e-4)


def test_model_curve_follows_the_data(benzene):
    p = Problem(LADDER["jump+iso"], benzene)
    x = np.array(BENCHMARK_X)
    for k, d in enumerate(benzene):
        resid = (d.y - p.model_curve(x, k)) / d.err
        assert abs(resid.mean()) < 0.2 and resid.std() < 1.5
