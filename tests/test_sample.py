"""The sampler returns its diagnostics."""

import numpy as np

from uqens import LADDER, Problem, sample

from .test_model import BENCHMARK_X


def test_sampler_returns_diagnostics(benzene):
    p = Problem(LADDER["jump+iso"], benzene)
    post = sample(p, np.array(BENCHMARK_X), n_walkers=32, n_steps=40)
    assert post.chain.shape == (40, 32, len(p.names))
    assert post.samples.shape == ((40 - post.burn) * 32, len(p.names))
    lo, hi = post.interval("lnD")
    assert lo < post.median("lnD") < hi
    assert 0.0 < post.acceptance < 1.0
    assert post.tau.shape == (len(p.names),)
    assert not post.converged  # 40 steps is far too short, and the result must say so
    assert "converged: False" in post.summary()
