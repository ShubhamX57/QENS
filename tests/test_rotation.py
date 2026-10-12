"""Rotational channels and the sum rule."""

import math

import numpy as np
from scipy.special import spherical_jn

from uqens.rotation import channel_weights, clm_table, lmax_for

R = 2.48


def test_benzene_l2_weights():
    """For an atom in the ring plane (θ₀ = 90°), l = 2 splits 1/4 (m = 0) and 3/4 (|m| = 2)."""
    ls, ms, cs = clm_table(2)
    w = {(int(a), int(b)): float(c) for a, b, c in zip(ls, ms, cs, strict=True)}
    assert abs(w[(2, 0)] - 0.25) < 1e-12 and abs(w[(2, 2)] - 0.75) < 1e-12
    assert (2, 1) not in w


def test_m_weights_sum_to_one():
    for theta0 in (math.pi / 2, 0.7):
        ls, ms, cs = clm_table(12, theta0)
        for lv in range(13):
            assert abs(cs[ls == lv].sum() - 1.0) < 1e-10


def test_sum_rule_is_met():
    q = np.linspace(0.3, 2.6, 50)
    lmax = lmax_for(q.max(), R)
    W, _, _ = channel_weights(q, "iso", R)
    assert np.all(W.sum(1) > 1 - 5e-4)
    assert lmax >= 8


def test_anisotropic_channels_carry_the_same_total():
    q = np.linspace(0.3, 2.6, 20)
    Wi, _, _ = channel_weights(q, "iso", R)
    Wa, _, _ = channel_weights(q, "aniso", R)
    assert np.allclose(Wi.sum(1), Wa.sum(1), atol=1e-10)


def test_stopping_at_l2_loses_most_of_the_signal_at_high_q():
    """Why the expansion is not truncated: at Q = 2.5 Å⁻¹, l ≤ 2 keeps about a tenth."""
    x = 2.5 * R
    kept = sum((2 * lv + 1) * spherical_jn(lv, x) ** 2 for lv in range(3))
    assert kept < 0.2
