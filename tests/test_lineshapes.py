"""Analytic resolution convolution."""

import numpy as np
import pytest
from scipy.special import voigt_profile

from uqens.lineshapes import res_line, voigt, voigt_bin

CASES = [(0.02, 0.0), (0.02, 0.005), (0.02, 0.02), (0.02, 0.15), (0.02, 0.5)]  # includes gamma > 10 sigma


@pytest.mark.parametrize(("sigma", "gamma"), CASES)
def test_voigt_matches_scipy(sigma, gamma):
    x = np.linspace(-1.0, 1.0, 2001)
    ours = np.array([voigt(v, sigma, gamma) for v in x])
    ref = voigt_profile(x, sigma, gamma)
    assert np.max(np.abs(ours - ref)) < 1e-3 * ref.max()


def test_voigt_is_area_normalised():
    x = np.linspace(-200.0, 200.0, 400001)
    v = np.array([voigt(t, 0.05, 0.03) for t in x])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 has only trapz
    assert abs(trapezoid(v, x) - 1.0) < 2e-4


def test_bin_average_matches_numerical_average():
    sigma, gamma, dw = 0.01, 0.004, 0.02
    for x0 in (0.0, 0.013, 0.05):
        fine = np.linspace(x0 - dw / 2, x0 + dw / 2, 4001)
        ref = np.mean([voigt(t, sigma, gamma) for t in fine])
        assert abs(voigt_bin(x0, dw, sigma, gamma) - ref) < 0.02 * ref


def test_symmetric_resolution_is_centred():
    """The convolution must not shift the line (the v0.1 FFT path did)."""
    x = np.linspace(-1.5, 1.5, 3001)
    y = np.array([res_line(t, 0.001, 0.04, 0.03, 0.0, 0.0, 1.0) for t in x])
    assert abs(np.sum(x * y) / np.sum(y)) < 1e-6
