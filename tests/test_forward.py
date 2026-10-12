"""Forward model and the exact nuisance solver."""

import numpy as np
import pytest
from scipy.optimize import nnls

from uqens.constants import HBAR, KB
from uqens.forward import FICKIAN, JUMP, chi2_nnls, spectrum
from uqens.lineshapes import voigt
from uqens.rotation import channel_weights


def run(omega, q, D=0.17, tau0=0.0, transl=FICKIAN, rotation="none", Dr=0.0, sigma=0.01, T=290.0):
    W, lch, mch = channel_weights(q, rotation)
    shape, el = np.empty(len(omega)), np.empty(len(omega))
    spectrum(
        omega, np.full(len(omega), 1e-4), q, W, lch, mch, D, tau0, transl, Dr, Dr, sigma, 0.0, 0.0, KB * T, shape, el
    )
    return shape, el


def test_detailed_balance():
    omega = np.linspace(-1.0, 1.0, 201)
    q = np.full(len(omega), 1.0)
    shape, _ = run(omega, q)
    kT = KB * 290.0
    ratio = shape[omega > 0] / shape[omega < 0][::-1]
    assert np.allclose(ratio, np.exp(omega[omega > 0] / kT), rtol=1e-6)


@pytest.mark.parametrize(("transl", "tau0"), [(FICKIAN, 0.0), (JUMP, 1.0)])
def test_translation_width(transl, tau0):
    """Without rotation the line is one Lorentzian of HWHM ħDQ² (Fickian) or
    ħDQ²/(1 + DQ²τ₀) (jump diffusion), convolved with the resolution."""
    D, q0, sigma = 0.17, 1.5, 0.01
    omega = np.linspace(-1.0, 1.0, 401)
    shape, _ = run(omega, np.full(len(omega), q0), D=D, tau0=tau0, transl=transl, sigma=sigma)
    gamma = HBAR * D * q0**2 / (1.0 + D * q0**2 * tau0)
    expected = np.array([voigt(w, sigma, gamma) for w in omega]) * np.exp(0.5 * omega / (KB * 290.0))
    assert np.allclose(shape, expected, rtol=1e-9)


def test_nnls_matches_scipy():
    rng = np.random.default_rng(1)
    n_bins, per = 6, 40
    bin_id = np.repeat(np.arange(n_bins), per)
    shape = rng.uniform(0.1, 2.0, n_bins * per)
    el = np.exp(-(np.tile(np.linspace(-3, 3, per), n_bins) ** 2))
    err = rng.uniform(0.05, 0.2, n_bins * per)
    a_true = rng.uniform(0, 1, (n_bins, 3)) * (rng.uniform(size=(n_bins, 3)) > 0.3)
    y = a_true[bin_id, 0] * shape + a_true[bin_id, 1] * el + a_true[bin_id, 2] + rng.normal(0, err)
    amps = np.zeros((n_bins, 3))
    total = chi2_nnls(y, err, shape, el, bin_id, n_bins, True, amps)
    ref = 0.0
    for b in range(n_bins):
        m = bin_id == b
        A = np.stack([shape[m], el[m], np.ones(m.sum())], 1) / err[m, None]
        sol, res = nnls(A, y[m] / err[m])
        ref += res**2
        assert np.allclose(amps[b], sol, atol=1e-8)
    assert abs(total - ref) < 1e-8 * ref


def test_measured_kernel_reproduces_the_analytic_convolution():
    """A Gaussian 'measured' kernel must give the same line as the analytic Voigt."""
    from uqens.forward import spectrum_kernel

    step, sigma, D, q0 = 0.002, 0.03, 0.17, 1.2
    grid = np.arange(-1.5, 1.5 + step / 2, step)
    k_val = np.exp(-0.5 * (grid / sigma) ** 2)
    k_val /= k_val.sum()
    k_off = np.array([0, len(grid)], np.int64)
    k_dw = np.full(len(grid), step)
    omega = grid[np.abs(grid) <= 1.0]
    q = np.full(len(omega), q0)
    W, lch, mch = channel_weights(q, "none")
    shape, el = np.empty(len(omega)), np.empty(len(omega))
    spectrum_kernel(
        omega,
        q,
        np.zeros(len(omega), np.int64),
        W,
        lch,
        mch,
        D,
        0.0,
        FICKIAN,
        0.0,
        0.0,
        np.inf,
        k_off,
        grid,
        k_val,
        k_dw,
        shape,
        el,
    )
    gamma = HBAR * D * q0**2
    expected = np.array([voigt(w, sigma, gamma) for w in omega])
    assert np.max(np.abs(shape - expected)) < 1e-3 * expected.max()
    assert np.allclose(el, np.interp(omega, grid, k_val / step))
