"""
Normalised lineshape primitives.

"""
from __future__ import annotations

import numpy as np

__all__ = ["lorentz", "gnorm", "lorentz_sum"]

#: smallest Γ allowed before clipping — below this Lorentzian becomes a δ-fn
GAMMA_FLOOR = 1e-5


def lorentz(omega: np.ndarray, gamma: float) -> np.ndarray:
    """Area-normalised Lorentzian, half-width-at-half-maximum ``gamma``.

    .. math::
        L(\\omega; \\Gamma) = \\frac{1}{\\pi}
            \\frac{\\Gamma}{\\omega^2 + \\Gamma^2}

    Parameters
    ----------
    omega : array
        Energy-transfer grid (any unit; HWHM ``gamma`` must match).
    gamma : float
        HWHM. Will be clipped to ``GAMMA_FLOOR`` if smaller, to avoid
        producing a δ-function and breaking later FFT-convolutions.
    """
    g = max(float(gamma), GAMMA_FLOOR)
    w = np.asarray(omega, dtype=float)
    return (1.0 / np.pi) * g / (w * w + g * g)


def gnorm(omega: np.ndarray, sigma: float) -> np.ndarray:
    """Area-normalised Gaussian with standard deviation ``sigma``.

    .. math::
        G(\\omega; \\sigma) = \\frac{1}{\\sigma\\sqrt{2\\pi}}
            \\exp\\!\\left( -\\frac{\\omega^2}{2\\sigma^2} \\right)

    For FWHM use ``sigma = FWHM / 2.355``.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    w = np.asarray(omega, dtype=float)
    return np.exp(-0.5 * (w / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def lorentz_sum(
    omega: np.ndarray,
    weights: np.ndarray,
    gammas: np.ndarray,
) -> np.ndarray:
    """Weighted sum of Lorentzians: ``Σ w_i L(ω; Γ_i)``.

    Vectorised — useful for evaluating multi-channel rotational models on a
    Q-grid efficiently.

    Parameters
    ----------
    omega : array, shape (N,)
    weights : array, shape (M,)
    gammas : array, shape (M,)

    Returns
    -------
    array, shape (N,)
    """
    weights = np.asarray(weights, dtype=float)
    gammas = np.maximum(np.asarray(gammas, dtype=float), GAMMA_FLOOR)
    w = np.asarray(omega, dtype=float)[:, None]
    L = (1.0 / np.pi) * gammas / (w * w + gammas * gammas)
    return L @ weights
