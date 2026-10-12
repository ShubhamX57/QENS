"""Resolution-convolved line shapes.

A Lorentzian of half-width gamma convolved with a Gaussian of standard deviation
sigma is a Voigt profile, so convolution with the resolution is analytic: no FFT,
no kernel grid, no centring errors.  Every function is a numba kernel so the
forward model can call it per point.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

SQ2PI = math.sqrt(2.0 * math.pi)
SQ2 = math.sqrt(2.0)


@njit(cache=True, inline="always")
def _w4_real(x, y):
    """Re w(z), z = x + i y (y >= 0); Humlicek (1982) W4, relative accuracy ~1e-4."""
    t = complex(y, -x)
    s = abs(x) + y
    if s >= 15.0:
        w = t * 0.5641896 / (0.5 + t * t)
    elif s >= 5.5:
        u = t * t
        w = t * (1.410474 + u * 0.5641896) / (0.75 + u * (3.0 + u))
    elif y >= 0.195 * abs(x) - 0.176:
        w = (16.4955 + t * (20.20933 + t * (11.96482 + t * (3.778987 + t * 0.5642236)))) / (
            16.4955 + t * (38.82363 + t * (39.27121 + t * (21.69274 + t * (6.699398 + t))))
        )
    else:
        u = t * t
        w = np.exp(u) - t * (
            36183.31 - u * (3321.9905 - u * (1540.787 - u * (219.0313 - u * (35.76683 - u * (1.320522 - u * 0.56419)))))
        ) / (
            32066.6
            - u * (24322.84 - u * (9022.228 - u * (2186.181 - u * (364.2191 - u * (61.57037 - u * (1.841439 - u))))))
        )
    return w.real


@njit(cache=True, inline="always")
def voigt(x, sigma, gamma):
    """Area-normalised Voigt profile: Gaussian sd ``sigma`` ⊗ Lorentzian HWHM ``gamma``."""
    if gamma <= 1e-12:
        return math.exp(-0.5 * (x / sigma) ** 2) / (sigma * SQ2PI)
    if gamma > 10.0 * sigma:
        # Lorentzian plus its second-order Gaussian correction
        x2g2 = x * x + gamma * gamma
        lor = gamma / (math.pi * x2g2)
        lor2 = 2.0 * gamma * (3.0 * x * x - gamma * gamma) / (math.pi * x2g2 * x2g2 * x2g2)
        return lor + 0.5 * sigma * sigma * lor2
    return _w4_real(x / (sigma * SQ2), gamma / (sigma * SQ2)) / (sigma * SQ2PI)


@njit(cache=True, inline="always")
def voigt_bin(x, dw, sigma, gamma):
    """Voigt averaged over an energy bin of width ``dw`` (2-point Gauss-Legendre when
    the bin is not small compared with the line)."""
    if dw > 0.3 * (sigma + gamma):
        h = dw / (2.0 * math.sqrt(3.0))
        return 0.5 * (voigt(x - h, sigma, gamma) + voigt(x + h, sigma, gamma))
    return voigt(x, sigma, gamma)


@njit(cache=True, inline="always")
def res_line(x, dw, sigma, gamma, f2, d2, s2):
    """A Lorentzian of HWHM ``gamma`` convolved with the resolution.

    The resolution is a main Voigt (Gaussian sd ``sigma``) plus an optional second
    component carrying fraction ``f2``, shifted by ``d2*sigma`` and ``s2`` times
    wider: a simple description of the asymmetric tail of a chopper spectrometer.
    """
    v = voigt_bin(x, dw, sigma, gamma)
    if f2 > 0.0:
        v = (1.0 - f2) * v + f2 * voigt_bin(x - d2 * sigma, dw, s2 * sigma, gamma)
    return v
