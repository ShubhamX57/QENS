"""Rotational diffusion channels for a scatterer on a sphere of radius R.

The rotational part of S(Q, ω) is a sum of Lorentzians, one per channel ``(l, abs(m))``,
with weight (2l+1) j_l²(QR) c_lm and HWHM ħ[l(l+1) D_t + m² (D_s − D_t)]
(symmetric-top rotational diffusion; isotropic when D_t = D_s).  The sum rule
Σ_l (2l+1) j_l²(QR) = 1 says how many l are needed: at Q = 2.5 Å⁻¹ and R = 2.48 Å,
stopping at l = 2 keeps only 10% of the rotational intensity.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, lpmv, spherical_jn


def clm_table(lmax, theta0=math.pi / 2):
    """Weights of the ``(l, abs(m))`` channels for an atom at polar angle ``theta0`` from the
    symmetry axis: ``(4π/(2l+1)) Σ_{±m} |Y_lm|²``, which sums to 1 over m for each l."""
    ls, ms, cs = [], [], []
    x = math.cos(theta0)
    for l in range(lmax + 1):
        for m in range(0, l + 1):
            p = lpmv(m, l, x)
            norm = math.exp(gammaln(l - m + 1) - gammaln(l + m + 1))
            c = norm * p * p * (1.0 if m == 0 else 2.0)
            if c > 1e-12:
                ls.append(l)
                ms.append(m)
                cs.append(c)
    return np.array(ls, np.int64), np.array(ms, np.int64), np.array(cs)


def lmax_for(qmax, R, tol=5e-4):
    """Smallest l_max for which the sum rule is met to within ``tol`` at ``qmax``."""
    x = qmax * R
    for L in range(0, 60):
        s = sum((2 * l + 1) * spherical_jn(l, x) ** 2 for l in range(L + 1))
        if 1 - s < tol:
            return L
    return 60


def channel_weights(q, rotation, R=2.48, lmax=None, theta0=math.pi / 2):
    """Per-point channel weights.

    Returns ``(W, l, m)`` with ``W[i, c]`` the weight of channel c at point i, for
    ``rotation`` in {'none', 'iso', 'aniso'}.  ``lmax=None`` uses the sum rule.
    """
    q = np.asarray(q, float)
    if rotation == "none":
        return np.ones((len(q), 1)), np.array([0], np.int64), np.array([0], np.int64)
    if lmax is None:
        lmax = lmax_for(q.max(), R)
    if rotation == "iso":
        ls = np.arange(lmax + 1)
        W = np.stack([(2 * l + 1) * spherical_jn(l, q * R) ** 2 for l in ls], axis=1)
        return W, ls.astype(np.int64), np.zeros(len(ls), np.int64)
    if rotation != "aniso":
        raise ValueError(f"rotation must be 'none', 'iso' or 'aniso', not {rotation!r}")
    ls, ms, cs = clm_table(lmax, theta0)
    W = np.stack([(2 * l + 1) * spherical_jn(l, q * R) ** 2 * c for l, c in zip(ls, cs, strict=True)], axis=1)
    return W, ls, ms
