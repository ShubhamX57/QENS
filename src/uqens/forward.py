"""Forward model and exact nuisance solver (numba kernels).

``spectrum`` evaluates translation × rotation convolved with the resolution at
every (Q, ω) point, with detailed balance.  ``chi2_nnls`` then solves, exactly
and per Q-bin, for the three non-negative nuisance amplitudes (scale, a
resolution-shaped elastic term such as can scattering, and a flat background),
so the samplers only ever see the physical and resolution parameters.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .constants import HBAR
from .lineshapes import res_line

FICKIAN, JUMP = 0, 1


@njit(cache=True, parallel=True)
def spectrum(omega, dw, q, W, lch, mch, D, tau0, transl, Dt, Ds, sigma, gR, e0, kT, shape, el, f2=0.0, d2=0.0, s2=1.0):
    """Fill ``shape`` (model line shape) and ``el`` (resolution-shaped elastic term).

    omega: energy transfer, meV, energy loss positive; dw: bin widths; q: Å⁻¹.
    W, lch, mch: rotational channel table (``rotation.channel_weights``).
    D: Å²/ps; tau0: ps (jump model only); transl: FICKIAN or JUMP (Singwi–Sjölander).
    Dt, Ds: tumbling and spinning rates, ps⁻¹.  sigma: Gaussian resolution sd, meV;
    gR: Lorentzian part of the resolution (HWHM); e0: elastic-line offset; kT: meV.
    """
    n = omega.shape[0]
    nch = lch.shape[0]
    for i in prange(n):
        x = omega[i] - e0
        q2 = q[i] * q[i]
        if transl == 0:
            gt = HBAR * D * q2
        else:
            gt = HBAR * D * q2 / (1.0 + D * q2 * tau0)
        s = 0.0
        for c in range(nch):
            wg = W[i, c]
            if wg < 1e-9:
                continue
            l = lch[c]
            m = mch[c]
            g = HBAR * (l * (l + 1) * Dt + m * m * (Ds - Dt))
            s += wg * res_line(x, dw[i], sigma, gR + gt + g, f2, d2, s2)
        shape[i] = s * math.exp(0.5 * omega[i] / kT)  # detailed balance
        el[i] = res_line(x, dw[i], sigma, gR, f2, d2, s2)


@njit(cache=True)
def _solve_subset(G, r, mask):
    """Least squares on the columns selected by ``mask``; (ok, amplitudes, χ² reduction)."""
    idx = np.empty(3, np.int64)
    k = 0
    for j in range(3):
        if (mask >> j) & 1:
            idx[k] = j
            k += 1
    a = np.zeros(3)
    if k == 1:
        g = G[idx[0], idx[0]]
        if g <= 0:
            return False, a, 0.0
        a[idx[0]] = r[idx[0]] / g
    elif k == 2:
        i0, i1 = idx[0], idx[1]
        g00, g01, g11 = G[i0, i0], G[i0, i1], G[i1, i1]
        det = g00 * g11 - g01 * g01
        if det <= 1e-300 * max(1.0, abs(g00 * g11)):
            return False, a, 0.0
        a[i0] = (r[i0] * g11 - r[i1] * g01) / det
        a[i1] = (r[i1] * g00 - r[i0] * g01) / det
    else:
        M = G.copy()
        det = (
            M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
            - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
            + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
        )
        if abs(det) <= 1e-300:
            return False, a, 0.0
        sol = np.linalg.solve(M, r)
        a[0], a[1], a[2] = sol[0], sol[1], sol[2]
    for j in range(3):
        if a[j] < 0:
            return False, a, 0.0
    red = 0.0
    for j in range(3):
        red += a[j] * r[j]
    return True, a, red


@njit(cache=True)
def chi2_nnls(y, err, shape, el, bin_id, n_bins, use_el, amps):
    """χ² with the per-bin amplitude, elastic and background terms (all ≥ 0) solved
    exactly by enumerating the active sets of the 3-column problem.

    Returns the total χ² and writes the amplitudes into ``amps`` (n_bins × 3).
    """
    Gs = np.zeros((n_bins, 3, 3))
    rs = np.zeros((n_bins, 3))
    syy = np.zeros(n_bins)
    for i in range(y.shape[0]):
        b = bin_id[i]
        wt = 1.0 / (err[i] * err[i])
        c0, c1, c2 = shape[i], el[i], 1.0
        Gs[b, 0, 0] += wt * c0 * c0
        Gs[b, 0, 1] += wt * c0 * c1
        Gs[b, 0, 2] += wt * c0 * c2
        Gs[b, 1, 1] += wt * c1 * c1
        Gs[b, 1, 2] += wt * c1 * c2
        Gs[b, 2, 2] += wt * c2 * c2
        rs[b, 0] += wt * c0 * y[i]
        rs[b, 1] += wt * c1 * y[i]
        rs[b, 2] += wt * c2 * y[i]
        syy[b] += wt * y[i] * y[i]
    total = 0.0
    for b in range(n_bins):
        G = Gs[b]
        G[1, 0] = G[0, 1]
        G[2, 0] = G[0, 2]
        G[2, 1] = G[1, 2]
        best = syy[b]
        amps[b, 0] = 0.0
        amps[b, 1] = 0.0
        amps[b, 2] = 0.0
        for mask in range(1, 8):
            if (not use_el) and ((mask >> 1) & 1):
                continue
            ok, a, red = _solve_subset(G, rs[b], mask)
            if ok:
                c = syy[b] - red
                if c < best:
                    best = c
                    amps[b, 0] = a[0]
                    amps[b, 1] = a[1]
                    amps[b, 2] = a[2]
        total += best
    return total
