"""
Translational-diffusion HWHM models.


"""
from __future__ import annotations

import numpy as np

from ..constants import HBAR_MEV_PS

__all__ = ["fickian_hwhm", "ce_hwhm", "ss_hwhm"]


def fickian_hwhm(q: np.ndarray, d: float) -> np.ndarray:
    """Fickian (continuum) diffusion: Γ = ℏ D Q².

    Parameters
    ----------
    q : array
        Momentum transfer in Å⁻¹.
    d : float
        Self-diffusion coefficient in Å²/ps.

    Returns
    -------
    Γ in meV.
    """
    return HBAR_MEV_PS * d * np.asarray(q, dtype=float) ** 2


def ce_hwhm(q: np.ndarray, d: float, ell: float) -> np.ndarray:
    """Chudley-Elliott jump-diffusion HWHM in meV.


    Parameters
    ----------
    q : array
    d : float
        Self-diffusion coefficient in Å²/ps.
    ell : float
        Mean jump length in Å.

    Returns
    -------
    Γ in meV.
    """
    d_arr = np.asarray(d, dtype=float)
    ell_arr = np.abs(np.asarray(ell, dtype=float))
    if np.any(d_arr <= 0):
        raise ValueError("d must be > 0")
    if np.any(ell_arr <= 0):
        raise ValueError("ell must be > 0")
    tau = ell_arr ** 2 / (6 * d_arr)
    return (HBAR_MEV_PS / tau) * (1 - np.sinc(np.asarray(q) * ell_arr / np.pi))


def ss_hwhm(q: np.ndarray, d: float, tau_s: float) -> np.ndarray:
    """Singwi-Sjölander HWHM: Γ = ℏ D Q² / (1 + D Q² τ_s).

    Parameters
    ----------
    q : array
    d : float
        Self-diffusion coefficient in Å²/ps.
    tau_s : float
        Residence time in ps.

    Returns
    -------
    Γ in meV.
    """
    q = np.asarray(q, dtype=float)
    if d <= 0 or tau_s <= 0:
        raise ValueError("d and tau_s must be > 0")
    return HBAR_MEV_PS * d * q ** 2 / (1 + d * q ** 2 * tau_s)
