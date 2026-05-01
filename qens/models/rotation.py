"""
Rotational structure-factor models for axially symmetric molecules.

"""
from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn

from ..constants import HBAR_MEV_PS

__all__ = [
    "rot_widths_isotropic",
    "rot_widths_anisotropic",
    "bessel_weights",
    "DEFAULT_RADIUS",
]



#: default radius of gyration in Å. Override for non-benzene molecules.
DEFAULT_RADIUS: float = 2.48




def rot_widths_isotropic(d_r: float) -> tuple[float, float]:
    """HWHMs for the isotropic axial-rotor model.

    Returns "(Γ_1, Γ_2)" in meV, the widths of the
    l = 1 and l = 2 Lorentzian channels:

    .. math::
        \\Gamma_1 = 2 \\hbar D_r,\\quad
        \\Gamma_2 = 6 \\hbar D_r

    Parameters
    ----------
    d_r : float
        Rotational diffusion coefficient in ps⁻¹.
    """
    if d_r <= 0:
        raise ValueError("d_r must be > 0")
    return HBAR_MEV_PS * 2.0 * d_r, HBAR_MEV_PS * 6.0 * d_r





def rot_widths_anisotropic(d_t: float, d_s: float) -> tuple[float, float, float]:
    """HWHMs for the anisotropic axial-rotor model (oblate top, e.g. benzene).

    Returns (Γ_1, Γ_2, Γ_3)" in meV

    Parameters
    ----------
    d_t : float
        Tumbling rotational diffusion coefficient in ps⁻¹.
    d_s : float
        Spinning rotational diffusion coefficient in ps⁻¹.
    """
    if d_t <= 0 or d_s <= 0:
        raise ValueError("d_t and d_s must be > 0")
    return (
        HBAR_MEV_PS * (d_s + d_t),
        HBAR_MEV_PS * 6.0 * d_t,
        HBAR_MEV_PS * (4.0 * d_s + 2.0 * d_t),
    )






def bessel_weights(q: float, R: float = DEFAULT_RADIUS) -> tuple[float, float, float]:
    """Spherical-Bessel weights "(j₀², j₁², j₂²)" at this Q.

    These weight the l = 0, 1, 2 channels of the rotational structure factor.
    "j₀²(QR)" is the elastic-rotational weight (the EISF in the rotation-only
    limit). At low Q, j₀² → 1; at high Q it falls off and j₂² dominates.

    Parameters
    ----------
    q : float
        Momentum transfer in Å⁻¹.
    R : float
        Radius of gyration in Å. Defaults to 2.48 Å (benzene).

    Returns
    -------
    (j0², j1², j2²)
    """
    x = q * R
    return (spherical_jn(0, x) ** 2,
            spherical_jn(1, x) ** 2,
            spherical_jn(2, x) ** 2)
