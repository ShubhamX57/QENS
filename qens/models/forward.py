"""
Full forward model for QENS line-shape inference.


"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal import fftconvolve

from ..constants import HBAR_MEV_PS
from .lineshapes  import lorentz, gnorm, GAMMA_FLOOR
from .rotation    import (
    rot_widths_isotropic, rot_widths_anisotropic,
    bessel_weights, DEFAULT_RADIUS,
)

__all__ = ["predict_sqw", "ForwardModel"]





# helper: resolve "sigma_res" — either a scalar Gaussian σ or a measured array

def _make_resolution_kernel(omega: np.ndarray, sigma_res) -> np.ndarray:
    """Return an area-normalised resolution kernel on "omega"."""
    omega = np.asarray(omega, dtype=float)
    dt = omega[1] - omega[0]
    if np.ndim(sigma_res) == 0:
        kernel = gnorm(omega, float(sigma_res))
    else:
        kernel = np.asarray(sigma_res, dtype=float).copy()
        kernel = np.where(np.isfinite(kernel) & (kernel > 0), kernel, 0.0)
        if kernel.shape != omega.shape:
            raise ValueError(
                f"resolution kernel shape {kernel.shape} ≠ omega shape "
                f"{omega.shape}"
            )
        # re-centre on its argmax so convolution doesn't shift the spectrum
        peak_idx = int(np.argmax(kernel))
        mid = len(kernel) // 2
        if peak_idx != mid:
            kernel = np.roll(kernel, mid - peak_idx)
    area = kernel.sum() * dt
    if area <= 0:
        raise ValueError("resolution kernel has zero / negative area")
    return kernel / area






# core forward-model evaluator

def predict_sqw(
    omega: np.ndarray,
    q: float,
    *,
    d_translation: float,
    u2: float,
    rotation: tuple[float, ...],
    rotation_model: str = "anisotropic",
    sigma_res,
    radius: float = DEFAULT_RADIUS,
) -> np.ndarray:
    """Predict the resolution-convolved S_inc(Q, ω) for one Q-bin.

    Parameters
    ----------
    omega : array, shape (N,)
        Energy grid in meV. Must be uniformly spaced.
    q : float
        Momentum transfer in Å⁻¹.
    d_translation : float
        Translational self-diffusion coefficient D* in Å²/ps.
    u2 : float
        Mean-square displacement ⟨u²⟩ in Å² (Debye-Waller factor).
    rotation : tuple
        "(D_r,)" for isotropic, "(D_t, D_s)" for anisotropic.
        Both in ps⁻¹.
    rotation_model : str
        ""isotropic"" or ""anisotropic"".
    sigma_res : float or array
        Resolution: Gaussian sigma in meV (scalar) or measured kernel
        (1-d array same length as "omega").
    radius : float
        Radius of gyration in Å. Default 2.48 (benzene).

    Returns
    -------
    array, shape (N,)
        Predicted shape, **unscaled** — the inference layer fits an overall
        amplitude and a flat background per Q-bin via NNLS, so this function
        only encodes the physics.
    """
    omega = np.asarray(omega, dtype=float)

    # translational HWHM (Fickian) — common to every rotational channel
    gamma_t = max(HBAR_MEV_PS * d_translation * q * q, GAMMA_FLOOR)

    # spherical-Bessel weights
    j0_sq, j1_sq, j2_sq = bessel_weights(q, radius)

    if rotation_model == "isotropic":
        g1, g2 = rot_widths_isotropic(rotation[0])
        s_unc = (j0_sq        * lorentz(omega, gamma_t)
                 + 3 * j1_sq  * lorentz(omega, gamma_t + g1)
                 + 5 * j2_sq  * lorentz(omega, gamma_t + g2))
    elif rotation_model == "anisotropic":
        g1, g2, g3 = rot_widths_anisotropic(rotation[0], rotation[1])
        s_unc = (j0_sq            * lorentz(omega, gamma_t)
                 + 3 * j1_sq      * lorentz(omega, gamma_t + g1)
                 + 5 * j2_sq * (0.25 * lorentz(omega, gamma_t + g2)
                                + 0.75 * lorentz(omega, gamma_t + g3)))
    elif rotation_model == "none":
        # purely translational — single Lorentzian, no rotation
        s_unc = lorentz(omega, gamma_t)
    else:
        raise ValueError(
            f"unknown rotation_model {rotation_model!r}; "
            f"expected 'none', 'isotropic' or 'anisotropic'"
        )

    # Debye-Waller factor
    s_unc *= np.exp(-q * q * u2 / 3.0)

    # resolution convolution
    kernel = _make_resolution_kernel(omega, sigma_res)
    dt = omega[1] - omega[0]
    return fftconvolve(s_unc, kernel, mode="same") * dt






# ForwardModel: bundle (predict_fn, params, prior_box) for the inference layer

@dataclass
class ForwardModel:
    """Encapsulates a forward-model definition for the inference layer.

    A ForwardModel knows:

         its name (for logging)
         the parameter names (in order)
         the prior box (uniform, "[lo, hi]" per parameter)
         how to predict S(Q,ω) for one Q-bin given a parameter vector

    Custom forward models register here (see :mod:`qens.models.registry`).

    Attributes
    ----------
    name : str
    param_names : tuple[str, ...]
    prior_lo, prior_hi : tuple[float, ...]
        Uniform-prior bounds, same length as "param_names".
    predict : callable
        "predict(omega, q, params, sigma_res, **extras) -> array".
    extras : dict
        Extra keyword arguments forwarded to "predict" (e.g. "radius").
    """

    name: str
    param_names: tuple[str, ...]
    prior_lo: tuple[float, ...]
    prior_hi: tuple[float, ...]
    predict: Callable[..., np.ndarray]
    extras: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if len(self.param_names) != len(self.prior_lo) != len(self.prior_hi):
            raise ValueError(
                "param_names, prior_lo, prior_hi must all have same length"
            )
        if any(lo >= hi for lo, hi in zip(self.prior_lo, self.prior_hi)):
            raise ValueError("each prior_lo must be < prior_hi")
        if self.extras is None:
            self.extras = {}

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def in_prior(self, params: np.ndarray) -> bool:
        """True iff every parameter is inside its uniform-prior box."""
        return bool(
            np.all(np.asarray(params) > np.asarray(self.prior_lo))
            and np.all(np.asarray(params) < np.asarray(self.prior_hi))
        )

    def random_in_prior(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a single random parameter vector from the prior box."""
        return rng.uniform(self.prior_lo, self.prior_hi)

    def __repr__(self) -> str:
        bounds = ", ".join(
            f"{n}∈[{lo:g},{hi:g}]"
            for n, lo, hi in zip(self.param_names, self.prior_lo, self.prior_hi)
        )
        return f"ForwardModel({self.name!r}: {bounds})"
