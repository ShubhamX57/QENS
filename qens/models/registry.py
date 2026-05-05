"""
Forward-model registry.


"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.signal import fftconvolve

from ..constants  import HBAR_MEV_PS
from .forward     import ForwardModel, predict_sqw, _make_resolution_kernel
from .lineshapes  import lorentz, GAMMA_FLOOR

__all__ = ["register_model", "get_model", "available_models"]

_REGISTRY: dict[str, ForwardModel] = {}





# built in models — wrap predict_sqw with the right rotation_model

def _predict_translation_only(omega, q, params, sigma_res, **kw):
    d_star, u2 = params
    return predict_sqw(omega, q,
                       d_translation=d_star, u2=u2,
                       rotation=(), rotation_model="none",
                       sigma_res=sigma_res, **kw)


def _predict_isotropic(omega, q, params, sigma_res, **kw):
    d_star, u2, d_r = params
    return predict_sqw(omega, q,
                       d_translation=d_star, u2=u2,
                       rotation=(d_r,), rotation_model="isotropic",
                       sigma_res=sigma_res, **kw)


def _predict_anisotropic(omega, q, params, sigma_res, **kw):
    d_star, u2, d_t, d_s = params
    return predict_sqw(omega, q,
                       d_translation=d_star, u2=u2,
                       rotation=(d_t, d_s), rotation_model="anisotropic",
                       sigma_res=sigma_res, **kw)



# Sensible default prior boxes - users can override for their own systems
# by re-registering the same name with new priors.
_REGISTRY["translation_only"] = ForwardModel(name="translation_only",
                                             param_names=("D_translation", "u2"),
                                             prior_lo=(1e-4, 0.0),
                                             prior_hi=(1.0,  0.5),
                                             predict=_predict_translation_only)

_REGISTRY["isotropic_rotor"] = ForwardModel(name="isotropic_rotor",
                                            param_names=("D_translation", "u2", "D_r"),
                                            prior_lo=(1e-4, 0.0,  1e-3),
                                            prior_hi=(1.0,  0.5,  5.0),
                                            predict=_predict_isotropic)

_REGISTRY["anisotropic_rotor"] = ForwardModel(name="anisotropic_rotor",
                                              param_names=("D_translation", "u2", "D_t", "D_s"),
                                              prior_lo=(1e-4, 0.0,  1e-3, 1e-3),
                                              prior_hi=(1.0,  0.5,  2.0,  5.0),
                                              predict=_predict_anisotropic)



# public API

def register_model(name: str,
                   *,
                    param_names: tuple[str, ...],
                    prior_lo: tuple[float, ...],
                    prior_hi: tuple[float, ...],
                    predict: Callable,
                    extras: dict | None = None,
                    overwrite: bool = False,
) -> ForwardModel:
    """Add a new forward model to the registry, or replace an existing one.

    Parameters
    ----------
    name : str
        Identifier used everywhere (in :func:`qens.fitting.find_map`,
        :func:`qens.sampling.run_mcmc`, etc.).
    param_names, prior_lo, prior_hi : tuples
        Same length. Defines a uniform prior box.
    predict : callable
        ``predict(omega, q, params, sigma_res, **extras) -> array``.
    extras : dict, optional
        Extra kwargs passed to ``predict`` (e.g. ``radius`` for a rotor).
    overwrite : bool
        Required to overwrite an existing model.

    Returns
    -------
    The registered :class:`ForwardModel` instance.
    """
    if name in _REGISTRY and not overwrite:
        raise KeyError(f"model {name!r} already registered; pass overwrite=True to replace")
    fm = ForwardModel(name=name,
                      param_names=tuple(param_names),
                      prior_lo=tuple(prior_lo),
                      prior_hi=tuple(prior_hi),
                      predict=predict,
                      extras=extras or {})
    _REGISTRY[name] = fm
    return fm


def get_model(name: str) -> ForwardModel:
    """Look up a forward model by name.

    Raises ``KeyError`` if not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def available_models() -> list[str]:
    """Return the list of currently-registered model names."""
    return sorted(_REGISTRY)
