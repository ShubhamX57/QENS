.. _custom_models:

Custom Models
=============

You can register your own forward model without modifying any core source
file. The custom model immediately has access to the full inference
machinery: joint-Q likelihood, NNLS amplitude optimisation, MAP search,
MCMC, and all summary and plotting functions.

Minimal Example
---------------

.. code-block:: python

   import numpy as np
   from scipy.signal           import fftconvolve
   from qens                   import register_model
   from qens.constants         import HBAR_MEV_PS
   from qens.models.lineshapes import lorentz
   from qens.models.forward    import _make_resolution_kernel

   def predict_two_lorentzian(omega, q, params, sigma_res, **_):
       """Translational diffusion + a fast secondary mode."""
       D_slow, D_fast, frac_fast = params
       g_slow = HBAR_MEV_PS * D_slow * q * q
       g_fast = HBAR_MEV_PS * D_fast * q * q
       s = ((1 - frac_fast) * lorentz(omega, g_slow)
            +      frac_fast * lorentz(omega, g_fast))
       kernel = _make_resolution_kernel(omega, sigma_res)
       return fftconvolve(s, kernel, mode="same") * (omega[1] - omega[0])

   register_model(
       name        = "two_lorentzian",
       param_names = ("D_slow", "D_fast", "frac_fast"),
       prior_lo    = (1e-4, 1e-3, 0.0),
       prior_hi    = (1.0,  3.0,  1.0),
       predict     = predict_two_lorentzian
   )

The model is now usable everywhere a built-in name would be:

.. code-block:: python

   from qens import find_map, run_mcmc, summarise_samples

   p_map, _  = find_map(data_bins, res_bins, model="two_lorentzian")
   samples   = run_mcmc(data_bins, res_bins, p_map, model="two_lorentzian")
   summarise_samples(samples, model="two_lorentzian")

Overriding Built-in Priors
---------------------------

Pass ``overwrite=True`` to replace an existing model's prior box without
changing the physics:

.. code-block:: python

   from qens.models.registry import get_model
   from qens                 import register_model

   old = get_model("anisotropic_rotor")

   register_model(
       name        = "anisotropic_rotor",
       param_names = old.param_names,
       prior_lo    = (1e-4, 0.0, 1e-3, 0.05),   # D_s must be ≥ 0.05
       prior_hi    = (0.5,  0.3, 1.0,  2.0),
       predict     = old.predict,
       overwrite   = True
   )

Passing Extra Fixed Parameters
--------------------------------

Use the ``extras`` dict for constants that are not sampled (e.g. a known
molecular radius, or a fixed background level):

.. code-block:: python

   def predict_rotor_custom(omega, q, params, sigma_res, radius=2.48, **_):
       ...

   register_model(
       name        = "my_rotor",
       param_names = ("D_star", "u2", "D_r"),
       prior_lo    = (1e-4, 0.0, 1e-3),
       prior_hi    = (1.0,  0.5, 5.0),
       predict     = predict_rotor_custom,
       extras      = {"radius": 3.12}    # naphthalene
   )

predict() Signature Contract
------------------------------

Your ``predict`` callable must match this signature exactly:

.. code-block:: python

   def my_model(
       omega:     np.ndarray,   # energy grid, meV, uniformly spaced
       q:         float,        # momentum transfer, Å⁻¹
       params:    tuple,        # sampled parameters, same order as param_names
       sigma_res,               # scalar σ (meV) or array resolution kernel
       **extras,                # forwarded from ForwardModel.extras
   ) -> np.ndarray:             # predicted S(Q,ω), same shape as omega
       ...

The return value is the unnormalised predicted spectrum. The NNLS step in
:func:`~qens.fitting.log_likelihood` fits an overall amplitude and a flat
background on top of it, so you do not need to normalise.
