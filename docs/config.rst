.. _config:

Configuration Reference
=======================

.. autoclass:: qens.config.Config
   :members:
   :undoc-members:
   :no-index:

Serialisation
-------------

.. code-block:: python

   from qens import Config

   cfg = Config(n_walkers=64, energy_window=1.5, random_seed=7)
   cfg.to_json("run_config.json")

   # Reload in a different script — exact same parameters
   cfg2 = Config.from_json("run_config.json")

Parameter Guide
---------------

Q range (``q_min``, ``q_max``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detectors outside this range are excluded from all fitting. A good starting
range for molecular liquids at ISIS is 0.6 to 1.8 Å\ :sup:`-1`. Avoid regions
affected by coherent scattering (de Gennes narrowing near *Q* ≈ 1.3 Å\ :sup:`-1`
for benzene). Widen to 0.3–2.5 Å\ :sup:`-1` only when using polarisation-separated data.

Energy window (``energy_window``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Half-width of the ω window in meV. Set wide enough to capture the full
quasi-elastic wing. Richardson et al. (2026) found ±1.25 meV necessary
for benzene anisotropy at *E*\ :sub:`i` = 3.60 meV.

MCMC settings
~~~~~~~~~~~~~

- **n_walkers**: must be even and ≥ 2 × number of parameters. Use 32 for
  2-parameter fits, 64 for 4-parameter anisotropic rotor fits.
- **n_warmup**: burn-in per walker. Start at 500 and increase if the
  acceptance fraction is outside 0.25–0.55.
- **n_keep**: production steps. 2000 gives smooth histograms; 5000 for
  publication figures.
- **thin**: default 5 reduces autocorrelation without wasting samples.

Resolution (``resolution_file``, ``frozen_temp_threshold``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``resolution_file`` is set explicitly, that file is always used as the
resolution reference regardless of temperature. Otherwise
:func:`~qens.preprocessing.assign_resolution` auto-picks any incoherent
file at T ≤ ``frozen_temp_threshold`` (default 270 K).
