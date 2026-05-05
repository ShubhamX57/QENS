.. _quickstart:

Quick Start
===========

This reproduces the 60-second example from the README with step-by-step
annotations. Use ``anisotropic_rotor`` for most molecular liquids; see
:doc:`models` for when to choose a simpler model.

Step 1 — Configure
------------------

.. code-block:: python

   from qens import Config

   cfg = Config(
       files_to_fit    = ["benzene_260_360_inc.nxspe",
                          "benzene_290_360_inc.nxspe"],
       primary_file    = "benzene_290_360_inc.nxspe",
       resolution_file = "benzene_260_360_inc.nxspe",
       q_min=0.6, q_max=1.8,
       energy_window=1.25,
       n_q_bins=12,
       n_walkers=32,
       n_warmup=500,
       n_keep=2000,
       random_seed=42
   )
   cfg.to_json("config.json")   # save for reproducibility

Step 2 — Load and Pre-process
------------------------------

.. code-block:: python

   from qens import load_dataset, fit_elastic_peak, assign_resolution

   dataset = load_dataset(cfg.files_to_fit, data_dir="data/",
                          critical_files=[cfg.primary_file])
   for d in dataset.values():
       fit_elastic_peak(d)         # centre elastic line at ħω = 0
   assign_resolution(dataset, cfg) # assign σ_res from frozen reference

Step 3 — Build Bins
--------------------

.. code-block:: python

   from qens import build_data_bins, build_resolution_bins

   target     = dataset[cfg.primary_file]
   resolution = dataset[cfg.resolution_file]

   data_bins = build_data_bins(target, cfg)
   res_bins  = build_resolution_bins(
       resolution, cfg,
       q_centres=[b[3] for b in data_bins]
   )

Step 4 — MAP + MCMC
--------------------

.. code-block:: python

   from qens import find_map, run_mcmc, summarise_samples

   p_map, _  = find_map(data_bins, res_bins,
                        model="anisotropic_rotor", cfg=cfg)
   samples   = run_mcmc(data_bins, res_bins, p_map,
                        model="anisotropic_rotor", cfg=cfg)

Step 5 — Report
----------------

.. code-block:: python

   import numpy as np

   summarise_samples(
       samples,
       model="anisotropic_rotor",
       derived={"D_s/D_t": lambda s: s[:, 3] / s[:, 2]}
   )

Expected output::

   D_translation   median=0.18312   95% CI=[0.17102, 0.19634]  Å²/ps
   u2              median=0.04021   95% CI=[0.03212, 0.05101]  Å²
   D_t             median=0.04134   95% CI=[0.03201, 0.05412]  ps⁻¹
   D_s             median=0.42301   95% CI=[0.37811, 0.48923]  ps⁻¹
   D_s/D_t         median=10.23     95% CI=[7.41, 14.32]

Step 6 — Plot
--------------

.. code-block:: python

   import qens.plotting as qp

   qp.plot_hwhm(target, cfg, samples,
                save_path="results/hwhm.png")
   qp.plot_posteriors(samples, model="anisotropic_rotor",
                      save_path="results/posteriors.png")
   qp.plot_sqw_maps(target,
                    save_path="results/sqw_map.png")

.. tip::

   For systems with no rotation (atoms, monatomic ions), use
   ``model="translation_only"`` and skip the resolution bins.
   See :doc:`usage_classical` for the classical per-Q HWHM workflow.
