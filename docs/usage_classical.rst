.. _usage_classical:

Classical HWHM Workflow
========================

Use this workflow for atomic or ionic systems where rotational broadening
is absent. It extracts Γ(Q) per Q-bin then fits a translational model.

Full Script
-----------

.. code-block:: python

   from qens import (
       Config, load_dataset, fit_elastic_peak, assign_resolution,
       extract_hwhm, save_hwhm_csv,
   )
   from qens.models    import fickian_hwhm, ce_hwhm
   from scipy.optimize import curve_fit
   import numpy as np

   cfg = Config(
       files_to_fit    = ["benzene_260_360_inc.nxspe",
                          "benzene_290_360_inc.nxspe"],
       primary_file    = "benzene_290_360_inc.nxspe",
       resolution_file = "benzene_260_360_inc.nxspe",
       q_min=0.3, q_max=2.5,
       energy_window=0.8,
       n_q_bins=13
   )

   # Load and pre-process
   dataset = load_dataset(cfg.files_to_fit, data_dir="data/",
                          critical_files=[cfg.primary_file])
   for d in dataset.values():
       fit_elastic_peak(d)
   assign_resolution(dataset, cfg)

   target = dataset[cfg.primary_file]

   # Extract HWHM per Q-bin
   q, gamma, gerr, eisf = extract_hwhm(target, cfg)
   save_hwhm_csv(q, gamma, gerr, eisf, save_dir="results/")

   # Fit Fickian model
   p_fick, cov_fick = curve_fit(
       fickian_hwhm, q, gamma, sigma=gerr,
       p0=[0.15], bounds=([1e-3], [3.0])
   )
   print(f"Fickian  D = {p_fick[0]:.4f} ± {cov_fick[0,0]**0.5:.4f} Å²/ps")

   # Fit Chudley-Elliott model
   p_ce, cov_ce = curve_fit(
       ce_hwhm, q, gamma, sigma=gerr,
       p0=[0.15, 2.0], bounds=([1e-3, 0.1], [3.0, 6.0])
   )
   print(f"CE       D = {p_ce[0]:.4f} ± {cov_ce[0,0]**0.5:.4f} Å²/ps"
         f"   ℓ = {p_ce[1]:.3f} Å")

   # Plot
   import qens.plotting as qp
   qp.plot_hwhm_classical(q, gamma, gerr,
                          fits={"Fickian": (fickian_hwhm, p_fick),
                                "CE":      (ce_hwhm,      p_ce)},
                          save_path="results/hwhm_classical.png")

Interpreting the Dispersion Plot
---------------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Interpretation
   * - Linear at low Q²
     - Slope = ħD. Consistent with continuous Fickian diffusion.
   * - Plateau at high Q²
     - Jump-diffusion saturation. CE or SS model applies.
   * - Below resolution line
     - Spectral resolution too coarse for this sample/Eᵢ combination.
   * - Non-monotonic dip
     - Likely coherent contamination near de Gennes peak.

.. warning::

   If the HWHM plateau looks identical to the CE jump-diffusion
   saturation *and* the sample is a molecular liquid, the broadening
   may be rotational rather than translational. Use the joint
   forward-model workflow (:doc:`usage_joint`) to distinguish them.
   See :doc:`models` for the "Lost in Translation" degeneracy.
