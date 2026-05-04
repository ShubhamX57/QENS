.. _pipeline:

Analysis Pipeline
=================

The library supports two distinct analysis pathways. Choose based on your system.

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Pathway
     - Best for
     - Key function
   * - Joint forward-model inference
     - Molecular liquids with translation + rotation
     - :func:`~qens.fitting.find_map` + :func:`~qens.sampling.run_mcmc`
   * - Classical per-Q HWHM extraction
     - Atomic/ionic systems, quick scans
     - :func:`~qens.fitting.extract_hwhm`

Shared Stages (both pathways)
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 8 22 25 45

   * - Stage
     - Name
     - Module
     - What it does
   * - 01
     - Data ingestion
     - :mod:`qens.io`
     - Reads ``.nxspe`` HDF5 files. Tries multiple key paths for incident energy and detector angles. Rejects broken detectors. Parses sample, temperature, Eᵢ, and kind from the filename.
   * - 02
     - Elastic peak alignment
     - :mod:`qens.preprocessing`
     - Fits a Gaussian to the elastic line; shifts the energy axis so ħω = 0 is correctly centred.
   * - 03
     - Resolution assignment
     - :mod:`qens.preprocessing`
     - Identifies frozen-sample files (T ≤ ``frozen_temp_threshold``) and assigns σ\ :sub:`res` per incident energy. Falls back with a warning if no reference is found.

Pathway 1 — Joint Forward-Model Inference
-------------------------------------------

Preferred for molecular liquids where rotational broadening is present.
Fits all Q-bins simultaneously to the full *S*(*Q*, ω) forward model.
The Bessel-weight structure j₀(QR)² : j₁(QR)² : j₂(QR)² acts as a
global Q constraint that prevents the per-bin degeneracy of classical analysis.

.. list-table::
   :header-rows: 1
   :widths: 8 22 25 45

   * - Stage
     - Name
     - Module
     - What it does
   * - 04
     - Resolution binning
     - :mod:`qens.fitting`
     - Averages the frozen-sample spectrum over each Q bin to produce a measured resolution kernel (captures Lorentzian wings that a Gaussian misses).
   * - 05
     - Data binning
     - :mod:`qens.fitting`
     - Bins detectors into Q groups; averages spectra with proper error propagation.
   * - 06
     - MAP optimisation
     - :mod:`qens.fitting`
     - Multi-start Nelder-Mead MAP search over the joint log-likelihood. Returns the starting point for MCMC.
   * - 07
     - MCMC sampling
     - :mod:`qens.sampling`
     - emcee ensemble sampler (or Metropolis-Hastings fallback). Reports acceptance fraction, autocorrelation time, and Gelman-Rubin R̂.
   * - 08
     - Visualisation
     - :mod:`qens.plotting`
     - *S*(*Q*, ω) colour maps, Γ(*Q*²) dispersion, marginal/joint posteriors.

Pathway 2 — Classical HWHM Extraction
---------------------------------------

Independent elastic + Lorentzian fit per Q-bin, then fit Γ(Q) vs Q²
to a translational model. Quicker and sufficient for systems without rotation.

.. list-table::
   :header-rows: 1
   :widths: 8 22 25 45

   * - Stage
     - Name
     - Module
     - What it does
   * - 04
     - HWHM extraction
     - :mod:`qens.fitting`
     - Per-Q-bin curve fit of the convolved elastic + quasi-elastic + background model. Returns Q, Γ, δΓ, EISF arrays.
   * - 05
     - Dispersion fitting
     - User code + :mod:`qens.models`
     - ``scipy.optimize.curve_fit`` or Bayesian fit of Γ(Q) to Fickian, CE, or SS model.

See :doc:`usage_classical` for the complete classical workflow code.
