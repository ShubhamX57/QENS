.. _installation:

Installation
============

Requirements
------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Package
     - Version
     - Purpose
   * - Python
     - ≥ 3.10
     - Runtime
   * - NumPy
     - ≥ 1.24
     - Array operations throughout
   * - SciPy
     - ≥ 1.10
     - FFT convolution, NNLS, curve fitting, spherical Bessel functions
   * - Matplotlib
     - ≥ 3.7
     - All plotting functions in ``qens.plotting``
   * - h5py
     - ≥ 3.7
     - Reading ``.nxspe`` HDF5 files from ISIS Mantid
   * - emcee
     - ≥ 3.1 *(optional)*
     - Ensemble MCMC sampler. Falls back to Metropolis-Hastings if absent.

From PyPI
---------

.. code-block:: bash

   pip install qens

   # with the recommended ensemble sampler
   pip install "qens[mcmc]"

From Source (recommended for development)
------------------------------------------

.. code-block:: bash

   git clone https://github.com/ShubhamX57/QENS.git
   cd QENS
   pip install -e ".[mcmc]"

Verify
------

.. code-block:: python

   import qens
   print(qens.__version__)   # 2.0.0
   print(qens.available_models())
   # ['anisotropic_rotor', 'isotropic_rotor', 'translation_only']

.. note::

   If ``emcee`` is not installed, :func:`~qens.sampling.run_mcmc` prints a
   warning and falls back to a 4-chain Metropolis-Hastings sampler.
   Results are statistically equivalent but the ensemble autocorrelation
   time estimate is not available.

Connecting to Read the Docs
----------------------------

The repo ships a ``.readthedocs.yaml`` at the project root. To publish:

1. Push your fork to GitHub.
2. Log in at `readthedocs.org <https://readthedocs.org>`_ with your GitHub account.
3. Click **Import a Project** and select ``ShubhamX57/QENS``.
4. Read the Docs detects ``.readthedocs.yaml`` automatically — click **Build**.
5. Docs go live at ``https://qens.readthedocs.io/en/latest/``.

Every ``git push`` to ``main`` triggers an automatic rebuild.
