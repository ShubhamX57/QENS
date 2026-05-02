.. qens documentation


qens — Quasi-Elastic Neutron Scattering Analysis


.. image:: https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square
   :alt: Python 3.10+
.. image:: https://img.shields.io/badge/version-2.0.0-38bdf8?style=flat-square
   :alt: v2.0.0
.. image:: https://img.shields.io/badge/license-MIT-green?style=flat-square
   :alt: MIT
.. image:: https://readthedocs.org/projects/qens/badge/?version=latest&style=flat-square
   :target: https://qens.readthedocs.io
   :alt: Docs

|

End-to-end Python library for analysing **Quasi-Elastic Neutron Scattering**
data from ISIS Mantid ``.nxspe`` files. Built around a clean separation
between **physics models** and **inference machinery** — from raw HDF5 files
to Bayesian posteriors in one pipeline.

.. grid:: 2
   :gutter: 2

   .. grid-item-card::  Getting Started
      :link: installation
      :link-type: doc

      Install the library and run the 60-second example.

   .. grid-item-card::  Quick Start
      :link: quickstart
      :link-type: doc

      Full workflow: load >> pre-process >> fit >> MCMC >> plot.

   .. grid-item-card::  Physical Models
      :link: models
      :link-type: doc

      Fickian, CE, SS, isotropic rotor, anisotropic rotor,
      and the custom model registry.

   .. grid-item-card::  API Reference
      :link: api/index
      :link-type: doc

      Auto-generated reference for every public function and class.

----

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   pipeline
   models
   custom_models
   config
   usage_classical
   usage_joint

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api/index
   api/io
   api/config
   api/preprocessing
   api/models
   api/fitting
   api/sampling
   api/plotting

.. toctree::
   :maxdepth: 1
   :caption: Project
   :hidden:

   changelog
   contributing
