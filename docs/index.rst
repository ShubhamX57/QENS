.. qens documentation master file

.. raw:: html

   <div class="qens-hero">

```
 <img src="_static/qens_logo_light.png" class="qens-logo qens-logo-light"
      alt="qens — Quasi-Elastic Neutron Scattering Analysis" />
 <img src="_static/qens_logo_dark.png"  class="qens-logo qens-logo-dark"
      alt="qens — Quasi-Elastic Neutron Scattering Analysis" />

 <div class="qens-badges">
   <a href="https://www.python.org/">
     <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square"
          alt="Python 3.10+">
   </a>
   <a href="https://github.com/ShubhamX57/QENS/releases">
     <img src="https://img.shields.io/badge/version-2.0.0-38bdf8?style=flat-square"
          alt="v2.0.0">
   </a>
   <a href="https://github.com/ShubhamX57/QENS/blob/main/LICENSE">
     <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square"
          alt="MIT License">
   </a>
   <a href="https://qens.readthedocs.io">
     <img src="https://img.shields.io/badge/docs-Read%20the%20Docs-teal?style=flat-square"
          alt="Read the Docs">
   </a>
 </div>

 <p class="qens-tagline">
   End-to-end Bayesian analysis of quasi-elastic neutron scattering data
   from ISIS Mantid <code>.nxspe</code> files — translational and rotational
   diffusion models, ensemble MCMC, full posterior inference.
 </p>

 <div class="qens-install">
   <span class="qens-prompt">$</span>
   <span class="qens-cmd">pip install "qens[mcmc]"</span>
 </div>

 <!-- ─── Library ecosystem strip ─── -->
 <p class="qens-eco-label">Built with</p>

 <div class="qens-eco-strip">

   <a href="https://numpy.org" class="qens-eco-card" title="NumPy">
     <div class="qens-eco-logo">
       <img src="https://numpy.org/images/logo.svg" alt="NumPy logo" />
     </div>
     <span class="qens-eco-name">NumPy</span>
     <span class="qens-eco-ver">≥&nbsp;1.24</span>
     <span class="qens-eco-desc">Array operations</span>
   </a>

   <a href="https://scipy.org" class="qens-eco-card" title="SciPy">
     <div class="qens-eco-logo">
       <img src="https://scipy.org/images/logo.svg" alt="SciPy logo" />
     </div>
     <span class="qens-eco-name">SciPy</span>
     <span class="qens-eco-ver">≥&nbsp;1.10</span>
     <span class="qens-eco-desc">FFT · NNLS · fitting</span>
   </a>

   <!-- Updated Matplotlib logo -->
   <a href="https://matplotlib.org" class="qens-eco-card" title="Matplotlib">
     <div class="qens-eco-logo">
       <img src="https://matplotlib.org/stable/_static/logo2.svg" alt="Matplotlib logo" />
     </div>
     <span class="qens-eco-name">Matplotlib</span>
     <span class="qens-eco-ver">≥&nbsp;3.7</span>
     <span class="qens-eco-desc">All plots</span>
   </a>

   <a href="https://www.h5py.org" class="qens-eco-card" title="h5py">
     <div class="qens-eco-logo qens-eco-icon" style="--ic:#1a73c8">
       <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
         <rect x="6"  y="28" width="36" height="10" rx="3" fill="var(--ic)" opacity=".4"/>
         <rect x="6"  y="17" width="36" height="10" rx="3" fill="var(--ic)" opacity=".65"/>
         <rect x="6"  y="6"  width="36" height="10" rx="3" fill="var(--ic)"/>
         <text x="24" y="15" text-anchor="middle" dominant-baseline="central"
               font-family="monospace" font-size="8" font-weight="700" fill="white">HDF5</text>
       </svg>
     </div>
     <span class="qens-eco-name">h5py</span>
     <span class="qens-eco-ver">≥&nbsp;3.7</span>
     <span class="qens-eco-desc">NeXus file I/O</span>
   </a>

   <a href="https://emcee.readthedocs.io" class="qens-eco-card" title="emcee">
     <div class="qens-eco-logo qens-eco-icon" style="--ic:#7c3aed">
       <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
         <rect width="48" height="48" rx="8" fill="var(--ic)" opacity=".12"/>
         <polyline
           points="4,38 9,26 14,34 19,20 24,30 29,18 34,28 39,16 44,22"
           stroke="var(--ic)" stroke-width="2.4" fill="none"
           stroke-linecap="round" stroke-linejoin="round"/>
         <circle cx="4"  cy="38" r="2.4" fill="var(--ic)"/>
         <circle cx="14" cy="34" r="2.4" fill="var(--ic)"/>
         <circle cx="24" cy="30" r="2.4" fill="var(--ic)"/>
         <circle cx="34" cy="28" r="2.4" fill="var(--ic)"/>
         <circle cx="44" cy="22" r="2.4" fill="var(--ic)"/>
       </svg>
     </div>
     <span class="qens-eco-name">emcee</span>
     <span class="qens-eco-ver">≥&nbsp;3.1</span>
     <span class="qens-eco-desc">Ensemble MCMC</span>
   </a>

   <!-- Updated Python logo -->
   <a href="https://www.python.org" class="qens-eco-card" title="Python">
     <div class="qens-eco-logo">
       <img src="https://www.python.org/static/community_logos/python-logo.svg" alt="Python logo" />
     </div>
     <span class="qens-eco-name">Python</span>
     <span class="qens-eco-ver">≥&nbsp;3.10</span>
     <span class="qens-eco-desc">Runtime</span>
   </a>

 </div><!-- /.qens-eco-strip -->
```

   </div><!-- /.qens-hero -->

   <style>
   /* (your full CSS unchanged) */
   </style>

.. ── Quick-nav card grid ──────────────────────────────────────────────────────

.. grid:: 2
:gutter: 2

.. grid-item-card::  Getting Started
:link: installation
:link-type: doc

```
  Install the library and run the 60-second example.
```

.. grid-item-card::  Quick Start
:link: quickstart
:link-type: doc

```
  Full workflow: load >> pre-process >> fit >> MCMC >> plot.
```

.. grid-item-card::  Physical Models
:link: models
:link-type: doc

```
  Fickian, Chudley-Elliott, Singwi-Sjölander, isotropic rotor,
  anisotropic rotor, and the custom model registry.
```

.. grid-item-card::  API Reference
:link: api/index
:link-type: doc

```
  Auto-generated reference for every public function and class.
```

---

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
