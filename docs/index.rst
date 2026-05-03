.. qens documentation master file

.. raw:: html

   <div class="qens-hero">

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

       <!-- Matplotlib: local static file -->
       <a href="https://matplotlib.org" class="qens-eco-card" title="Matplotlib">
         <div class="qens-eco-logo">
           <img src="_static/matplotlib_logo.svg"
                alt="Matplotlib logo"
                class="matplotlib-logo" />
         </div>
         <span class="qens-eco-name">Matplotlib</span>
         <span class="qens-eco-ver">≥&nbsp;3.7</span>
         <span class="qens-eco-desc">All plots</span>
       </a>

       <a href="https://www.h5py.org" class="qens-eco-card" title="h5py">
         <div class="qens-eco-logo qens-eco-icon" style="--ic:#1a73c8">
           <!-- h5py: chunked-storage icon — three stacked H5 file layers -->
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
           <!-- emcee: MCMC walker chain symbol -->
           <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
             <rect width="48" height="48" rx="8" fill="var(--ic)" opacity=".12"/>
             <!-- chain of walkers as winding path -->
             <polyline
               points="4,38 9,26 14,34 19,20 24,30 29,18 34,28 39,16 44,22"
               stroke="var(--ic)" stroke-width="2.4" fill="none"
               stroke-linecap="round" stroke-linejoin="round"/>
             <!-- walker dots -->
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

       <!-- Python: local static file -->
       <a href="https://www.python.org" class="qens-eco-card" title="Python">
         <div class="qens-eco-logo python-logo-container">
           <img src="_static/python_logo.png"
                alt="Python logo" />
         </div>
         <span class="qens-eco-name">Python</span>
         <span class="qens-eco-ver">≥&nbsp;3.10</span>
         <span class="qens-eco-desc">Runtime</span>
       </a>

     </div><!-- /.qens-eco-strip -->

   </div><!-- /.qens-hero -->

   <style>
   /* ─── Hero wrapper ─── */
   .qens-hero {
     text-align: center;
     padding: 2.5rem 0 2rem;
     border-bottom: 1px solid var(--color-border-secondary, #e0e0e0);
     margin-bottom: 2.5rem;
   }

   /* ─── Logo switch (light / dark) ─── */
   .qens-logo { max-width: min(680px,90%); height:auto; display:block; margin:0 auto 1.8rem; }
   .qens-logo-dark  { display:none; }
   .qens-logo-light { display:block; }
   body[data-theme="dark"]  .qens-logo-light { display:none;  }
   body[data-theme="dark"]  .qens-logo-dark  { display:block; }
   @media (prefers-color-scheme:dark) {
     body[data-theme="auto"] .qens-logo-light { display:none;  }
     body[data-theme="auto"] .qens-logo-dark  { display:block; }
   }

   /* ─── Status badges ─── */
   .qens-badges {
     display:flex; flex-wrap:wrap; justify-content:center;
     gap:0.45rem; margin-bottom:1.4rem;
   }
   .qens-badges a {
     display:inline-block;
     line-height:0;
     transition:opacity .15s;
   }
   .qens-badges a:hover { opacity:.78; text-decoration:none; }
   .qens-badges img {
     border:0;
     height:auto;
   }

   /* ─── Tagline ─── */
   .qens-tagline {
     font-size:1.0rem; color:var(--color-foreground-secondary,#555);
     max-width:580px; margin:0 auto 1.6rem; line-height:1.65;
   }

   /* ─── Install box ─── */
   .qens-install {
     display:inline-flex; align-items:center;
     background:var(--color-background-secondary,#f8f8f8);
     border:1px solid var(--color-background-border,#ccc);
     border-radius:7px; overflow:hidden;
     font-family:var(--font-stack--monospace,monospace);
     font-size:0.88rem; margin-bottom:2.4rem;
   }
   .qens-prompt {
     padding:0.65rem 0.9rem;
     background:rgba(56,189,248,.10); color:#0284c7;
     border-right:1px solid var(--color-background-border,#ccc);
     user-select:none; font-weight:600;
   }
   .qens-cmd { padding:0.65rem 1.1rem; color:var(--color-foreground-primary,#111); }
   body[data-theme="dark"] .qens-prompt { color:#38bdf8; }
   @media (prefers-color-scheme:dark){
     body[data-theme="auto"] .qens-prompt { color:#38bdf8; }
   }

   /* ─── "Built with" label ─── */
   .qens-eco-label {
     font-family:var(--font-stack--monospace,monospace);
     font-size:0.68rem; letter-spacing:0.16em; text-transform:uppercase;
     color:var(--color-foreground-secondary,#888);
     margin:0 0 1.1rem;
   }

   /* ─── Ecosystem strip ─── */
   .qens-eco-strip {
     display:flex; flex-wrap:wrap;
     justify-content:center; gap:1rem;
     padding:0 1rem 0.5rem;
   }

   /* ─── Individual library card ─── */
   .qens-eco-card {
     display:flex; flex-direction:column; align-items:center; gap:0.35rem;
     width:100px; padding:1.1rem 0.8rem 0.9rem;
     border-radius:12px;
     border:1px solid var(--color-background-border,#e2e8f0);
     background:var(--color-background-primary,#fff);
     text-decoration:none;
     transition:transform .18s, box-shadow .18s, border-color .18s;
     box-shadow:0 1px 4px rgba(0,0,0,.06);
   }
   .qens-eco-card:hover {
     transform:translateY(-4px);
     box-shadow:0 8px 24px rgba(0,0,0,.12);
     border-color:var(--color-brand-primary,#0a7abf);
     text-decoration:none;
   }
   body[data-theme="dark"] .qens-eco-card {
     background:#111827; border-color:#1e2d47;
     box-shadow:0 1px 6px rgba(0,0,0,.5);
   }
   body[data-theme="dark"] .qens-eco-card:hover {
     border-color:#38bdf8;
     box-shadow:0 8px 28px rgba(56,189,248,.15);
   }
   @media (prefers-color-scheme:dark){
     body[data-theme="auto"] .qens-eco-card { background:#111827; border-color:#1e2d47; box-shadow:0 1px 6px rgba(0,0,0,.5); }
     body[data-theme="auto"] .qens-eco-card:hover { border-color:#38bdf8; box-shadow:0 8px 28px rgba(56,189,248,.15); }
   }

   /* ─── Logo box ─── */
   .qens-eco-logo {
     width:52px; height:52px;
     display:flex; align-items:center; justify-content:center;
   }
   .qens-eco-logo img, .qens-eco-logo svg {
     width:48px; height:48px; object-fit:contain; display:block;
   }
   /* icon-only cards: add subtle bg circle */
   .qens-eco-icon {
     border-radius:10px;
     background:rgba(0,0,0,.04);
   }
   body[data-theme="dark"] .qens-eco-icon,
   body[data-theme="auto"] .qens-eco-icon { background:rgba(255,255,255,.05); }

   /* Fix Matplotlib logo for dark mode */
   body[data-theme="dark"] .matplotlib-logo,
   body[data-theme="auto"] .matplotlib-logo {
     filter: invert(1);
   }

   /* Python logo container - ensure visibility */
   .python-logo-container {
     background: white;
     border-radius: 12px;
     padding: 4px;
   }
   body[data-theme="dark"] .python-logo-container,
   body[data-theme="auto"] .python-logo-container {
     background: #1e1e2f;
   }

   /* ─── Card text ─── */
   .qens-eco-name {
     font-family:var(--font-stack--monospace,monospace);
     font-size:0.80rem; font-weight:700;
     color:var(--color-foreground-primary,#111);
     letter-spacing:0.01em;
   }
   .qens-eco-ver {
     font-family:var(--font-stack--monospace,monospace);
     font-size:0.62rem;
     color:var(--color-foreground-secondary,#777);
     background:var(--color-background-secondary,#f3f4f6);
     padding:0.1rem 0.45rem;
     border-radius:20px;
     border:1px solid var(--color-background-border,#e2e8f0);
     white-space:nowrap;
   }
   .qens-eco-desc {
     font-size:0.65rem;
     color:var(--color-foreground-secondary,#888);
     text-align:center; line-height:1.3;
   }
   body[data-theme="dark"] .qens-eco-name  { color:#cdd5e0; }
   body[data-theme="dark"] .qens-eco-ver   { background:#1a2236; color:#7a95b4; border-color:#1e2d47; }
   body[data-theme="dark"] .qens-eco-desc  { color:#4b6080; }
   @media (prefers-color-scheme:dark){
     body[data-theme="auto"] .qens-eco-name { color:#cdd5e0; }
     body[data-theme="auto"] .qens-eco-ver  { background:#1a2236; color:#7a95b4; border-color:#1e2d47; }
     body[data-theme="auto"] .qens-eco-desc { color:#4b6080; }
   }

   /* ─── Responsive ─── */
   @media (max-width:520px) {
     .qens-eco-strip { gap:0.6rem; }
     .qens-eco-card  { width:82px; padding:0.8rem 0.5rem 0.7rem; }
     .qens-eco-logo, .qens-eco-logo img, .qens-eco-logo svg { width:38px; height:38px; }
   }
   </style>

.. ── Quick-nav card grid ──────────────────────────────────────────────────────

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 🚀 Getting Started
      :link: installation
      :link-type: doc

      Install the library and run the 60-second example.

   .. grid-item-card:: ⚡ Quick Start
      :link: quickstart
      :link-type: doc

      Full workflow: load → pre-process → fit → MCMC → plot.

   .. grid-item-card:: 🔬 Physical Models
      :link: models
      :link-type: doc

      Fickian, Chudley-Elliott, Singwi-Sjölander, isotropic rotor,
      anisotropic rotor, and the custom model registry.

   .. grid-item-card:: 📖 API Reference
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
