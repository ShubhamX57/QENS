"""
qens — Quasi-Elastic Neutron Scattering analysis library

A modular Python library for analysing QENS data from Pelican instruments.
Implements Bayesian inference of self-diffusion parameters using the
Chudley-Elliott jump-diffusion model.

The basic workflow looks like this:

    from qens.config        import Config
    from qens.io            import load_dataset
    from qens.preprocessing import fit_elastic_peak, assign_resolution
    from qens.fitting       import extract_hwhm, build_data_bins, find_map
    from qens.sampling      import run_mcmc, summarise
    import qens.plotting as qplt

    cfg = Config(q_min=0.3, q_max=2.5)

    ds = load_dataset(["my_data_290_360_inc.nxspe"], data_dir="./data")
    for d in ds.values():
        fit_elastic_peak(d)
    assign_resolution(ds)

    d_inc = ds["my_data_290_360_inc.nxspe"]
    q, g, g_err, eisf = extract_hwhm(d_inc, cfg)

    bins = build_data_bins(d_inc, cfg)
    d_map, l_map, tau = find_map(bins, d_inc["sigma_res"], cfg)
    samples = run_mcmc(bins, d_inc["sigma_res"], d_map, l_map, cfg)

    d_med, d_lo, d_hi = summarise(samples[:,0], "D (Å²/ps)")

Modules
-------
    constants   — physical constants and instrument layout
    config      — all tunable parameters in one dataclass
    io          — loading .nxspe binary files
    preprocessing — elastic peak alignment, resolution assignment
    models      — diffusion models (CE, Fickian, SS) and spectral functions
    fitting     — HWHM extraction, Bayesian posterior, MAP estimation
    sampling    — MCMC sampling, Gelman-Rubin convergence diagnostic
    plotting    — publication-quality figures
"""

from .config        import Config
from .io            import read_nxspe, load_dataset
from .preprocessing import fit_elastic_peak, assign_resolution
from .models        import ce, fickian, ss_model, lorentz, gnorm, make_basis
from .fitting       import (extract_hwhm,
                            save_hwhm_csv,
                            build_data_bins,
                            log_likelihood,
                            log_prior,
                            log_posterior,
                            find_map,)

from .sampling      import run_mcmc, summarise, gelman_rubin
from .              import plotting

__version__ = "0.1.0"
__author__  = "QENS Analysis Contributors"

__all__ = [
    # config
    "Config",
    # io
    "read_nxspe", "load_dataset",
    # preprocessing
    "fit_elastic_peak", "assign_resolution",
    # models
    "ce", "fickian", "ss_model", "lorentz", "gnorm", "make_basis",
    # fitting
    "extract_hwhm", "save_hwhm_csv", "build_data_bins",
    "log_likelihood", "log_prior", "log_posterior", "find_map",
    # sampling
    "run_mcmc", "summarise", "gelman_rubin",
    # plotting
    "plotting",]
