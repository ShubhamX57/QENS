"""
qens — Quasi-Elastic Neutron Scattering analysis
"""

from .config        import Config
from .io            import read_nxspe, read_nxspe_hdf5, load_dataset
from .preprocessing import fit_elastic_peak, assign_resolution
from .models        import ce, fickian, ss_model, lorentz, gnorm, make_basis
from .fitting       import (extract_hwhm, save_hwhm_csv, build_data_bins,
                            log_likelihood, log_prior, log_posterior, find_map)
from .sampling      import run_mcmc, summarise, gelman_rubin
from .              import plotting

__version__ = "0.2.0"
__author__  = "QENS Analysis Contributors"

__all__ = ["Config",
           "read_nxspe", "read_nxspe_hdf5", "load_dataset",
           "fit_elastic_peak", "assign_resolution",
           "ce", "fickian", "ss_model", "lorentz", "gnorm", "make_basis",
           "extract_hwhm", "save_hwhm_csv", "build_data_bins",
           "log_likelihood", "log_prior", "log_posterior", "find_map",
           "run_mcmc", "summarise", "gelman_rubin",
           "plotting",]
