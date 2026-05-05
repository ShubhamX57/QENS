"""
qens — Quasi-Elastic Neutron Scattering analysis library.

Open source toolbox for end to end QENS data analysis on ISIS format
".nxspe" files (IRIS, OSIRIS, LET, MARI, MAPS, Etc) and any other
Mantid produced inelastic neutron scattering data.

"""
__version__ = "2.0.0"
__author__  = "qens contributors"
__license__ = "MIT"



#  core dataclass
#  plotting (separate)
from . import plotting
from .config import Config

#  fitting + sampling
from .fitting import (build_data_bins,
                      build_resolution_bins,
                      extract_hwhm,
                      find_map,
                      log_likelihood,
                      log_posterior,
                      log_prior,
                      save_hwhm_csv)

#  IO + preprocessing
from .io import (compute_q_from_2theta,
                 inspect_nxspe,
                 load_dataset,
                 read_nxspe,
                 read_nxspe_with_overrides)

#  models API
from .models import (ForwardModel,
                     available_models,
                     bessel_weights,
                     ce_hwhm,
                     # translational HWHM
                     fickian_hwhm,
                     get_model,
                     gnorm,
                     # primitives
                     lorentz,
                     lorentz_sum,
                     # forward model + registry
                     predict_sqw,
                     register_model,
                     rot_widths_anisotropic,
                     # rotational structure-factor utilities
                     rot_widths_isotropic,
                     ss_hwhm)
from .preprocessing import assign_resolution, fit_elastic_peak
from .sampling import (gelman_rubin,
                       run_mcmc,
                       summarise,
                       summarise_samples)

__all__ = ["__version__",
           # config
           "Config",
           # IO
           "inspect_nxspe", "read_nxspe", "read_nxspe_with_overrides",
           "load_dataset", "compute_q_from_2theta",
           # preprocessing
           "fit_elastic_peak", "assign_resolution",
           # primitives
           "lorentz", "gnorm", "lorentz_sum",
           # translation
           "fickian_hwhm", "ce_hwhm", "ss_hwhm",
           # rotation
           "rot_widths_isotropic", "rot_widths_anisotropic", "bessel_weights",
           # forward
           "predict_sqw", "ForwardModel",
           "register_model", "get_model", "available_models",
           # fitting
           "build_data_bins", "build_resolution_bins",
           "extract_hwhm", "save_hwhm_csv",
           "log_likelihood", "log_prior", "log_posterior", "find_map",
           # sampling
           "run_mcmc", "summarise", "summarise_samples", "gelman_rubin",
           # plotting
           "plotting"]
