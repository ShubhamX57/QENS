"""
Physical models for QENS line shape analysis.

This subpackage is structured so that:

     :mod:`qens.models.lineshapes` provides the building blocks
      (Lorentzian, Gaussian).

     :mod:`qens.models.translation` contains models for translational
      diffusion (Fickian, Chudley-Elliott, Singwi-Sjölander).

     :mod:`qens.models.rotation` contains models for rotational diffusion
      (isotropic and anisotropic axial rotor).
      
     :mod:`qens.models.forward` assembles a full forward model
      ``S(Q,ω) = exp(-Q²<u²>/3) · [translation x rotation] x resolution``.

     :mod:`qens.models.registry` lets users register their own forward
      models without touching the core code.

For a typical user, ``from qens.models import predict_sqw`` is enough.


"""


from .lineshapes  import lorentz, gnorm, lorentz_sum
from .translation import fickian_hwhm, ce_hwhm, ss_hwhm
from .rotation    import (rot_widths_isotropic, rot_widths_anisotropic, bessel_weights)
from .forward     import predict_sqw, ForwardModel
from .registry    import register_model, get_model, available_models



__all__ = [# building blocks
           "lorentz", "gnorm", "lorentz_sum",
           # translational HWHM
           "fickian_hwhm", "ce_hwhm", "ss_hwhm",
           # rotational
           "rot_widths_isotropic", "rot_widths_anisotropic", "bessel_weights",
           # forward
           "predict_sqw", "ForwardModel",
           # extension API
           "register_model", "get_model", "available_models"]
