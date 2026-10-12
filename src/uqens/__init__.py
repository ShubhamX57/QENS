"""uqens: one physical model, fitted to every incident energy and Q at once,
with uncertainties that are tested before they are reported.

Typical use::

    import uqens
    data = uqens.let.load(["benzene_290_124_inc.nxspe", "benzene_290_197_inc.nxspe",
                           "benzene_290_360_inc.nxspe", "benzene_290_861_inc.nxspe"])
    problem = uqens.Problem(uqens.LADDER["jump+iso"], data)
    best = uqens.fit(problem)
    post = uqens.sample(problem, best.x)
    print(best.summary(), post.summary(), sep="\\n")
"""

from . import let, validate
from .fit import FitResult, fit
from .model import LADDER, ModelSpec, Problem
from .sample import Posterior, sample

__version__ = "0.2.0.dev0"

__all__ = ["LADDER", "FitResult", "ModelSpec", "Posterior", "Problem", "fit", "let", "sample", "validate"]
