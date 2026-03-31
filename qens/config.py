"""
config.py

One place for all the knobs you'd want to turn between runs.
The idea is that you create a Config object at the top of your script,
adjust whatever you need, and pass it through the pipeline.
That way there's never any question about what settings produced a
particular result — just serialize the config alongside your output.

Example usage
-------------
    from qens.config import Config

    cfg = Config(q_min=0.4, q_max=2.2, n_walkers=64)
    cfg.to_json("run_settings.json")

    # or reload later
    cfg = Config.from_json("run_settings.json")
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class Config:
    # --- which files to actually fit -----------------------------------------
    # files_to_fit is the list that goes through the full Bayesian pipeline.
    # primary_file is the one used for single-spectrum diagnostic plots.

    files_to_fit: List[str] = field(default_factory=lambda: ["benzene_290_197_inc.nxspe",
                                                             "benzene_290_360_inc.nxspe",])

    primary_file: str = "benzene_290_360_inc.nxspe"


    # --- Q range -------------------------------------------------------------
    # below q_min the resolution usually dominates; above q_max you start
    # worrying about multiple scattering. adjust to your sample.

    q_min: float = 0.30    # Å⁻¹
    q_max: float = 2.50    # Å⁻¹


    # --- energy windows ------------------------------------------------------
    # ewin_hwhm is used when fitting each Q-bin to extract the linewidth.
    # ewin_mcmc is fed to the Bayesian likelihood — can be narrower if the
    # wings are noisy.

    ewin_hwhm: float = 0.80   # meV
    ewin_mcmc: float = 0.80   # meV


    # --- binning -------------------------------------------------------------
    n_bins:    int = 13   # Q bins for the HWHM extraction
    n_bins_mc: int = 10   # Q bins used in the MCMC likelihood


    # --- MCMC settings -------------------------------------------------------
    # n_walkers must be even and at least 2*ndim (ndim=2 here, so min is 4,
    # but 32 or more is sensible in practice).
    # n_warmup is discarded burn-in, n_keep is what you actually sample from.

    n_walkers: int = 32
    n_warmup:  int = 500
    n_keep:    int = 2000
    thin:      int = 5      # keep every nth sample to reduce autocorrelation

    random_seed: int = 42   # used everywhere: MAP starts, walkers, fan plots


    # --- output --------------------------------------------------------------
    save_dir: str = "results"


    # --- helpers -------------------------------------------------------------

    def to_dict(self):
        return asdict(self)

    def to_json(self, path):
        """save config to a json file so you can reproduce the run later"""
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_json(cls, path):
        """reload a config that was saved with to_json"""
        with open(path) as fh:
            data = json.load(fh)
        return cls(**data)

    def __repr__(self):
        lines = ["Config("]
        for k, v in self.to_dict().items():
            lines.append(f"    {k} = {v!r},")
        lines.append(")")
        return "\n".join(lines)
