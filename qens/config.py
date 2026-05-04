"""
Configuration dataclass - runtime parameters for the QENS analysis pipeline.

Holds every tunable knob in one place: file lists, Q range, energy window,
binning, MCMC settings, output directory. 

Save/restore as JSON forreproducibility.

"""
from __future__ import annotations


import json
from dataclasses import dataclass, field, asdict
from typing import List



@dataclass
class Config:
    """
    
    Runtime parameters for the analysis pipeline.

    Attributes
    ----------

    files_to_fit : list[str]
        Filenames to load (relative to ``data_dir`` passed to
        :func:`qens.io.load_dataset`).

    primary_file : str
        Which of the loaded files is the *target* spectrum to fit.

    resolution_file : str | None
        Frozen-sample file used as resolution function. If None, the loader
        auto-picks any T ≤ ``frozen_temp_threshold`` incoherent file.

    frozen_temp_threshold : int
        Files at T ≤ this temperature (K) are treated as resolution refs.
    q_min, q_max : float
        Q range in Å⁻¹ over which the fit is performed.

    energy_window : float
        Half-width of the ω window in meV used for the joint fit
        (paper found ±1.25 meV needed for benzene anisotropy).

    n_q_bins : int
        Number of Q-bins for the joint S(Q,ω) fit.

    n_walkers : int
        Number of emcee walkers (must be even, ≥ 2 × n_dim).

    n_warmup, n_keep : int
        Burn-in and production steps per walker.

    thin : int
        Chain thinning factor.

    n_map_starts : int
        Random starts for the MAP search.

    random_seed : int
        Master seed for reproducibility.

    save_dir : str
        Output directory for figures, samples, summaries.

    """

    #  data
    files_to_fit: List[str] = field(default_factory=list)
    primary_file: str = ""
    resolution_file: str | None = None
    frozen_temp_threshold: int = 270 # defaukt threhold 


    #  fit window
    q_min: float = 0.30
    q_max: float = 2.50
    energy_window: float = 1.25


    #  binning
    n_q_bins: int = 12


    #  sampling
    n_walkers: int = 32
    n_warmup: int = 500
    n_keep: int = 2000
    thin: int = 5
    n_map_starts: int = 30


    #  misc
    random_seed: int = 42
    save_dir: str = "qens_results"


    #  validation on init
    def __post_init__(self):
        if self.q_min >= self.q_max:
            raise ValueError(
                f"q_min ({self.q_min}) must be < q_max ({self.q_max})"
            ) # check Q range
        if self.energy_window <= 0:
            raise ValueError(
                f"energy_window must be > 0, got {self.energy_window}"
            ) # energy window should larger than 0
        if self.n_walkers < 4 or self.n_walkers % 2:
            raise ValueError("n_walkers must be even and ≥ 4")  # emcee requirement
        if self.n_q_bins < 2:
            raise ValueError("n_q_bins must be ≥ 2")
        if self.thin < 1:
            raise ValueError("thin must be ≥ 1") # >= 1 to aviod error and infinite loop
        if self.n_warmup < 0 or self.n_keep < 1:
            raise ValueError("n_warmup ≥ 0 and n_keep ≥ 1 required")


    #  (de)serialise
    def to_dict(self) -> dict:
        # store as dict
        return asdict(self)


    def to_json(self, path: str) -> None:
        # store as json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(**json.load(f))


    def __repr__(self) -> str:
        # ouput parameters and values for viewing
        lines = ["Config("]
        for k, v in self.to_dict().items():
            lines.append(f"    {k} = {v!r},")
        lines.append(")")
        return "\n".join(lines)
