"""
Config dataclass — all analysis parameters in one place.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import List

@dataclass
class Config:
    files_to_fit: List[str] = field(default_factory=lambda: ["benzene_290_197_inc.nxspe",
                                                             "benzene_290_360_inc.nxspe",])
    
    primary_file: str = "benzene_290_360_inc.nxspe"

    q_min: float = 0.30
    q_max: float = 2.50

    ewin_hwhm: float = 0.80
    ewin_mcmc: float = 0.80

    n_bins:    int = 13
    n_bins_mc: int = 10

    n_walkers: int = 32
    n_warmup:  int = 500
    n_keep:    int = 2000
    thin:      int = 5

    random_seed: int = 42
    save_dir: str = "results"

    def __post_init__(self):
        if self.q_min >= self.q_max:
            raise ValueError(f"q_min ({self.q_min}) must be < q_max ({self.q_max})")
        if self.ewin_hwhm <= 0 or self.ewin_mcmc <= 0:
            raise ValueError("Energy windows must be > 0")
        if self.n_walkers < 4:
            raise ValueError("n_walkers must be ≥ 4")
        if self.n_walkers % 2 != 0:
            raise ValueError("n_walkers must be even")
        if self.n_bins < 2 or self.n_bins_mc < 2:
            raise ValueError("n_bins and n_bins_mc must be ≥ 2")
        if self.thin < 1:
            raise ValueError("thin must be ≥ 1")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  config saved → {path}")

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def __repr__(self) -> str:
        lines = ["Config("]
        for k, v in self.to_dict().items():
            lines.append(f"    {k} = {v!r},")
        lines.append(")")
        return "\n".join(lines)
