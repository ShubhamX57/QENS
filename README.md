<p align="center">
  <img src="logo/qens_logo_dark.png" width="1000"/>
</p>


# qens — Quasi-Elastic Neutron Scattering analysis
[![Documentation Status](https://readthedocs.org/projects/qens/badge/?version=latest)](https://qens.readthedocs.io/en/latest/)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

End-to-end Python toolbox for analysing **Quasi-Elastic Neutron Scattering**
data from ISIS Mantid `.nxspe` files (IRIS, OSIRIS, LET, MARI, MAPS, …) and
any other Mantid-produced inelastic-spectrometer output.

The library is built around a clean separation between **physics models**
and **inference machinery**:

* The model catalogue includes Lorentzian and Gaussian primitives, three
  translational-diffusion HWHM models (Fickian, Chudley-Elliott,
  Singwi-Sjölander), and rotational structure factors (isotropic and
  anisotropic axial rotor, expanded to *l* = 2).
* These are combined into a full forward model
  S(Q, ω) = e<sup>−Q²⟨u²⟩/3</sup> · [translation x rotation] x resolution
  with a measured-kernel resolution path (use the frozen-sample
  S(Q, ω) directly : Lorentzian wings of real instruments don't fit a
  Gaussian).
* Inference is a joint-Q likelihood across **all** Q-bins simultaneously,
  with NNLS amplitude fitting per bin. MAP search is multi-start
  Nelder-Mead; MCMC is `emcee` (with a pure-NumPy Metropolis-Hastings
  fallback). Convergence diagnostics (autocorrelation time, Gelman-Rubin
  R̂) are reported automatically.
* Custom forward models can be registered without modifying core code :
  see [Custom models](#custom-models) below.

---


###  Package Installation

The `qens` package is not yet published on PyPI. Install directly from this repository:

```bash
# Clone and install locally
git clone https://github.com/ShubhamX57/QENS.git
cd QENS
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/ShubhamX57/QENS.git
```

Then use normally:
```python
import qens
```

> **Note**: The name `qens` was already taken on PyPI. We are working on publishing as **`uqens`** (Unified Quasi-Elastic Neutron Scattering analysis) soon.


---


## Library layout

```
qens/
├── __init__.py             public API surface
├── config.py               Config dataclass
├── constants.py            physical constants
├── io.py                   .nxspe reader + Q-from-2θ
├── preprocessing.py        elastic-peak alignment, resolution assignment
├── models/
│   ├── __init__.py
│   ├── lineshapes.py       Lorentzian, Gaussian, lorentz_sum
│   ├── translation.py      Fickian, CE, SS HWHM
│   ├── rotation.py         isotropic + anisotropic rotor widths,
│   │                       spherical-Bessel weights
│   ├── forward.py          predict_sqw, ForwardModel dataclass
│   └── registry.py         register_model / get_model / available_models
├── fitting.py              binning, joint likelihood, MAP, classical HWHM
├── sampling.py             emcee + MH fallback, summary, R̂
└── plotting.py             publication figures
```

---

## Citing

If `qens` is useful in published work please cite the repository and (if
the anisotropic-rotor model is used) the underlying paper:

> Richardson H., McColl K., Nilsen G. J., Armstrong J., McCluskey A. R.,
> *Lost in Translation: Simulation-Informed Bayesian Inference Improves
> Understanding of Molecular Motion From Neutron Scattering*,
> arXiv:2603.06080 (2026). https://arxiv.org/abs/2603.06080

---

## Contributing

Pull requests welcome. See `tests/` for the existing unit tests
(`pytest -v` to run). Style: `ruff check qens`. New physical models
should land in `qens/models/` and be registered in
`qens/models/registry.py`.


