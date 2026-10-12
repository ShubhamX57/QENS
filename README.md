<p align="center">
  <img src="logo/qens_logo_dark.png#gh-dark-mode-only" width="1000"/>
  <img src="logo/qens_logo_light.png#gh-light-mode-only" width="1000"/>
</p>

# uqens: Quasi-Elastic Neutron Scattering analysis

[![tests](https://github.com/ShubhamX57/QENS/actions/workflows/ci.yml/badge.svg?branch=v2)](https://github.com/ShubhamX57/QENS/actions/workflows/ci.yml)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

One physical model, fitted to every incident energy and every Q at once, with
uncertainties that are tested before they are reported.

> **Status: version 2, in development.** This branch replaces the v0.1 library
> (tag `v0.1.0-legacy`). v0.1 reads the Q axis of LET `.nxspe` files wrongly
> (median error 42%), so results produced with it should not be used.

## What it does

- **Reads LET S(Q, ω) files correctly.** The constant-Q axis these reductions do
  not store is recovered and checked against the aluminium Bragg peak of the can;
  error bars are audited against the scatter of the data and rescaled.
- **Fits physics, not line shapes.** Translation (Fickian or jump diffusion) times
  rotation (isotropic or symmetric top, expanded until the sum rule is met),
  convolved analytically with a fitted, asymmetric resolution, with detailed
  balance. Per-Q amplitudes, can scattering and background are solved exactly.
- **Uses every incident energy together.** LET records several energies in one
  run; each sees a different time window, so one joint fit pins down what no
  single energy can.
- **Reports every optimum and sampled uncertainties.** Several starts, every
  distinct mode kept; error bars from posterior samples, with convergence
  diagnostics returned in the result.
- **Tests models before trusting them.** Leave-one-energy-out prediction and
  synthetic coverage tests are part of the library.

## Install

```bash
git clone https://github.com/ShubhamX57/QENS.git
cd QENS
git switch v2
pip install -e ".[dev]"
```

The name `qens` on PyPI belongs to an unrelated project; this package will be
published as `uqens`.

## Quick start

```python
import uqens

files = [f"data/benzene_290_{e}_inc.nxspe" for e in ("124", "197", "360", "861")]
data = uqens.let.load(files, verbose=True)          # Q axis, masks, error audit

problem = uqens.Problem(uqens.LADDER["jump+iso"], data)
best = uqens.fit(problem)                           # several starts, all optima kept
print(best.summary())

post = uqens.sample(problem, best.x, n_steps=3000)  # posterior samples
print(post.summary())                               # medians, 95% intervals, tau, R-hat
```

Or run the benchmark: `python examples/benzene_let.py --sample 3000 --holdout`.

With a measured resolution and an empty can (both must be at the same incident energy):

```python
data = uqens.let.load(files, background={3.60: "data/empty_360_inc.nxspe"})
uqens.let.attach_resolution(data[2], "data/benzene_260_360_inc.nxspe")
spec = uqens.ModelSpec("measured-res", resolution="measured")  # energies without a kernel stay analytic
```

## Benchmark: liquid benzene at 290 K (LET)

| Quantity | uqens (jump diffusion + isotropic rotation, 4 energies) | Reference |
|---|---|---|
| D (Å²/ps) | 0.172, 95% interval 0.162–0.182 | PFG-NMR ≈ 0.18 |
| τ₀ (ps) | 0.95 (0.71–1.20) | |
| D_r (ps⁻¹) | 0.166 (0.163–0.169) | |

The published LET analysis of the same data (Richardson et al. 2026) gives D = 0.182
with a different model and data selection. With the choices that paper describes,
these files give D ≈ 0.13 instead; see [docs/gate1.md](docs/gate1.md) and
`examples/gate1_published_setup.py`.

## Tests

```bash
pytest            # fast: Q axis, line shapes, rotation sum rule, solver, benchmark likelihood
pytest -m slow    # minutes: full benchmark fit and synthetic recovery
```

## Layout

```
src/uqens/
  let.py         LET reader: Q axis, binning, Bragg masks, error audit, resolution prior
  lineshapes.py  analytic Voigt convolution, asymmetric resolution
  rotation.py    rotational channels and the sum rule
  forward.py     forward model and exact per-bin nuisance solver (numba)
  model.py       ModelSpec, model ladder, parameters, priors, likelihood
  fit.py         multi-start best fit, distinct optima
  sample.py      posterior sampling with diagnostics
  validate.py    leave-one-energy-out and synthetic coverage tests
examples/        benzene benchmark
tests/           fast and slow tests
```

## Licence

MIT, see [LICENSE](LICENSE).
