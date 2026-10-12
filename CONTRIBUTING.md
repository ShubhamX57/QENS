# Contributing

Thanks for contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/ShubhamX57/QENS.git
cd QENS
git switch v2
pip install -e ".[dev]"
```

## Workflow

1. Create a branch from `v2`: `git switch -c feature/your-feature`
2. Make changes, then run `ruff check .`, `ruff format .` and `pytest`
3. If you touched the reader, the forward model or the fitting, also run `pytest -m slow`
4. Commit, push and open a pull request against `v2`

## Rules that keep results trustworthy

- Every change to the physics or the data handling comes with a test that would have
  failed before it (a known value, a sum rule, a synthetic recovery).
- `tests/test_model.py::test_benchmark_likelihood_is_unchanged` guards the benzene
  benchmark. If a change is meant to move it, say why in the pull request and update
  the stored value in the same commit.
- Uncertainties come from samples, not from the curvature at the best fit.

## Reporting bugs

Include: Python version, uqens version (`uqens.__version__`), the traceback, and the
smallest code and data that reproduce it.

## Questions?

Open an [issue](https://github.com/ShubhamX57/QENS/issues).
