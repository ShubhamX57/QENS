# Installation

uqens needs Python 3.10 or newer. It is not on PyPI yet (the name `qens` there belongs to
an unrelated project; this package will be published as `uqens`), so install it from GitHub:

```bash
pip install "uqens[plot] @ git+https://github.com/ShubhamX57/QENS.git@v2"
```

The `plot` extra adds matplotlib for the report plots; leave it out if you only fit.

## For development

```bash
git clone https://github.com/ShubhamX57/QENS.git
cd QENS
git switch v2
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plot]"
python -m pytest
```

`python -m pytest` runs the fast tests (about a minute the first time). `python -m pytest -m slow`
adds the full benchmark fit and a synthetic recovery test, which take several minutes.

```{note}
The forward model is compiled by numba the first time it runs, so the first fit or test run
stays silent for a minute or two. The compiled code is cached, and later runs start in seconds.
Use `python -m pytest` rather than `pytest` so the tests run with the environment's Python.
```

## Dependencies

| Package | Used for |
|---|---|
| NumPy, SciPy | arrays, optimisation, special functions |
| numba | the compiled forward model and nuisance solver |
| h5py | reading `.nxspe` files |
| emcee | posterior sampling |
| matplotlib (optional) | report plots |
