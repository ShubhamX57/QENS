# Quick start

The benchmark: liquid benzene at 290 K measured on LET (ISIS) with four incident energies
recorded at once. The data files are in the repository's `data/` folder.

## Fit one model to every energy

```python
import uqens

files = [f"data/benzene_290_{e}_inc.nxspe" for e in ("124", "197", "360", "861")]
data = uqens.let.load(files, verbose=True)          # Q axis, masks, error audit

problem = uqens.Problem(uqens.LADDER["jump+iso"], data)
best = uqens.fit(problem)                           # several starts, every optimum kept
print(best.summary())
```

`let.load` prints what it did to each file: the number of points and Q-bins, and how much
it scaled the error bars after comparing them with the scatter of the data. Expect
D ≈ 0.172 Å²/ps, τ₀ ≈ 0.95 ps and D_r ≈ 0.166 ps⁻¹; PFG-NMR gives D ≈ 0.18 Å²/ps.

## Uncertainties

```python
post = uqens.sample(problem, best.x, n_steps=3000)
print(post.summary())                # medians, 95% intervals, autocorrelation times, R-hat
post.interval("lnD")                 # (low, high) in Å²/ps
post.converged                       # check this before quoting the numbers
```

## Does the model predict data it has not seen?

```python
rows = uqens.validate.holdout(problem.spec, data, problem, best.x, verbose=True)
```

Each energy is left out in turn: the physics is fitted on the other three and predicts the
fourth, with only the instrument terms refitted.

## Plots

```python
from uqens import plot

plot.spectra(problem, best.x, energy=3.60).savefig("spectra.png")
plot.misfit_map({"jump + rotation": (problem, best.x)}).savefig("misfit.png")
plot.holdout(rows).savefig("holdout.png")
plot.timescales(problem, best.x, posterior=post).savefig("timescales.png")
```

## Compare models

The ladder holds the models used in the benzene analysis:

```python
for name, spec in uqens.LADDER.items():
    result = uqens.fit(uqens.Problem(spec, data))
    print(name, round(result.neglogpost, 1), result.physics())
```

Lower `neglogpost` fits better, but choose models by how well they predict a held-out energy,
not by fit alone. The whole benchmark runs as a script: `python examples/benzene_let.py --sample 3000 --holdout`.
