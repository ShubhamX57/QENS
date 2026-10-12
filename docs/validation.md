# Earning the result

A fit that looks good is not yet a result. uqens checks four things before a number is
reported.

## Every optimum, not just one

The posterior can have several modes; for anisotropic rotation, for example, spinning faster
than tumbling and the reverse. `uqens.fit` starts from physics seeds covering both orderings
plus random draws from the prior, and reports every distinct optimum within Δχ² = 100 of the
best (`FitResult.modes`). On the benzene data one random start stops at a worse point
(−log posterior 6253.7 against 6088.4); with one start, that would have been the answer.

## Uncertainties from samples

`uqens.sample` runs an ensemble sampler (emcee) from the best fit and returns a `Posterior`
with medians, intervals, autocorrelation times and R̂. Check `Posterior.converged` (chain at
least 50 autocorrelation times long, R̂ below 1.1) before quoting numbers.

Error bars from the curvature at the best fit are not used. On 50 synthetic benzene datasets
they ranged from 1.2% to 7.6% for D between statistically identical replicas and held the
true value only 81% of the time at 95%, while sampled widths matched the actual scatter.

## Prediction of an energy left out

`uqens.validate.holdout` fits the physics on all incident energies but one, then predicts the
one left out with only the instrument terms refitted. The `penalty` is the χ² lost by fixing
the physics; `plot.holdout` shows the prediction on the spectra. On benzene, jump diffusion with
rotation predicts held-out energies best, and Fickian translation with rotation stopped at
l = 2 predicts worst of the three models tested.

## Coverage on synthetic data

`uqens.validate.coverage` simulates datasets from a fitted model with the real grids and error
bars, refits each from a start 3σ away from the truth, and counts how often the intervals
contain the true values. Starting at the truth flatters the error bars, because the optimiser
stops early, so starts are offset by default.

| Check on the benzene benchmark | Result |
|---|---|
| Fitted D scatter vs sampled width (50 replicas) | 2.6% vs 2.9% |
| True D inside the 68% / 95% interval | 66% / 100% (chance allows 54–80% / 88–100%) |
| Release target (200 replicas) | 62–74% inside 68% intervals, 92–98% inside 95% |

## Tests that guard the benchmark

`tests/test_model.py::test_benchmark_likelihood_is_unchanged` fails if anything in the chain
(reader, binning, error audit, forward model, solver) changes the benzene posterior. A change
meant to move it has to say so and update the stored value in the same commit.
