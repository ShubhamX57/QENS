# Changelog

## 0.2.0.dev0 (branch `v2`)

A new core, replacing v0.1.

- **Reader:** the constant-Q axis of LET files is recovered and tested against the aluminium
  Bragg peak; can Bragg peaks masked; error bars audited against the scatter and rescaled.
- **Model:** one forward model for every incident energy and Q at once; Fickian or jump
  translation; isotropic or symmetric-top rotation expanded until the sum rule is met;
  analytic, asymmetric resolution or a measured kernel; per-bin amplitudes, can term and
  background solved exactly.
- **Fitting:** several starts with every distinct optimum reported; posterior sampling with
  convergence diagnostics returned.
- **Validation:** leave-one-energy-out prediction and synthetic coverage tests.
- **Plots:** spectra, misfit maps, held-out predictions and a timescale map.
- **Gate 1:** the published LET analysis rebuilt option by option; see {doc}`gate1`.
- **Engineering:** src layout, tests in CI that can fail, distribution name `uqens`.

## 0.1.0 (tag `v0.1.0-legacy`)

The original per-energy library. Its LET reader pairs the 212 detector-ring angles with the
249 constant-Q rows, so Q is wrong by a median 42%; results produced with it should not be
used.
