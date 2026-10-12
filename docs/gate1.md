# Gate 1: reproducing the published diffusion coefficient

**Question.** Richardson et al. 2026 ([arXiv:2603.06080](https://arxiv.org/abs/2603.06080)) report
D = 0.182 ± 0.002 Å²/ps for liquid benzene at 290 K from these LET data. Which analysis
choice separates that number from what uqens finds on the same files?

**Answer.** Two lines in the published fitting code, not the analysis choices. The code's
model energy axis is stretched by about 1.2, which inflates every fitted rate by 21–22%,
and its "±1.25 meV" window is really ±0.82 meV. Corrected for the axis alone, the published
0.182 becomes about 0.150 Å²/ps; with the stated window as well, the 3.60 meV fit gives
0.144. See [What the published code does](#what-the-published-code-does).

Reproduce: `python examples/gate1_published_setup.py` (about 15 minutes; `--part B`
takes about 3) and `python examples/gate1_replicate_published.py` (needs `pip install scipp`;
a few minutes per variant).

## What the published code does

The analysis code is public at
[github.com/scams-research/fitting-qens](https://github.com/scams-research/fitting-qens)
(commit `8f9fcd3`, MIT licence). Its 3.60 meV fitting model, copied into
`examples/gate1_replicate_published.py` with the same equations, grids and binning and run on
the files in `data/`, gives D = 0.195 Å²/ps,
close to the published 0.182 (which also uses 1.97 meV). So the files are equivalent, and two
lines explain the difference:

1. **The model's energy axis is stretched by about 1.2.** In
   `experiment_nested_sampling/{aniso,iso}/runner.py` and `sampler.py`, the model is evaluated on
   `np.linspace(-2, 2, 2N - 1)`, a spacing of 0.0220 meV at 3.60 meV and 0.0119 meV at 1.97 meV,
   and then convolved point by point with data and a resolution sampled every 0.018 and
   0.00985 meV. Every fitted rate (D, D_s, D_t) comes out 22% (3.60 meV) and 21% (1.97 meV) too
   large, with exactly the same χ², so no fit statistic can reveal it.
2. **The window is ±0.82 meV, not ±1.25 meV.** `plet_data.py` line 50 computes bin centres as
   `self.omega[:-1] + self.omega[1:] / 2` (missing parentheses), which gives 1.5ω + 0.009 meV.
   The window mask, and the energy axis of the published spectrum figures, use these values.

The Q axis (bin edges 0.05409 + 0.0115 n Å⁻¹ at 3.60 meV) agrees with the one used here to
0.0015 Å⁻¹, and line widths are half-widths converted with ħ, so neither contributes.

| Published fit at 3.60 meV, on the files in `data/` | D (Å²/ps) | D_s/D_t | χ²/N |
|---|---|---|---|
| as published | 0.195 | 26 | 1.92 |
| model on the data's own energy spacing | 0.160 | 26 | 1.92 |
| … and the stated ±1.25 meV window | 0.144 | 10 | 2.56 |

The fixes are one line each: compute bin centres as `(omega[:-1] + omega[1:]) / 2`, and build
the model grid with the data's own spacing, `step * (np.arange(2 * n - 1) - (n - 1))`. Corrected,
the published D sits near the uqens reconstruction of the paper's recipe (0.134, below) and the
MD value (0.142), and about 17% below PFG-NMR. The joint 1.97 + 3.60 meV number needs the
paper's own 1.97 meV empty-can and 260 K files, which are on
[Zenodo](https://doi.org/10.5281/zenodo.20412682) but not in this repository.

## The paper's choices, rebuilt in uqens

Before the code was checked, each choice the paper describes was rebuilt in uqens and switched on
one at a time. This showed that the gap could not come from those choices.

### What the paper says it did

- 1.97 and 3.60 meV, ω within ±1.25 meV; Q 0.59–1.53 Å⁻¹ (14 bins) and 0.67–1.72 Å⁻¹ (12 bins).
- Empty can subtracted; the sample frozen at 260 K used as the resolution, by numerical convolution.
- Fickian translation × anisotropic rotation to l = 2 (R = 2.48 Å).
- One scale per energy × exp(−Q²⟨u²⟩/3), ⟨u²⟩ shared; a flat background per Q; no can term;
  detailed balance not mentioned.
- Nested sampling with uniform priors. Result: D = 0.182 ± 0.002 Å²/ps, D_s/D_t ≈ 11.

This repository has the empty can and the 260 K run only at 3.60 meV. At 1.97 meV the can
stays in the data and the resolution stays analytic, so Part A is the closest two-energy
version possible here and Part B repeats the full recipe at 3.60 meV alone.

### Part A: 1.97 + 3.60 meV, can not subtracted

Each row adds one of the paper's choices to the one above; the "only" rows change one
choice at a time.

| Configuration | D (Å²/ps) | D_s/D_t | χ²/N (1.97, 3.60) |
|---|---|---|---|
| uqens choices | 0.140 | 1.00 | 1.10, 1.04 |
| + one scale per energy × exp(−Q²⟨u²⟩/3) | 0.122 | 0.01 (D_s at its lower bound) | 1.15, 1.26 |
| + no can term | 0.070 | 0.99 | 1.22, 1.45 |
| + no detailed balance | 0.071 | 1.00 | 1.23, 1.47 |
| + 260 K sample as resolution (3.60 meV) | 0.105 | 0.10 | 1.22, 1.55 |
| only one scale per energy | 0.122 | 0.01 | 1.15, 1.26 |
| only no can term | 0.088 | 1.00 | 1.19, 1.25 |
| only no detailed balance | 0.141 | 1.00 | 1.11, 1.07 |
| only 260 K resolution | 0.134 | 1.00 | 1.10, 1.00 |

Without the can subtracted, dropping the can term pulls D down sharply: the can's elastic
line is then fitted by a narrower translational line. That is why Part B matters.

### Part B: 3.60 meV only, the paper's full recipe

| Configuration | D (Å²/ps) | D_s/D_t | χ²/N |
|---|---|---|---|
| uqens choices, can not subtracted | 0.127 | 1.00 | 1.04 |
| uqens choices, can subtracted | 0.140 | 1.00 | 1.03 |
| **paper's choices** | **0.134** | 0.01 (D_s at its lower bound) | 1.41 |
| paper's, but analytic resolution | 0.095 | 0.01 | 1.32 |
| paper's, but a free scale per Q-bin | 0.143 | 1.00 | 1.10 |
| paper's, but rotation to the sum rule | 0.143 | 6.3 | 1.17 |
| paper's, but jump diffusion | 0.197 | 1.00 | 1.28 |

### What this showed

1. With the paper's recipe, these files give D = 0.134 at 3.60 meV, 26% below 0.182, and a
   worse fit than uqens's choices (χ²/N 1.41 against 1.03). The anisotropy goes the other way
   from the paper: tumbling fast, spinning pinned at the bottom of its prior.
2. No choice the paper lists moves D up towards 0.182. The only change that does is a different
   model, jump diffusion (0.197 at 3.60 meV alone), which the paper did not use.
3. Expanding rotation to the sum rule, instead of stopping at l = 2, turns the anisotropy into
   spinning faster than tumbling (D_s/D_t ≈ 6) on the same data, the direction the paper reports.
4. Nothing in the described recipe could explain the gap; the published code
   (above) does.

## Still open

1. The authors to confirm the two issues and rerun the joint 1.97 + 3.60 meV fit with both fixed.
2. The paper's 1.97 meV empty-can and 260 K files (on Zenodo), so the corrected joint fit can
   also be run here.
