# Gate 1: reproducing the published diffusion coefficient

**Question.** Richardson et al. 2026 ([arXiv:2603.06080](https://arxiv.org/abs/2603.06080)) report
D = 0.182 ± 0.002 Å²/ps for liquid benzene at 290 K from these LET data. Which analysis
choice separates that number from what uqens finds on the same files?

**Answer so far.** None of the choices the paper describes. Applied to the files in `data/`,
they give D ≈ 0.13 Å²/ps, and the rotational anisotropy runs to the edge of its prior.
The difference has to come from something the paper does not describe (see the questions
at the end).

Reproduce: `python examples/gate1_published_setup.py` (about 15 minutes; `--part B`
takes about 3).

## What the paper did

- 1.97 and 3.60 meV, ω within ±1.25 meV; Q 0.59–1.53 Å⁻¹ (14 bins) and 0.67–1.72 Å⁻¹ (12 bins).
- Empty can subtracted; the sample frozen at 260 K used as the resolution, by numerical convolution.
- Fickian translation × anisotropic rotation to l = 2 (R = 2.48 Å).
- One scale per energy × exp(−Q²⟨u²⟩/3), ⟨u²⟩ shared; a flat background per Q; no can term;
  detailed balance not mentioned.
- Nested sampling with uniform priors. Result: D = 0.182 ± 0.002 Å²/ps, D_s/D_t ≈ 11.

This repository has the empty can and the 260 K run only at 3.60 meV. At 1.97 meV the can
stays in the data and the resolution stays analytic, so Part A is the closest two-energy
version possible here and Part B repeats the full recipe at 3.60 meV alone.

## Part A: 1.97 + 3.60 meV, can not subtracted

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

## Part B: 3.60 meV only, the paper's full recipe

| Configuration | D (Å²/ps) | D_s/D_t | χ²/N |
|---|---|---|---|
| uqens choices, can not subtracted | 0.127 | 1.00 | 1.04 |
| uqens choices, can subtracted | 0.140 | 1.00 | 1.03 |
| **paper's choices** | **0.134** | 0.01 (D_s at its lower bound) | 1.41 |
| paper's, but analytic resolution | 0.095 | 0.01 | 1.32 |
| paper's, but a free scale per Q-bin | 0.143 | 1.00 | 1.10 |
| paper's, but rotation to the sum rule | 0.143 | 6.3 | 1.17 |
| paper's, but jump diffusion | 0.197 | 1.00 | 1.28 |

## What this shows

1. With the paper's recipe, these files give D = 0.134 at 3.60 meV, 26% below 0.182, and a
   worse fit than uqens's choices (χ²/N 1.41 against 1.03). The anisotropy goes the other way
   from the paper: tumbling fast, spinning pinned at the bottom of its prior.
2. No choice the paper lists moves D up towards 0.182. The only change that does is a different
   model, jump diffusion (0.197 at 3.60 meV alone), which the paper did not use.
3. Expanding rotation to the sum rule, instead of stopping at l = 2, turns the anisotropy into
   spinning faster than tumbling (D_s/D_t ≈ 6) on the same data, the direction the paper reports.
4. For the paper's model to reach 0.182 on these files, Q would have to be about 14% lower than
   the axis used here (D scales roughly as 1/Q²). That axis puts the aluminium Bragg peak of the
   can within 0.01 Å⁻¹ of 2.687 Å⁻¹ (`tests/test_let.py`).

## Questions for the authors

1. Which reduced files were fitted? Are they the ones in `data/`, and were the empty can and
   the 260 K run also reduced at 1.97 meV?
2. How was Q assigned to each spectrum, and what were the Q-bin edges?
3. In the fitting code, is the Lorentzian width a half or a full width, and is it ħDQ² in meV
   or DQ² in ps⁻¹?
4. Could we have the fitting code, or the fitted model and data at one Q, to compare point by point?
