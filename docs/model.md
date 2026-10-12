# The model

uqens fits one physical model to every incident energy and every Q at once. This page
says exactly what that model is.

## What is fitted

In Q-bin $b$ the measured spectrum is modelled as

$$
I_b(\omega) = A_b \left[ S(Q,\omega) \otimes R \right](\omega)\, e^{\hbar\omega/2k_BT}
+ E_b\, R(\omega) + B_b ,
$$

with energy loss counted positive. $A_b$ (scale), $E_b$ (a resolution-shaped elastic term,
which picks up can scattering) and $B_b$ (flat background) are all non-negative and are
solved exactly at every likelihood evaluation, so the sampler only sees the physics and the
resolution parameters. With `amplitudes="global"`, $A_b$ is replaced by one scale per energy
times $e^{-Q^2\langle u^2\rangle/3}$, and `use_el=False` drops $E_b$.

## Translation × rotation

$$
S(Q,\omega) = \sum_c W_c(Q)\, L\!\left(\omega;\ \Gamma_t(Q) + \Gamma_c\right),
\qquad L(\omega;\Gamma) = \frac{1}{\pi}\frac{\Gamma}{\omega^2+\Gamma^2}.
$$

Translation sets the width $\Gamma_t$:

| `translation` | $\Gamma_t(Q)$ |
|---|---|
| `"fickian"` | $\hbar D Q^2$ |
| `"jump"` (Singwi–Sjölander) | $\hbar D Q^2 / (1 + D Q^2 \tau_0)$ |

Rotation of a scatterer on a sphere of radius $R$ (2.48 Å for the hydrogens of benzene)
adds channels $c = (l, |m|)$ with

$$
W_c(Q) = (2l+1)\, j_l^2(QR)\, c_{lm}(\theta_0), \qquad
\Gamma_c = \hbar\left[l(l+1) D_t + m^2 (D_s - D_t)\right],
$$

where $D_t$ is the tumbling rate, $D_s$ the spinning rate and $c_{lm}$ the weight of each $|m|$
for a scatterer at angle $\theta_0$ from the symmetry axis (summing to 1 over $m$). With
`rotation="iso"` there is one rate, $D_r = D_t = D_s$.

The weights obey the sum rule $\sum_l (2l+1) j_l^2(QR) = 1$, and uqens keeps adding $l$
until it is met (`lmax=None`). Stopping at $l = 2$, as many analyses do, keeps only about a
tenth of the rotational intensity at $Q = 2.5$ Å⁻¹; `lmax=2` reproduces that choice when you
need to compare.

## Resolution

**Analytic** (default). A Voigt profile: Gaussian standard deviation $\sigma_E$ (one per
incident energy) and Lorentzian half-width $\kappa\sigma_E$ (shared). With `asym=True`, a second,
shifted and wider component carries a fraction $f_2$ of the area, describing the asymmetric
tail of a chopper spectrometer. A Lorentzian convolved with a Voigt is again a Voigt, so the
convolution is exact; wide energy bins are averaged with two-point Gauss–Legendre.

**Measured** (`resolution="measured"`). The spectrum of a vanadium or frozen-sample run,
averaged over exactly the same rows and energies as the data, becomes each Q-bin's
convolution kernel:

```python
uqens.let.attach_resolution(data[2], "data/benzene_260_360_inc.nxspe",
                            background="data/empty_360_inc.nxspe")
spec = uqens.ModelSpec("measured", resolution="measured")
```

Energies without a kernel keep the analytic resolution. A frozen sample that still moves
(benzene at 260 K reorients) makes the kernel too broad.

## Parameters and priors

| Parameter | Prior |
|---|---|
| `lnD` | uniform, D from 0.01 to 2 Å²/ps |
| `lntau0` (jump only) | uniform, τ₀ from 0.01 to 20 ps |
| `lnDr`, or `lnDt` and `lnDs` | uniform, 0.003 to 5 ps⁻¹ |
| `lnsig_<Ei>` | normal around the instrument's width $\sigma_0(E_i)$, sd 0.3 in log, within ±1.5 |
| `e0_<Ei>` (elastic-line offset) | normal, mean 0, sd $\sigma_0/4$, within $\pm\sigma_0$ |
| `kappa` | uniform, 0 to 3 |
| `f2`, `d2`, `s2` (asymmetric resolution) | uniform: 0–0.6, −4 to 4, 0.7–4 |
| `lnC_<Ei>`, `u2` (global amplitudes only) | uniform: scale 10⁻⁴ to 10⁴; ⟨u²⟩ 0 to 0.5 Å² |

$\sigma_0(E_i)$ comes from PyChop's elastic widths for LET (high-flux mode, 200 Hz), scaled to
the width fitted at 3.60 meV. It is a prior, not a fixed value.

## Reading LET files

The LET `.nxspe` files hold S(|Q|, ω) in 249 constant-Q rows but do not store the Q axis.
`uqens.let` uses the axis recovered from the data,

$$Q_n = k_i\,(0.0399 + 0.00873\,(n + \tfrac12)),$$

which puts the aluminium (111) Bragg peak of the can at 2.687 Å⁻¹ (a test checks this). The
reader then drops cells at the kinematic edge and rows near the can's Bragg peaks, averages
neighbouring rows into Q-bins of about 0.1 Å⁻¹, and compares the error bars with the
point-to-point scatter of the data. Neighbouring rows share counts, so the scatter is larger
than the propagated errors, and the error bars are scaled up to match (by 9–23% for benzene).
