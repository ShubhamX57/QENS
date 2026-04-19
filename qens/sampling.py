"""
sampling.py — MCMC posterior sampling for the CE diffusion model
================================================================

Entry point
-----------
``run_mcmc`` — preferred function.  Uses emcee (affine-invariant ensemble
sampler) if installed; falls back to a hand-written Metropolis-Hastings
implementation if emcee is unavailable.

Sampler choice
--------------
emcee (default):
    An affine-invariant ensemble sampler.  32 walkers propose moves based on
    the positions of other walkers ("stretch move"), making it efficient on
    correlated posteriors.  The (D, l) posterior is correlated because τ =
    l²/(6D) ties the two parameters together.  Standard Metropolis would need
    careful tuning of the proposal covariance; emcee adapts automatically.

Metropolis-Hastings (fallback):
    4 independent chains with Gaussian proposals on D and log(l).  The
    log-normal proposal on l keeps it positive without explicit truncation.
    Convergence is checked via the Gelman-Rubin R-hat statistic.

Convergence diagnostics
-----------------------
``gelman_rubin`` — R-hat statistic.  R-hat < 1.1 indicates that all chains
are sampling from the same distribution.  R-hat > 1.1 means the chains have
not mixed — increase n_warmup or check for multimodality.

Posterior samples format
------------------------
The returned ``samples`` array has shape (N_samples, 2):
    samples[:, 0] = D values (Å²/ps)
    samples[:, 1] = l values (Å)   — absolute value enforced

Derived quantities (e.g. τ = l²/6D) are computed from these samples in
``plot_posteriors`` and ``summarise``, propagating the full D-l correlation.
"""

from __future__ import annotations

import numpy as np

from .fitting import log_posterior
from .config  import Config


# ── Convergence diagnostic ────────────────────────────────────────────────────

def gelman_rubin(chains: list) -> float:
    """
    Compute the Gelman-Rubin R-hat convergence statistic for a single parameter.

    R-hat compares the variance within individual chains to the variance
    between chain means.  Values close to 1.0 indicate convergence; values
    above 1.1 suggest the chains have not yet explored the same region.

    Formula
    -------
        W   = mean within-chain variance
        B   = n × variance of chain means   (between-chain variance × n)
        V̂   = ((n−1)/n) · W + B/n
        R̂   = √(V̂ / W)

    Parameters
    ----------
    chains : list of array-like
        Each element is a 1D array of posterior draws for one parameter from
        one chain.  All chains are truncated to the length of the shortest.

    Returns
    -------
    float
        R-hat.  Returns nan if within-chain variance is zero (all chains stuck).

    References
    ----------
    Gelman & Rubin (1992) "Inference from iterative simulation using multiple
    sequences", Statistical Science.
    """
    m = len(chains)                              # number of chains
    n = min(len(c) for c in chains)             # truncate to shortest chain
    chains = [np.asarray(c[:n]) for c in chains]

    # Within-chain variance (averaged across chains)
    w = np.mean([c.var(ddof=1) for c in chains])

    if w == 0.0:
        # All chains are stuck — can't compute a meaningful R-hat
        return float("nan")

    # Between-chain variance (variance of chain means, scaled by n)
    chain_means = np.array([c.mean() for c in chains])
    b           = n * chain_means.var(ddof=1)

    # Pooled variance estimate
    var_hat = (1 - 1 / n) * w + b / n

    return float(np.sqrt(var_hat / w))


# ── emcee sampler (primary) ───────────────────────────────────────────────────

def _run_emcee(data_bins, sr, d_map, l_map, cfg):
    """
    Run the emcee affine-invariant ensemble MCMC sampler.

    Walkers are initialised with 5% Gaussian scatter around the MAP point
    and clipped to remain inside the prior box.  After running
    n_warmup + n_keep steps, the warmup is discarded and the remaining
    samples are thinned by ``cfg.thin``.

    Parameters
    ----------
    data_bins : list
        Output of ``build_data_bins``.
    sr : float
        Instrument resolution sigma (meV).
    d_map, l_map : float
        MAP estimates used to initialise walkers.
    cfg : Config

    Returns
    -------
    numpy.ndarray, shape (N_samples, 2)
        Posterior samples with columns [D, l].
    """
    import emcee

    def log_prob(params):
        """Wrapper around log_posterior for emcee (takes a single array)."""
        return log_posterior(params[0], abs(params[1]), data_bins, sr)

    ndim = 2  # number of free parameters: D and l
    rng  = np.random.default_rng(cfg.random_seed)

    # Initialise walkers: cluster around MAP with 5% spread, clip to prior
    p0 = [
        np.array([d_map, l_map]) * (1 + rng.normal(0, 0.05, ndim))
        for _ in range(cfg.n_walkers)
    ]
    # Clip all walkers into the prior box to avoid -inf at step 0
    p0 = [np.clip(p, [1e-3, 0.6], [2.9, 5.9]) for p in p0]

    sampler     = emcee.EnsembleSampler(cfg.n_walkers, ndim, log_prob)
    total_steps = cfg.n_warmup + cfg.n_keep

    print(
        f"  running emcee: {cfg.n_walkers} walkers × {total_steps} steps "
        f"(thin={cfg.thin})"
    )
    sampler.run_mcmc(p0, total_steps, progress=True)

    # Discard burn-in and thin the chain
    samples = sampler.get_chain(discard=cfg.n_warmup, thin=cfg.thin, flat=True)

    # Acceptance fraction — healthy range is 0.2–0.5 for emcee stretch moves
    acc = float(np.mean(sampler.acceptance_fraction))
    print(f"  acceptance fraction: {acc:.3f}")

    # Autocorrelation time — ideally thin ≥ τ/2
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"  autocorrelation time: D={tau[0]:.1f}  l={tau[1]:.1f} steps")
    except Exception:
        print("  autocorrelation estimate failed (chain may be too short)")

    return samples


# ── Metropolis-Hastings fallback ──────────────────────────────────────────────

def _run_mh(data_bins, sr, d_map, l_map, cfg):
    """
    Run 4 independent Metropolis-Hastings chains (fallback when emcee is absent).

    Proposal distributions
    ----------------------
    D : Gaussian with std = d_map × 0.1
    l : log-normal  (propose log(l) + Normal(0, 0.1)), ensures l > 0 always.
        The Jacobian correction (log_l_new − log_l_cur) is applied in the
        acceptance ratio to maintain detailed balance.

    Parameters
    ----------
    data_bins : list
        Output of ``build_data_bins``.
    sr : float
        Instrument resolution sigma (meV).
    d_map, l_map : float
        MAP estimates used to start each chain (with ±5% scatter).
    cfg : Config

    Returns
    -------
    numpy.ndarray, shape (N_samples, 2)
        Stacked posterior samples from all 4 chains after thinning.
    """
    rng_global = np.random.default_rng(cfg.random_seed)

    def _chain(start, n_steps, step_d, step_log_l, seed):
        """
        Run a single Metropolis-Hastings chain.

        Parameters
        ----------
        start     : array-like [D, l] — starting position
        n_steps   : total steps to run (including warmup)
        step_d    : proposal std for D
        step_log_l: proposal std for log(l)
        seed      : integer RNG seed for this chain

        Returns
        -------
        samples : ndarray, shape (n_steps+1, 2)
        acc_frac: float — fraction of proposals accepted
        """
        rng_c         = np.random.default_rng(seed)
        d_cur, l_cur  = float(start[0]), float(start[1])
        cur_lp        = log_posterior(d_cur, l_cur, data_bins, sr)
        samples       = [(d_cur, l_cur)]
        n_acc         = 0

        for _ in range(n_steps):
            # Gaussian proposal for D
            d_new = d_cur + rng_c.normal(0, step_d)

            # Log-normal proposal for l: propose in log-space to keep l > 0
            log_l_new = np.log(l_cur) + rng_c.normal(0, step_log_l)
            l_new     = np.exp(log_l_new)

            new_lp = log_posterior(d_new, l_new, data_bins, sr)

            # Metropolis-Hastings acceptance ratio including Jacobian for log(l)
            log_accept = new_lp - cur_lp + (log_l_new - np.log(l_cur))

            if np.log(rng_c.random() + 1e-300) < log_accept:
                # Accept proposal
                d_cur, l_cur = d_new, l_new
                cur_lp       = new_lp
                n_acc       += 1

            samples.append((d_cur, l_cur))

        return np.array(samples), n_acc / n_steps

    # Proposal step sizes based on MAP values
    step_d     = d_map * 0.1
    step_log_l = 0.1
    n_total    = cfg.n_warmup + cfg.n_keep

    chains = []
    print(f"  running 4 MH chains × {n_total} steps (thin={cfg.thin})")

    for cid in range(4):
        # Small random perturbation around MAP for each chain
        start = np.array([d_map, l_map]) * (1 + rng_global.normal(0, 0.05, 2))
        start = np.clip(start, [1e-3, 0.6], [2.9, 5.9])

        chain, acc = _chain(start, n_total, step_d, step_log_l,
                            seed=cfg.random_seed + cid)

        # Discard warmup and thin
        chains.append(chain[cfg.n_warmup :: cfg.thin])
        print(f"  chain {cid+1}: acceptance={acc:.3f}  kept={len(chains[-1])}")

    # ── Gelman-Rubin convergence check ────────────────────────────────────────
    rhat_d = gelman_rubin([c[:, 0] for c in chains])
    rhat_l = gelman_rubin([c[:, 1] for c in chains])
    print(f"  R-hat: D={rhat_d:.4f}  l={rhat_l:.4f}")

    if rhat_d > 1.1 or rhat_l > 1.1:
        print(
            "  WARNING: R-hat > 1.1 — chains may not have converged.  "
            "Consider increasing n_warmup."
        )

    return np.vstack(chains)


# ── Public entry point ────────────────────────────────────────────────────────

def run_mcmc(
    data_bins: list,
    sr: float,
    d_map: float,
    l_map: float,
    cfg: Config | None = None,
) -> np.ndarray:
    """
    Run MCMC sampling and return posterior samples of (D, l).

    Automatically chooses emcee if installed, otherwise falls back to the
    hand-written Metropolis-Hastings implementation.

    Parameters
    ----------
    data_bins : list
        Output of ``build_data_bins``.
    sr : float
        Instrument resolution sigma (meV).
    d_map, l_map : float
        MAP estimates from ``find_map`` used to initialise walkers.
    cfg : Config or None

    Returns
    -------
    numpy.ndarray, shape (N_samples, 2)
        Posterior samples.  Column 0 = D (Å²/ps), column 1 = l (Å).
        l values are guaranteed positive (abs applied before returning).

    Notes
    -----
    Total samples ≈ n_walkers × n_keep / thin  for emcee,
                  ≈ 4 × n_keep / thin          for MH fallback.
    """
    if cfg is None:
        cfg = Config()

    # Try importing emcee; fall back gracefully
    try:
        import emcee        # noqa: F401
        use_emcee = True
    except ImportError:
        use_emcee = False
        print("  emcee not found — using Metropolis-Hastings fallback")
        print("  Install emcee for better performance: pip install emcee")

    if use_emcee:
        samples = _run_emcee(data_bins, sr, d_map, l_map, cfg)
    else:
        samples = _run_mh(data_bins, sr, d_map, l_map, cfg)

    # Enforce l > 0 (the sampler may explore negative l via the stretch move)
    samples[:, 1] = np.abs(samples[:, 1])

    print(f"  total posterior samples: {len(samples)}")
    return samples


# ── Summary helper ────────────────────────────────────────────────────────────

def summarise(arr: np.ndarray, label: str) -> tuple[float, float, float]:
    """
    Print and return the median and 95% credible interval for a parameter.

    The 95% credible interval [lo, hi] = [2.5th, 97.5th percentile] means:
    given this data and the prior, there is a 95% probability that the
    parameter lies within [lo, hi].  This is a genuinely probabilistic
    statement, unlike a frequentist confidence interval.

    Parameters
    ----------
    arr : numpy.ndarray
        1D array of posterior samples for one parameter.
    label : str
        Human-readable label printed in the summary line.

    Returns
    -------
    med : float — posterior median
    lo  : float — 2.5th percentile (lower 95% CI bound)
    hi  : float — 97.5th percentile (upper 95% CI bound)
    """
    lo, hi = np.percentile(arr, [2.5, 97.5])
    med    = float(np.median(arr))
    print(f"  {label:<16}  median={med:.5f}   95% CI=[{lo:.5f}, {hi:.5f}]")
    return med, float(lo), float(hi)
