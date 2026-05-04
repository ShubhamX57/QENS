"""
MCMC over a registered forward model.

"""
from __future__ import annotations

import numpy as np

from .config  import Config
from .fitting import log_posterior
from .models  import get_model

__all__ = ["run_mcmc", "summarise", "summarise_samples", "gelman_rubin"]





# diagnostics

def gelman_rubin(chains: list[np.ndarray]) -> float:
    """
    Gelman-Rubin :math:'\\hat R' for a list of 1-D chains.

    ''< 1.01'' is well-converged; ''> 1.1'' indicates problems.

    """
    m = len(chains)
    n = min(len(c) for c in chains)
    chains = [np.asarray(c[:n]) for c in chains]
    w = np.mean([c.var(ddof=1) for c in chains])
    if w == 0.0:
        return float("nan")
    chain_means = np.array([c.mean() for c in chains])
    b = n * chain_means.var(ddof=1)
    var_hat = (1 - 1 / n) * w + b / n
    return float(np.sqrt(var_hat / w))







# emcee path

def _initial_ball(p_map: np.ndarray, n_walkers: int, prior_lo, prior_hi, rng,
                  fractional_jitter: float = 0.03,
) -> np.ndarray:
    """
    
    Initial Gaussian ball around MAP, clipped into the prior box."""
    p_map = np.asarray(p_map, dtype=float)
    lo = np.asarray(prior_lo, dtype=float)
    hi = np.asarray(prior_hi, dtype=float)
    pad = 1e-6 * (hi - lo)

    def one():
        p = p_map * (1 + rng.normal(0, fractional_jitter, p_map.size))
        # nudge anything that landed at/below 0 up to a small positive value
        p = np.where(p <= 0, np.maximum(p_map * 0.1, lo + pad), p)
        return np.clip(p, lo + pad, hi - pad)

    return np.array([one() for _ in range(n_walkers)])






def _run_emcee(data_bins, sigma_res, p_map, model, cfg, extras, verbose,
):
    import emcee
    fm = get_model(model)
    rng = np.random.default_rng(cfg.random_seed)
    p0 = _initial_ball(p_map, cfg.n_walkers, fm.prior_lo, fm.prior_hi, rng)

    def log_prob(p):
        return log_posterior(p, data_bins, sigma_res, model=model, **extras)

    sampler = emcee.EnsembleSampler(cfg.n_walkers, fm.n_params, log_prob)
    total = cfg.n_warmup + cfg.n_keep
    if verbose:
        print(f"  emcee: {cfg.n_walkers} walkers × {total} steps "
              f"(thin={cfg.thin}, n_dim={fm.n_params}, model={model!r})")
    sampler.run_mcmc(p0, total, progress=False)
    samples = sampler.get_chain(discard=cfg.n_warmup, thin=cfg.thin, flat=True)

    if verbose:
        print(f"  acceptance fraction: "
              f"{np.mean(sampler.acceptance_fraction):.3f}")
        try:
            tau = sampler.get_autocorr_time(quiet=True)
            print(f"  autocorrelation time: " +
                  "  ".join(f"{t:.1f}" for t in tau))
        except Exception:
            pass
        print(f"  total samples kept: {len(samples)}")
    return samples







# MH fallback

def _run_mh(data_bins, sigma_res, p_map, model, cfg, extras, verbose):
    fm = get_model(model)
    rng_global = np.random.default_rng(cfg.random_seed)
    n_dim = fm.n_params
    p_map = np.asarray(p_map, dtype=float)
    step = np.maximum(np.abs(p_map) * 0.05, 1e-6)
    n_total = cfg.n_warmup + cfg.n_keep

    def chain(start, seed):
        rng = np.random.default_rng(seed)
        cur = start.copy()
        cur_lp = log_posterior(cur, data_bins, sigma_res,
                               model=model, **extras)
        out = [cur.copy()]
        n_acc = 0
        for _ in range(n_total):
            new = cur + rng.normal(0, step, n_dim)
            new_lp = log_posterior(new, data_bins, sigma_res,
                                   model=model, **extras)
            if np.log(rng.random() + 1e-300) < new_lp - cur_lp:
                cur, cur_lp = new, new_lp
                n_acc += 1
            out.append(cur.copy())
        return np.array(out), n_acc / n_total

    n_chains = 4
    chains = []
    if verbose:
        print(f"  MH fallback: {n_chains} chains × {n_total} steps "
              f"(thin={cfg.thin}, n_dim={n_dim})")
    for cid in range(n_chains):
        start = p_map * (1 + rng_global.normal(0, 0.03, n_dim))
        ch, acc = chain(start, cfg.random_seed + cid)
        chains.append(ch[cfg.n_warmup::cfg.thin])
        if verbose:
            print(f"    chain {cid+1}: acceptance={acc:.3f}  "
                  f"kept={len(chains[-1])}")
    rhats = [gelman_rubin([c[:, i] for c in chains]) for i in range(n_dim)]
    if verbose:
        print(f"  Gelman-Rubin R̂: " + "  ".join(f"{r:.4f}" for r in rhats))
    return np.vstack(chains)







# top-level entry point

def run_mcmc(data_bins,
             sigma_res,
             p_map,
             model: str = "anisotropic_rotor",
             cfg: Config | None = None,
             verbose: bool = True,
             **extras,
) -> np.ndarray:
    """Run MCMC over a registered forward model.

    Parameters
    ----------
    data_bins : list
        From :func:'qens.fitting.build_data_bins'.

    sigma_res : float | array | list[array]
        Resolution: scalar Gaussian σ in meV, single measured kernel, or
        one kernel per Q-bin.

    p_map : array
        MAP starting point (from :func:'qens.fitting.find_map').

    model : str
        Registered forward-model name.

    cfg : Config

    verbose : bool

    extras :
        Forwarded to the model's ''predict'' callable.


    Returns
    -------
    samples : ndarray of shape ''(n_kept, n_dim)''

    """
    if cfg is None:
        cfg = Config()
    try:
        import emcee  # noqa: F401
        return _run_emcee(data_bins, sigma_res, p_map, model, cfg,
                          extras, verbose)
    except ImportError:
        if verbose:
            print("  emcee not found — using Metropolis-Hastings fallback "
                  "(consider: pip install emcee)")
        return _run_mh(data_bins, sigma_res, p_map, model, cfg,
                       extras, verbose)






# summarisation

def summarise(arr: np.ndarray, label: str = "", verbose: bool = True
              ) -> tuple[float, float, float]:
    """
    Median and 95% credible interval for a single parameter chain.
    
    """
    arr = np.asarray(arr)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    med = float(np.median(arr))
    if verbose and label:
        print(f"    {label:<24}  median={med:.5f}   "
              f"95% CI=[{lo:.5f}, {hi:.5f}]")
    return med, float(lo), float(hi)




def summarise_samples(samples: np.ndarray,
                      model: str = "anisotropic_rotor",
                      derived: dict | None = None,
                      verbose: bool = True,
) -> dict[str, tuple[float, float, float]]:
    """
    Per-parameter median + 95% CI for a registered model.

    Parameters
    ----------
    samples : ndarray, shape (n, n_dim)
    
    model : str
        Registered model name (used to look up parameter names).

    derived : dict, optional
        ''{label: callable(samples) -> 1-D array}'' — extra derived
        quantities to summarise (e.g. ''D_s / D_t'' for anisotropic).

    verbose : bool

    
    Returns
    -------
    dict[label, (median, lo95, hi95)]


    """
    fm = get_model(model)
    out: dict[str, tuple[float, float, float]] = {}
    for i, name in enumerate(fm.param_names):
        out[name] = summarise(samples[:, i], name, verbose=verbose)
    if derived:
        for label, fn in derived.items():
            out[label] = summarise(fn(samples), label, verbose=verbose)
    return out
