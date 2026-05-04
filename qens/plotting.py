"""
Figures.

Each function here takes already processed inputs (datasets, samples,
HWHM table) and writes out a single figure. None of them re-compute the
physics — they're presentation-only. The forward-model evaluations are
done by :mod:'qens.models' and called by the inference layer.


All figures use the same colour scheme:


    elastic component     — blue   ''#1565c0''
    quasi-elastic         — orange ''#e67e22''
    fit / MAP             — red    ''#c0392b''
    posterior fan / data  — slate  ''#2471a3''
    resolution            — grey   ''#888''

    
Save by passing ''save_path''; if None the figure is returned for the
user to handle.


"""


from __future__ import annotations


from typing import Iterable


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors            import LogNorm, LinearSegmentedColormap
from scipy.ndimage                import gaussian_filter
from scipy.optimize               import nnls


from .models import (
    fickian_hwhm, ce_hwhm, ss_hwhm, get_model,
)


__all__ = [
    "plot_overview",
    "plot_sqw_map",
    "plot_per_q_fits",
    "plot_hwhm_vs_q2",
    "plot_posteriors",
    "plot_joint_posterior",
]


_SQW_CMAP = LinearSegmentedColormap.from_list(
    "qens",
    ["#0a0e1a", "#0c2d6b", "#1565c0", "#42a5f5",
     "#e3f2fd", "#ff8f00", "#e65100"],
    N=512,
)




def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)




def _save(fig, save_path):
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)


# overview of all loaded files

def plot_overview(dataset: dict, save_path: str | None = None):
    """
    One-line elastic-peak summary per loaded file. Useful sanity check
    after :func:'qens.preprocessing.assign_resolution'.
    """
    names = sorted(dataset)
    n = len(names)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for ax, name in zip(axes, names):
        d = dataset[name]
        good = d["good"]
        n_lo = max(2, len(good) // 5)
        avg = np.nanmean([d["data"][good[j]] for j in range(n_lo)], axis=0)
        avg = np.where(np.isfinite(avg), avg, 0.0)
        ewin = min(0.5 * d["ei"], 1.2)
        m = (d["e"] >= -ewin) & (d["e"] <= ewin)
        peak = avg[m].max()
        y = avg[m] / peak if peak > 0 else avg[m]
        col = ("#2ca02c" if d["temp"] <= 270
               else "#c0392b" if d["kind"] == "inc" else "#2471a3")
        ax.plot(d["e"][m], y, color=col, lw=1.5)
        ax.axvline(0, color="#aaa", lw=0.7, ls=":")
        ax.set_title(name, fontsize=8, color=col)
        ax.set_xlabel("ω (meV)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18)
        info = (f"E₀={d.get('e0', 0):+.3f} meV\n"
                f"FWHM={d.get('fwhm_res', 0)*1000:.0f} µeV "
                f"[{d.get('res_source', '?')}]")
        ax.text(0.03, 0.96, info, transform=ax.transAxes, va="top",
                fontsize=6.5,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        _despine(ax)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("loaded datasets   green=frozen   red=INC   blue=COH",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes



# 2 D S(Q,ω) map for one dataset
def plot_sqw_map(d: dict, ewin: float = 1.2, save_path: str | None = None):
    """
    Single S(Q,ω) heatmap, log-scaled.
    
    """
    g = d["good"]
    qg = d["q"][g]
    e = d["e"]
    em = (e >= -ewin) & (e <= ewin)
    img = d["data"][np.ix_(g, em)]
    img = np.where(np.isfinite(img) & (img > 0), img, np.nan)
    qs = np.argsort(qg)
    ism = gaussian_filter(np.where(np.isfinite(img[qs]), img[qs], 0.0),
                          sigma=[1.5, 0.8])
    ism[ism <= 0] = np.nan
    vmin = max(np.nanpercentile(ism, 2), 1e-8)
    vmax = np.nanpercentile(ism, 99)



    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")
    im = ax.pcolormesh(e[em], qg[qs], ism, cmap=_SQW_CMAP,
                       norm=LogNorm(vmin=vmin, vmax=vmax),
                       shading="auto", rasterized=True)
    ax.axvline(0, color="white", lw=1, ls="--", alpha=0.4)
    ax.set_xlabel("ω (meV)", color="white", fontsize=12)
    ax.set_ylabel("Q (Å⁻¹)", color="white", fontsize=12)
    ax.set_title(f"S(Q,ω)  —  {d.get('name','?')}",
                 color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cb.set_label("S(Q,ω)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight",
                    facecolor="#0d1117", dpi=150)
    return fig, ax




# per-Q-bin model-vs-data panel
def plot_per_q_fits(
    data_bins, sigma_res, params,
    model: str = "anisotropic_rotor",
    save_path: str | None = None, **extras,
):
    """
    Grid of subplots, one per Q-bin, showing data vs forward-model fit.
    
    """
    from .fitting import _resolve_kernel_for_bin
    fm = get_model(model)
    n = len(data_bins)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols, 2.8 * nrows),
                             squeeze=False)
    axes = axes.flatten()
    for ax, k, (omega, spec, errs, q) in zip(axes, range(n), data_bins):
        sr_k = _resolve_kernel_for_bin(sigma_res, k)
        shape = fm.predict(omega, q, params, sr_k, **{**fm.extras, **extras})
        basis = np.column_stack([shape, np.ones_like(shape)])
        amp, _ = nnls(basis / errs[:, None], spec / errs)
        fit = amp[0] * shape + amp[1]
        chi2r = np.sum(((spec - fit) / errs) ** 2) / max(len(spec) - 2, 1)
        ax.errorbar(omega, spec, yerr=errs, fmt=".", color="#222",
                    ms=2.5, elinewidth=0.5, alpha=0.6)
        ax.plot(omega, fit, "-", color="#c0392b", lw=1.6,
                label=f"χ²ᵣ={chi2r:.2f}")
        ax.axhline(amp[1], color="#7f8c8d", lw=0.6, ls=":", alpha=0.6)
        ax.set_title(f"Q = {q:.2f} Å⁻¹", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    pretty = ", ".join(f"{n}={v:.3f}"
                       for n, v in zip(fm.param_names, params))
    fig.suptitle(f"forward-model fit per Q-bin   model={model}   {pretty}",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes




# Γ(Q) vs Q^2 (legacy approach)
def plot_hwhm_vs_q2(
    q_centres, hwhm, hwhm_err,
    model_results: dict | None = None,
    samples: np.ndarray | None = None,
    map_params: tuple | None = None,
    res_hwhm_uev: float = 50,
    save_path: str | None = None,
):
    """
    Γ(Q) vs Q^2 with optional model curves and posterior fan.
    
    """


    q_fine = np.linspace(max(q_centres.min() * 0.85, 0.05),
                         q_centres.max() * 1.10, 350)
    q2f = q_fine ** 2
    fig, ax = plt.subplots(figsize=(8, 5.5))



    if samples is not None and map_params is not None and len(map_params) >= 2:
        # CE posterior fan, only meaningful for CE-shaped models
        d_s, l_s = samples[:, 0], np.abs(samples[:, 1])
        rng = np.random.default_rng(0)
        idx = rng.choice(len(d_s), min(300, len(d_s)), replace=False)
        try:
            fan = np.array([ce_hwhm(q_fine, d_s[i], l_s[i]) * 1000 for i in idx])
            ax.fill_between(q2f, np.percentile(fan, 2.5, axis=0),
                            np.percentile(fan, 97.5, axis=0),
                            alpha=0.18, color="#2471a3",
                            label=f"95% posterior  n={len(d_s)}")
        except Exception:
            pass

    if model_results:
        for name, res in model_results.items():
            if "error" in res:
                continue
            D = res.get("D", 0.30)
            try:
                if name == "ce":
                    L = res.get("l", 2.5)
                    y = ce_hwhm(q_fine, D, L) * 1000
                    label = (rf"CE  $D={D:.4f}$  $\ell={L:.3f}$")
                elif name == "fickian":
                    y = fickian_hwhm(q_fine, D) * 1000
                    label = rf"Fickian  $D={D:.4f}$"
                elif name == "ss":
                    ts = res.get("tau_s", 1.0)
                    y = ss_hwhm(q_fine, D, ts) * 1000
                    label = rf"SS  $D={D:.4f}$  $\tau_s={ts:.3f}$"
                else:
                    continue
                ax.plot(q2f, y, "-", lw=2.0, label=label)
            except Exception:
                continue



    ax.errorbar(q_centres ** 2, hwhm * 1000, yerr=2 * hwhm_err * 1000,
                fmt="o", color="#111", ms=6, capsize=3.5, elinewidth=1.4,
                label=r"data  $\pm 2\sigma$")
    ax.axhline(res_hwhm_uev, color="#888", ls=":", lw=1.3,
               label=rf"resolution HWHM = {res_hwhm_uev:.0f} µeV")
    ax.axhspan(0, res_hwhm_uev * 1.1, alpha=0.04, color="#888")
    ax.set_xlabel(r"$Q^2$  (Å$^{-2}$)", fontsize=12)
    ax.set_ylabel(r"$\Gamma(Q)$  (µeV)", fontsize=12)
    ax.set_title(r"$\Gamma(Q)$ vs $Q^2$  —  deviation from linearity is "
                 "non-Fickian", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(left=-0.05)
    ax.set_ylim(bottom=-10)
    ax.grid(alpha=0.18)
    _despine(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


# 1D marginal posteriors
def plot_posteriors(
    samples: np.ndarray,
    model: str = "anisotropic_rotor",
    reference_values: dict | None = None,
    derived: dict | None = None,
    save_path: str | None = None,
):
    """
    Histograms with median, 95% CI band and reference lines.

    Parameters
    ----------
    samples : ndarray, shape (n, n_params)

    model : str
        Registered model name (provides parameter names).

    reference_values : dict, optional
        ''{param_name: [(value, label), ...]}'' to plot vertical reference
        lines (e.g. literature values).

    derived : dict, optional
        ''{label: callable(samples)->array}'' for extra panels.

    """


    fm = get_model(model)
    panels: list[tuple[str, np.ndarray]] = [
        (n, samples[:, i]) for i, n in enumerate(fm.param_names)
    ]
    if derived:
        for name, fn in derived.items():
            panels.append((name, np.asarray(fn(samples))))


    n = len(panels)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.5 * rows),
                             squeeze=False)
    axes = axes.flatten()



    palette = ["#c0392b", "#7f8c8d", "#1e8449", "#2471a3",
               "#8e44ad", "#e67e22", "#16a085", "#d35400"]



    for ax, (name, arr), col in zip(axes, panels,
                                    palette * (1 + n // len(palette))):
        med = float(np.median(arr))
        lo, hi = np.percentile(arr, [2.5, 97.5])
        cnt, _, _ = ax.hist(arr, bins=60, density=True,
                            color=col, alpha=0.78,
                            edgecolor="white", lw=0.3)
        pk = cnt.max()
        ax.axvspan(lo, hi, alpha=0.16, color=col)
        ax.axvline(med, color="black", lw=1.8,
                   label=f"median = {med:.4f}")
        if reference_values and name in reference_values:
            for val, lab in reference_values[name]:
                ax.axvline(val, color="#1565c0", lw=1.7, ls=":",
                           label=f"{lab} = {val}")
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel("density", fontsize=9)
        ax.set_title(name, fontsize=10, color=col, fontweight="bold")
        ax.legend(fontsize=7.5)
        ax.set_ylim(0, pk * 1.25)
        ax.grid(alpha=0.18)
        _despine(ax)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"posteriors — {model} ({len(samples)} samples)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes


# joint posterior for a pair of parameters
def plot_joint_posterior(
    samples: np.ndarray,
    indices: tuple[int, int] = (0, 1),
    labels: tuple[str, str] = ("p₁", "p₂"),
    map_point: tuple[float, float] | None = None,
    reference_point: tuple[float, float] | None = None,
    save_path: str | None = None,
):
    """
    Scatter+contour for any two parameter columns of ''samples''.
    
    """


    i, j = indices
    x = samples[:, i]
    y = samples[:, j]


    fig, ax = plt.subplots(figsize=(7, 6))
    n_plot = min(len(x), 3000)
    idx = np.random.default_rng(0).choice(len(x), n_plot, replace=False)
    ax.scatter(x[idx], y[idx], c="#2471a3", alpha=0.10, s=4, rasterized=True)


    if map_point is not None:
        ax.scatter([map_point[0]], [map_point[1]],
                   color="#f1c40f", marker="o", s=120, edgecolor="black",
                   lw=1.0, zorder=5,
                   label=f"MAP ({map_point[0]:.3f}, {map_point[1]:.3f})")
    if reference_point is not None:
        ax.scatter([reference_point[0]], [reference_point[1]],
                   color="#c0392b", marker="*", s=240, edgecolor="black",
                   lw=1.0, zorder=5,
                   label=f"reference "
                         f"({reference_point[0]:.3f}, {reference_point[1]:.3f})")
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        xg = np.linspace(x.min(), x.max(), 100)
        yg = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(xg, yg)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        zf = np.sort(Z.ravel())[::-1]
        cdf = np.cumsum(zf) / zf.sum()
        l68 = zf[np.searchsorted(cdf, 0.68)]
        l95 = zf[np.searchsorted(cdf, 0.95)]
        ax.contour(X, Y, Z, levels=[l95, l68],
                   colors=["#2471a3", "#c0392b"], linewidths=[1.2, 2.0])
    except Exception:
        pass

    ax.set_xlabel(labels[0], fontsize=12)
    ax.set_ylabel(labels[1], fontsize=12)
    ax.set_title(f"joint posterior  ({labels[0]}, {labels[1]})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.18)
    _despine(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax
