"""Report plots: the four pictures that answer what scientists ask of a fit.

- ``spectra``: data, model and residuals at chosen Q.
- ``misfit_map``: where each model fails, as significance in every (Q, ω) cell.
- ``holdout``: an energy the physics was not fitted to, with the prediction on top.
- ``timescales``: which incident energy sees which motion.

Every function returns a matplotlib Figure and leaves global settings alone.
Needs matplotlib: ``pip install "uqens[plot]"``.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from .constants import HBAR
from .rotation import channel_weights

SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ("#9ec5f4", "#6da7ec", "#2a78d6", "#184f95")
DIVERGING = LinearSegmentedColormap.from_list(
    "uqens_diverging", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a8a7", "#e34948", "#9c1f1f"], N=256
)
STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "text.color": INK,
    "lines.linewidth": 1.8,
}


def _dataset_index(problem, energy):
    for k, d in enumerate(problem.datasets):
        if abs(d.ei - energy) < 0.05:
            return k
    raise ValueError(f"no dataset at Ei = {energy} meV")


def _bins_near(dataset, q):
    if q is None:  # three well-covered bins spread across the Q range
        counts = np.bincount(dataset.bin_id, minlength=dataset.n_bins)
        good = np.where(counts >= 0.6 * counts.max())[0]
        return [int(good[i]) for i in np.unique(np.linspace(0, len(good) - 1, 3).round().astype(int))]
    return [int(np.argmin(np.abs(dataset.q_bins - v))) for v in q]


def _header(fig, title, subtitle=None, legend_from=None):
    """Title (and subtitle) top left; the legend, if any, top right."""
    fig.text(0.01, 0.985, title, fontsize=11.5, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.01, 0.985 - 0.32 / fig.get_size_inches()[1], subtitle, fontsize=8.8, color=INK2, va="top")
    if legend_from is not None:
        handles, labels = legend_from.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.995), ncol=len(labels))


def _spectra_panels(fig, grid, dataset, curve, bins, title_prefix, label):
    axes = []
    for c, b in enumerate(bins):
        inner = grid[c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax, axr = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        m = dataset.bin_id == b
        w = dataset.omega[m]
        ax.errorbar(
            w,
            dataset.y[m],
            yerr=dataset.err[m],
            fmt="o",
            ms=2.0,
            color=INK2,
            ecolor=GRID,
            elinewidth=0.7,
            label="data",
            zorder=2,
        )
        ax.plot(w, curve[m], color=BLUE, label=label, zorder=3)
        ax.set_yscale("log")
        ax.tick_params(labelbottom=False)
        ax.set_title(f"{title_prefix}Q = {dataset.q_bins[b]:.2f} Å⁻¹")
        z = (dataset.y[m] - curve[m]) / dataset.err[m]
        axr.axhline(0, color=AXIS, lw=0.8)
        axr.plot(w, z, ".", ms=2.2, color=INK2)
        axr.set_ylim(-4.5, 4.5)
        axr.set_yticks([-3, 0, 3])
        axr.set_xlabel("energy transfer (meV)")
        if c == 0:
            ax.set_ylabel("intensity")
            axr.set_ylabel("Δ/σ")
        axes.append((ax, axr))
    return axes


def spectra(problem, x, energy, q=None):
    """Data, fitted model and normalised residuals at the Q-bins nearest ``q``."""
    k = _dataset_index(problem, energy)
    d = problem.datasets[k]
    bins = _bins_near(d, q)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(3.6 * len(bins) + 0.6, 4.2))
        grid = fig.add_gridspec(1, len(bins), wspace=0.25, left=0.08, right=0.98, top=0.82, bottom=0.13)
        axes = _spectra_panels(fig, grid, d, problem.model_curve(x, k), bins, "", "model")
        _header(fig, f"{problem.spec.name} at Eᵢ = {d.ei:.2f} meV", legend_from=axes[0][0])
    return fig


def misfit(problem, x, block=5):
    """Misfit significance per dataset: a list of (q_bins, [(ω centres, z) per bin]).

    Residuals are averaged over ``block`` neighbouring energy points and scaled by
    sqrt(block), so pure noise keeps unit spread and systematic misfit stands out.
    """
    out = []
    for k, d in enumerate(problem.datasets):
        z = (d.y - problem.model_curve(x, k)) / d.err
        rows = []
        for b in range(d.n_bins):
            m = np.where(d.bin_id == b)[0]
            n = len(m) // block
            w = d.omega[m][: n * block].reshape(n, block).mean(1)
            zz = z[m][: n * block].reshape(n, block).mean(1) * np.sqrt(block)
            rows.append((w, zz))
        out.append((d.q_bins, rows))
    return out


def misfit_map(models, block=5, vmax=4.0):
    """Misfit maps, one row per model and one column per incident energy.

    ``models``: {label: (problem, x)} or a single (problem, x).  Red means the data
    lie above the model, blue below.
    """
    if isinstance(models, tuple):
        models = {models[0].spec.name: models}
    labels = list(models)
    n_e = len(models[labels[0]][0].datasets)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            len(labels),
            n_e,
            figsize=(2.9 * n_e + 1.2, 2.4 * len(labels) + 0.8),
            squeeze=False,
            gridspec_kw={"wspace": 0.2, "hspace": 0.35},
        )
        for r, label in enumerate(labels):
            problem, x = models[label]
            chi = problem.chi2_parts(x)
            for k, (qs, rows) in enumerate(misfit(problem, x, block)):
                d = problem.datasets[k]
                ax = axes[r, k]
                dq = np.median(np.diff(qs)) if len(qs) > 1 else 0.1
                for q, (w, zz) in zip(qs, rows, strict=True):
                    if len(w) < 2:
                        continue
                    dw = np.median(np.diff(w))
                    edges = np.concatenate([w - dw / 2, [w[-1] + dw / 2]])
                    ax.pcolormesh(
                        edges,
                        [q - dq / 2, q + dq / 2],
                        zz[None, :],
                        cmap=DIVERGING,
                        vmin=-vmax,
                        vmax=vmax,
                        shading="flat",
                        rasterized=True,
                    )
                wmax = np.max(np.abs(d.omega))
                ax.set_xlim(-wmax, wmax)
                ax.set_ylim(qs.min() - dq / 2, qs.max() + dq / 2)
                ax.text(
                    0.97,
                    0.04,
                    f"χ²/N {chi[k] / len(d.y):.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    fontsize=8,
                    color=INK2,
                    bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none", "alpha": 0.85},
                )
                if r == 0:
                    ax.set_title(f"Eᵢ = {d.ei:.2f} meV")
                if r == len(labels) - 1:
                    ax.set_xlabel("energy transfer (meV)")
                if k == 0:
                    ax.set_ylabel(f"{label}\nQ (Å⁻¹)")
        height = fig.get_size_inches()[1]
        fig.subplots_adjust(left=0.1, right=0.88, top=1 - 0.95 / height, bottom=0.55 / height)
        cax = fig.add_axes([0.9, 0.2, 0.012, 0.6])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=DIVERGING, norm=Normalize(-vmax, vmax)), cax=cax)
        cb.set_label("misfit (σ): data above model is red", fontsize=8.5, color=INK2)
        cb.outline.set_visible(False)
        _header(
            fig,
            "Where each model fails",
            f"Each cell averages {block} energy points, scaled so pure noise has unit spread.",
        )
    return fig


def holdout(rows, q=None):
    """Held-out spectra from ``validate.holdout``: one row of panels per energy left out."""
    with plt.rc_context(STYLE):
        n_col = len(q) if q is not None else 3
        fig = plt.figure(figsize=(3.6 * n_col + 0.6, 3.9 * len(rows) + 0.4))
        height = fig.get_size_inches()[1]
        outer = fig.add_gridspec(
            len(rows), 1, hspace=0.35, left=0.08, right=0.98, top=1 - 0.75 / height, bottom=0.55 / height
        )
        first = None
        for r, row in enumerate(rows):
            problem, x = row["problem"], row["x"]
            d = problem.datasets[0]
            grid = outer[r].subgridspec(1, n_col, wspace=0.25)
            panels = _spectra_panels(
                fig,
                grid,
                d,
                problem.model_curve(x, 0),
                _bins_near(d, q),
                f"{d.ei:.2f} meV held out, ",
                "prediction from the other energies",
            )
            first = first or panels[0][0]
        _header(fig, "Physics fitted on the other energies predicts the one left out", legend_from=first)
    return fig


def timescales(problem, x, posterior=None, min_weight=0.06):
    """What each incident energy can see: its time window (ħ/window to ħ/resolution
    HWHM) against the fitted translational and rotational relaxation times.

    With a ``Posterior``, the translational time gets its 95% band.
    """
    p = problem.physics(x)
    q = np.linspace(0.25, 2.65, 300)

    def tau_translation(D, tau0):
        return 1.0 / (D * q**2 / (1.0 + D * q**2 * tau0))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for k, d in enumerate(problem.datasets):
            name = f"lnsig_{d.ei:.2f}"
            sigma = np.exp(x[problem.idx[name]]) if name in problem.idx else d.meta["sigma0"]
            t_lo, t_hi = HBAR / np.max(np.abs(d.omega)), HBAR / (1.1774 * sigma)
            dq = np.median(np.diff(d.q_bins))
            q_lo, q_hi = d.q_bins.min() - dq / 2, d.q_bins.max() + dq / 2
            colour = BLUE_RAMP[k % len(BLUE_RAMP)]
            ax.fill_between([q_lo, q_hi], t_lo, t_hi, color=colour, alpha=0.08, lw=0)
            ax.plot([q_lo, q_hi, q_hi, q_lo, q_lo], [t_lo, t_lo, t_hi, t_hi, t_lo], color=colour, lw=1.3)
            ax.text(q_hi - 0.02, t_hi * 1.1, f"Eᵢ {d.ei:.2f} meV", color=INK2, fontsize=8.5, ha="right")
        if posterior is not None:
            s = posterior.samples
            lnD = s[:, posterior.names.index("lnD")]
            lntau = s[:, posterior.names.index("lntau0")] if "lntau0" in posterior.names else np.full(len(s), -np.inf)
            pick = np.random.default_rng(0).choice(len(s), size=min(600, len(s)), replace=False)
            band = np.array([tau_translation(np.exp(lnD[i]), np.exp(lntau[i])) for i in pick])
            lo, hi = np.percentile(band, [2.5, 97.5], axis=0)
            ax.fill_between(q, lo, hi, color=ORANGE, alpha=0.2, lw=0)
        ax.plot(q, tau_translation(p["D"], p["tau0"]), color=ORANGE, label="translation")
        if problem.spec.rotation != "none":
            W, ls, ms = channel_weights(
                q, problem.spec.rotation, problem.spec.R, problem.spec.lmax, problem.spec.theta0
            )
            labelled = False
            for c in range(len(ls)):
                lv, mv = int(ls[c]), int(ms[c])
                if lv == 0 or W[:, c].max() < min_weight:
                    continue
                rate = lv * (lv + 1) * p["Dt"] + mv * mv * (p["Ds"] - p["Dt"])
                on = W[:, c] >= min_weight
                ax.plot(
                    np.where(on, q, np.nan),
                    np.full_like(q, 1.0 / rate),
                    color=AQUA,
                    alpha=float(np.clip(W[:, c].max() / 0.5, 0.3, 1.0)),
                    lw=2.2,
                    solid_capstyle="butt",
                    label=None if labelled else "rotation channels",
                )
                labelled = True
        ax.set_yscale("log")
        ax.set_xlabel("Q (Å⁻¹)")
        ax.set_ylabel("relaxation time (ps)")
        ax.grid(True, which="major", axis="y", color=GRID, lw=0.8)
        ax.legend(loc="upper right")
        _header(
            fig,
            "Which incident energy sees which motion",
            "Boxes: the times each energy resolves, from ħ/window to ħ/resolution half-width.\n"
            f"Green: rotational channels where they carry at least {min_weight:.0%} of the intensity.",
        )
        fig.subplots_adjust(left=0.11, right=0.97, top=0.8, bottom=0.12)
    return fig
