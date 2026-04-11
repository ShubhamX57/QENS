"""
Plots for QENS analysis.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls
from scipy.signal import fftconvolve
from .models import ce, fickian, gnorm, lorentz

_SQW_CMAP = LinearSegmentedColormap.from_list("qens", ["#0a0e1a", "#0c2d6b", "#1565c0", "#42a5f5",
             "#e3f2fd", "#ff8f00", "#e65100"], N=512)



def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



def plot_overview(dataset, save_path=None):
    fnames = sorted(dataset.keys())
    n = len(fnames)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.0*nrows))
    axes = np.array(axes).flatten()


    for ax, fname in zip(axes, fnames):
        d = dataset[fname]
        e = d["e"]
        good = d["good"]
        n_lo = max(2, len(good) // 5)
        avg = np.nanmean([d["data"][good[j]] for j in range(n_lo)], axis=0)
        avg = np.where(np.isfinite(avg), avg, 0.0)
        ewin = min(0.5 * d["ei"], 1.2)
        mask = (e >= -ewin) & (e <= ewin)
        peak = avg[mask].max()
        y = avg[mask] / peak if peak > 0 else avg[mask]


        col = "#2ca02c" if d["temp"] <= 270 else "#c0392b" if d["kind"] == "inc" else "#2471a3"
        ax.plot(e[mask], y, color=col, lw=1.8)
        ax.axvline(0, color="#aaa", lw=0.8, ls=":")
        ax.set_title(fname, fontsize=7, color=col)
        ax.set_xlabel("ω (meV)", fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)
        info = f"E₀={d.get('e0',0):+.3f} meV\nFWHM={d.get('fwhm_res',0)*1000:.0f} µeV [{d.get('res_source','?')}]"
        ax.text(0.03, 0.96, info, transform=ax.transAxes, va="top", fontsize=5.5,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        _despine(ax)


    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("All datasets  |  green=frozen  red=INC  blue=COH", fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  figure saved → {save_path}")
    return fig, axes


def plot_spectrum(d_inc, d_map, l_map, q_target=1.06, d_res=None, ewin=0.8, save_path=None):
    gp = d_inc["good"]
    qg = d_inc["q"][gp]
    sr = d_inc["sigma_res"]
    emask = (d_inc["e"] >= -ewin) & (d_inc["e"] <= ewin)
    ew = d_inc["e"][emask]


    near = np.where(np.abs(qg - q_target) < 0.10)[0]
    if len(near) == 0:
        near = np.argsort(np.abs(qg - q_target))[:4]


    spec = np.nanmean([d_inc["data"][gp[j]][emask] for j in near], axis=0)
    errs = np.sqrt(np.nanmean([d_inc["errs"][gp[j]][emask]**2 for j in near], axis=0))
    spec = np.where(np.isfinite(spec), spec, 0.0)
    err_floor = max(spec.max() * 0.05, 1e-12)
    errs = np.where(errs > 0, errs, err_floor)


    peak = spec.max()
    sn, en = spec / peak, errs / peak


    wf = np.linspace(-ewin, ewin, 1000)
    dt = wf[1] - wf[0]
    gamma = float(ce(q_target, d_map, l_map))


    el = gnorm(wf, sr)
    el /= el.max()
    ql_ = fftconvolve(lorentz(wf, gamma), gnorm(wf, sr), mode="same") * dt
    ql = ql_ / ql_.max() if ql_.max() > 0 else ql_


    sn_fine = np.interp(wf, ew, sn)
    amp, _ = nnls(np.column_stack([el, ql, np.ones(len(wf))]), sn_fine)
    fit = amp[0] * el + amp[1] * ql + amp[2]


    fit_on_data = np.interp(ew, wf, fit)
    chi2r = np.sum(((sn - fit_on_data) / en)**2) / max(len(ew) - 4, 1)


    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.fill_between(wf, amp[2], amp[0]*el + amp[2], alpha=0.22, color="#2471a3", label="elastic")
    ax.fill_between(wf, amp[2], amp[1]*ql + amp[2], alpha=0.22, color="#e67e22", label="quasi-elastic")
    ax.errorbar(ew, sn, yerr=en, fmt=".", color="#333", ms=3.5, elinewidth=0.7, alpha=0.8,
                label=f"data Q={qg[near].mean():.2f} Å⁻¹")


    if d_res is not None:
        gc = d_res["good"]
        qc = d_res["q"][gc]
        mk = (d_res["e"] >= -ewin) & (d_res["e"] <= ewin)
        ec = d_res["e"][mk]
        nr = np.where(np.abs(qc - q_target) < 0.10)[0]
        if len(nr) == 0:
            nr = np.argsort(np.abs(qc - q_target))[:4]
        rs = np.nanmean([d_res["data"][gc[j]][mk] for j in nr], axis=0)
        rs = np.where(np.isfinite(rs), rs, 0.0)
        if rs.max() > 0:
            rs /= rs.max()
        ax.plot(ec, rs, "--", color="#c0392b", lw=1.8, label=f"resolution ({d_res['name']})")


    ax.plot(wf, fit, "-", color="#c0392b", lw=2.2, label=rf"CE MAP $\chi^2_r={chi2r:.2f}$")
    ax.annotate("", xy=(gamma, 0.50), xytext=(0, 0.50),
                arrowprops=dict(arrowstyle="<->", color="#e67e22", lw=1.8))
    ax.text(gamma/2, 0.56, f"HWHM = {gamma*1000:.0f} µeV", ha="center", color="#e67e22", fontsize=9.5)


    ax.axvline(0, color="#aaa", lw=0.8, ls=":")
    ax.set_xlabel(r"energy transfer $\hbar\omega$ (meV)", fontsize=12)
    ax.set_ylabel(r"$S(Q,\omega)$ normalised", fontsize=12)
    ax.set_title(rf"spectrum at $Q\approx{q_target:.2f}$ Å$^{{-1}}$  |  {d_inc['name']}", fontsize=11)
    ax.legend(fontsize=9.5)
    ax.set_xlim(-ewin, ewin)
    ax.set_ylim(-0.05, 1.20)
    ax.grid(True, alpha=0.18)
    _despine(ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  figure saved → {save_path}")
    return fig, ax


def plot_sqw_maps(d_inc, d_coh=None, ewin=1.2, save_path=None):
    left_data = d_coh if d_coh is not None else d_inc
    left_label = f"COH {left_data.get('temp','?')} K" if d_coh else "INC (no COH)"
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#0d1117")


    panels = [(left_data, left_label), (d_inc, f"INC {d_inc.get('temp','?')} K")]
    for ax, (d, title) in zip(axes, panels):
        g = d["good"]
        qg = d["q"][g]
        e = d["e"]
        emask = (e >= -ewin) & (e <= ewin)
        img = d["data"][np.ix_(g, emask)]
        img = np.where(np.isfinite(img) & (img > 0), img, np.nan)
        qs = np.argsort(qg)
        ism = gaussian_filter(np.where(np.isfinite(img[qs]), img[qs], 0.0), sigma=[1.5, 0.8])
        ism[ism <= 0] = np.nan


        vmin = max(np.nanpercentile(ism, 2), 1e-8)
        vmax = np.nanpercentile(ism, 99)
        im = ax.pcolormesh(e[emask], qg[qs], ism, cmap=_SQW_CMAP, norm=LogNorm(vmin=vmin, vmax=vmax),
                           shading="auto", rasterized=True)
        ax.axvline(0, color="white", lw=1.0, ls="--", alpha=0.4)
        ax.set_xlabel("ω (meV)", color="white", fontsize=12)
        ax.set_ylabel("Q (Å⁻¹)", color="white", fontsize=12)
        ax.set_title(title, color="white", fontsize=11, pad=10)
        ax.tick_params(colors="white")
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
        cb.set_label("S(Q,ω)", color="white", fontsize=10)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")


    fig.suptitle("S(Q,ω)", color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="#0d1117", dpi=150)
        print(f"  figure saved → {save_path}")
    return fig, axes


def plot_hwhm(q_hwhm, g_hwhm, g_err, samples, d_map, l_map, d_inc,
              q_plot_min=0.3, q_plot_max=2.5, save_path=None):
    d_s, l_s = samples[:,0], np.abs(samples[:,1])
    q_fine = np.linspace(q_plot_min, q_plot_max, 400)
    q2_fine = q_fine**2


    fig, ax = plt.subplots(figsize=(9.5, 6.5))


    g_fan = ce(q_fine[None,:], d_s[:,None], l_s[:,None]) * 1000
    rng = np.random.default_rng(0)
    idx_f = rng.choice(len(d_s), min(400, len(d_s)), replace=False)
    for i in idx_f:
        ax.plot(q2_fine, g_fan[i], color="#2471a3", alpha=0.015, lw=0.8)

    ax.fill_between(q2_fine, np.percentile(g_fan, 2.5, axis=0), np.percentile(g_fan, 97.5, axis=0),
                    alpha=0.25, color="#2471a3", label=f"95% posterior (n={len(d_s)})")
    ax.plot(q2_fine, ce(q_fine, d_map, l_map)*1000, "-", color="#c0392b", lw=3.0,
            label=rf"CE MAP $D={d_map:.4f}$ Å²/ps $\ell={l_map:.4f}$ Å")
    ax.plot(q2_fine, fickian(q_fine, d_map)*1000, "--", color="#555", lw=1.8,
            label=rf"Fickian $D={d_map:.4f}$ (low‑Q limit)")

    ax.errorbar(q_hwhm**2, g_hwhm*1000, yerr=2*g_err*1000, fmt="o", color="#111", ms=7,
                capsize=4, elinewidth=1.8, label="data ±2σ")


    res_hwhm_uev = d_inc["fwhm_res"] / 2 * 1000
    ax.axhline(res_hwhm_uev, color="#888", ls=":", lw=1.5, label=f"resolution HWHM = {res_hwhm_uev:.0f} µeV")
    ax.axhspan(0, res_hwhm_uev*1.1, alpha=0.04, color="#888")

    ax.set_xlabel(r"$Q^2$ (Å$^{-2}$)", fontsize=13)
    ax.set_ylabel(r"$\Gamma(Q)$ (µeV)", fontsize=13)
    ax.set_title("peak width vs $Q^2$ — deviation from linearity = jump diffusion", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.05, q2_fine[-1]+0.1)
    ax.set_ylim(bottom=-10)
    ax.grid(True, alpha=0.20)
    _despine(ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  figure saved → {save_path}")
    return fig, ax



def plot_posteriors(samples, d_map, l_map, d_inc, ref_values=None, save_path=None):
    d_s = samples[:,0]
    l_s = np.abs(samples[:,1])
    tau_s = l_s**2 / (6 * d_s)
    tau_map = l_map**2 / (6 * d_map)

    d_lo, d_hi = np.percentile(d_s, [2.5, 97.5])
    l_lo, l_hi = np.percentile(l_s, [2.5, 97.5])
    tau_lo, tau_hi = np.percentile(tau_s, [2.5, 97.5])


    if ref_values is None:
        ref_values = {}
    try:
        import emcee
        sampler_label = "emcee"
    except ImportError:
        sampler_label = "MH fallback"


    params = [
        (d_s,   d_map,   d_lo,   d_hi,   "D (Å²/ps)", "#c0392b", ref_values.get("D", [])),
        (l_s,   l_map,   l_lo,   l_hi,   "ℓ (Å)",      "#1e8449", ref_values.get("l", [])),
        (tau_s, tau_map, tau_lo, tau_hi, "τ (ps)",      "#e67e22", ref_values.get("tau", [])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(f"Bayesian posteriors | CE model | {len(d_s)} samples ({sampler_label})\n"
                 f"res FWHM = {d_inc['fwhm_res']*1000:.0f} µeV [{d_inc['res_source']}]", fontsize=11)

    for ax, (arr, map_val, lo, hi, xlabel, col, refs) in zip(axes, params):
        med = float(np.median(arr))
        cnt, _, _ = ax.hist(arr, bins=80, density=True, color=col, alpha=0.80, edgecolor="white", lw=0.25)
        pk = cnt.max()
        ax.axvspan(lo, hi, alpha=0.18, color=col)
        ax.axvline(med, color="#111", lw=2.5, label=f"median = {med:.4f}")
        ax.axvline(map_val, color=col, lw=1.8, ls="--", label=f"MAP = {map_val:.4f}")
        for ref_val, ref_label in refs:
            ax.axvline(ref_val, color="#1565c0", lw=1.8, ls=":", label=f"{ref_label} = {ref_val}")
        ax.annotate("", xy=(hi, pk*1.10), xytext=(lo, pk*1.10),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=2.0))
        ax.text((lo+hi)/2, pk*1.17, f"95% CI [{lo:.4f}, {hi:.4f}]",
                ha="center", fontsize=8.5, color=col, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("density", fontsize=11)
        ax.set_title(xlabel, fontsize=12, color=col, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.92)
        ax.set_ylim(0, pk*1.32)
        ax.grid(True, alpha=0.20)
        _despine(ax)


    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  figure saved → {save_path}")
    return fig, axes



def plot_joint_posterior(samples, d_map, l_map, save_path=None):
    d_s = samples[:,0]
    l_s = np.abs(samples[:,1])
    fig, ax = plt.subplots(figsize=(7, 6))


    n_plot = min(len(d_s), 3000)
    idx = np.random.default_rng(0).choice(len(d_s), n_plot, replace=False)
    ax.scatter(d_s[idx], l_s[idx], c="#2471a3", alpha=0.12, s=4, rasterized=True)


    ax.axvline(d_map, color="#c0392b", lw=1.8, ls="--", label=f"D MAP = {d_map:.4f}")
    ax.axhline(l_map, color="#1e8449", lw=1.8, ls="--", label=f"l MAP = {l_map:.4f}")


    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([d_s, l_s])
        kde = gaussian_kde(xy)
        d_g = np.linspace(d_s.min(), d_s.max(), 100)
        l_g = np.linspace(l_s.min(), l_s.max(), 100)
        D, L = np.meshgrid(d_g, l_g)
        Z = kde(np.vstack([D.ravel(), L.ravel()])).reshape(D.shape)
        z_flat = np.sort(Z.ravel())[::-1]
        cdf = np.cumsum(z_flat) / z_flat.sum()
        lvl68 = z_flat[np.searchsorted(cdf, 0.68)]
        lvl95 = z_flat[np.searchsorted(cdf, 0.95)]
        ax.contour(D, L, Z, levels=[lvl95, lvl68], colors=["#2471a3", "#c0392b"], linewidths=[1.2, 2.0])
    except Exception:
        pass

    ax.set_xlabel(r"$D$ (Å²/ps)", fontsize=13)
    ax.set_ylabel(r"$\ell$ (Å)", fontsize=13)
    ax.set_title("Joint posterior (D, l)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.20)
    _despine(ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  figure saved → {save_path}")
    return fig, ax
