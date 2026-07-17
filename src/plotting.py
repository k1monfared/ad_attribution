"""Figure generation for the Bayesian attribution demo.

Plain, readable figures: one accent color per role, credible bands shaded, the
campaign gap highlighted, no chart junk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# Brand-neutral, colorblind-safe roles.
C_OBSERVED = "#1f2933"
C_COUNTER = "#2f6f9f"
C_BAND = "#8fc0e0"
C_EFFECT = "#c25a2b"
C_EFFECT_BAND = "#f0c3a8"
C_TRUTH = "#3f8f5f"
C_BAD = "#b3261e"
C_PLACEBO = "#b7c0c9"
C_GRID = "#e6e6e6"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    }
)


def _campaign_marker(ax, cdate, label="campaign start"):
    ax.axvline(cdate, color="#9aa5b1", linestyle="--", linewidth=1.2)
    ax.annotate(
        label, xy=(cdate, ax.get_ylim()[1]), xytext=(6, -14),
        textcoords="offset points", color="#616e7c", fontsize=9,
    )


def _fmt_dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def _save(fig, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_observed_vs_counterfactual(
    dates, observed, cf_mean, cf_lo, cf_hi, n_pre, true_cf, title, out_path
):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    cdate = dates[n_pre]
    post = slice(n_pre, len(observed))

    ax.fill_between(dates, cf_lo, cf_hi, color=C_BAND, alpha=0.55, linewidth=0,
                    label="counterfactual 95% credible interval")
    ax.fill_between(
        dates[post], cf_mean[post], observed[post],
        color=C_EFFECT_BAND, alpha=0.7, linewidth=0, label="campaign effect (gap)",
    )
    ax.plot(dates, cf_mean, color=C_COUNTER, linewidth=1.8, label="posterior mean counterfactual")
    ax.plot(dates, observed, color=C_OBSERVED, linewidth=1.6, label="observed streams")
    if true_cf is not None:
        ax.plot(dates, true_cf, color=C_TRUTH, linewidth=1.1, linestyle=":", label="true counterfactual")

    _campaign_marker(ax, cdate)
    _fmt_dates(ax)
    ax.set_ylabel("daily streams")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    _save(fig, out_path)


def plot_cumulative(dates, cum_mean, cum_lo, cum_hi, n_pre, true_cum, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.0))
    post_dates = dates[n_pre:]

    ax.fill_between(post_dates, cum_lo, cum_hi, color=C_BAND, alpha=0.55,
                    linewidth=0, label="95% credible interval")
    ax.plot(post_dates, cum_mean, color=C_EFFECT, linewidth=2.0, label="posterior mean cumulative effect")
    if true_cum is not None:
        ax.plot(post_dates, true_cum, color=C_TRUTH, linewidth=1.3, linestyle=":", label="true cumulative effect")
    ax.axhline(0, color="#9aa5b1", linewidth=1.0)

    _fmt_dates(ax)
    ax.set_ylabel("cumulative extra streams")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _save(fig, out_path)


def plot_pointwise_effect(dates, eff_mean, eff_lo, eff_hi, n_pre, true_effect, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cdate = dates[n_pre]

    ax.fill_between(dates, eff_lo, eff_hi, color=C_BAND, alpha=0.5, linewidth=0,
                    label="95% credible interval")
    ax.plot(dates, eff_mean, color=C_EFFECT, linewidth=1.8, label="posterior mean effect")
    if true_effect is not None:
        ax.plot(dates, true_effect, color=C_TRUTH, linewidth=1.1, linestyle=":", label="true effect")
    ax.axhline(0, color="#9aa5b1", linewidth=1.0)

    _campaign_marker(ax, cdate)
    _fmt_dates(ax)
    ax.set_ylabel("daily extra streams")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _save(fig, out_path)


def plot_ppc(dates_pre, observed_pre, prior_lo, prior_hi, post_lo, post_hi, post_med, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)

    ax1.fill_between(dates_pre, prior_lo, prior_hi, color=C_PLACEBO, alpha=0.6, linewidth=0,
                     label="prior predictive 95%")
    ax1.plot(dates_pre, observed_pre, color=C_OBSERVED, linewidth=1.4, label="observed (pre-period)")
    ax1.set_title("Prior predictive check", fontweight="bold", fontsize=11)
    ax1.set_ylabel("daily streams")
    ax1.legend(loc="upper left", frameon=False, fontsize=9)

    ax2.fill_between(dates_pre, post_lo, post_hi, color=C_BAND, alpha=0.6, linewidth=0,
                     label="posterior predictive 95%")
    ax2.plot(dates_pre, post_med, color=C_COUNTER, linewidth=1.4, label="posterior predictive median")
    ax2.plot(dates_pre, observed_pre, color=C_OBSERVED, linewidth=1.2, label="observed (pre-period)")
    ax2.set_title("Posterior predictive check (pre-period)", fontweight="bold", fontsize=11)
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    for ax in (ax1, ax2):
        _fmt_dates(ax)
    _save(fig, out_path)


def plot_convergence(idata, out_path):
    """Trace plots for key scalars plus the energy (BFMI) diagnostic."""
    post = idata.posterior
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    def trace(ax, name, idx=None):
        for c in range(post.sizes["chain"]):
            if idx is None:
                v = post[name].isel(chain=c).values
            else:
                v = post[name].isel(chain=c, **idx).values
            ax.plot(v, linewidth=0.7, alpha=0.85)
        ax.set_title(name if idx is None else f"{name}{list(idx.values())}", fontsize=10, fontweight="bold")
        ax.set_xlabel("draw")

    trace(axes[0], "sigma")
    if "sigma_level" in post.data_vars:
        trace(axes[1], "sigma_level")
    else:
        trace(axes[1], "beta", {"beta_dim_0": 0})

    energy = idata.sample_stats["energy"].values
    marg = energy - energy.mean()
    trans = np.diff(energy, axis=1).ravel()
    axes[2].hist(marg.ravel(), bins=30, density=True, color=C_BAND, alpha=0.7, label="marginal energy")
    axes[2].hist(trans, bins=30, density=True, color=C_EFFECT, alpha=0.5, label="energy transitions")
    axes[2].set_title("energy (BFMI)", fontsize=10, fontweight="bold")
    axes[2].legend(loc="upper right", frameon=False, fontsize=8.5)
    _save(fig, out_path)


def plot_coverage(true_cum, est_cum, covered_flags, coverage_frac, title, out_path):
    true_cum = np.asarray(true_cum) / 1000.0
    est_cum = np.asarray(est_cum) / 1000.0
    covered = np.asarray(covered_flags, dtype=bool)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    lo = min(true_cum.min(), est_cum.min())
    hi = max(true_cum.max(), est_cum.max())
    pad = 0.05 * (hi - lo)
    line = np.array([lo - pad, hi + pad])
    ax.plot(line, line, color="#9aa5b1", linewidth=1.2, linestyle="--", label="perfect recovery")
    ax.scatter(true_cum[covered], est_cum[covered], s=32, color=C_COUNTER, alpha=0.8,
               label="true effect inside 95% CI")
    if (~covered).any():
        ax.scatter(true_cum[~covered], est_cum[~covered], s=32, color=C_BAD, alpha=0.85,
                   label="true effect outside 95% CI")
    ax.set_xlabel("true cumulative effect (thousands)")
    ax.set_ylabel("estimated cumulative effect (thousands)")
    ax.set_title(f"{title}\n95% CI coverage = {coverage_frac*100:.0f}%", fontweight="bold", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _save(fig, out_path)


def plot_seasonality_comparison(aware, blind, out_path):
    """Aware vs blind: coverage and recovery error."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.4))
    labels = ["seasonality\naware", "seasonality\nblind"]
    x = np.arange(2)

    cov = [aware["coverage_95"] * 100, blind["coverage_95"] * 100]
    ax1.bar(x, cov, color=[C_COUNTER, C_BAD], alpha=0.85, width=0.6)
    ax1.axhline(95, color="#3f8f5f", linewidth=1.4, linestyle="--", label="nominal 95%")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("95% CI coverage (%)")
    ax1.set_title("Interval calibration", fontweight="bold", fontsize=11)
    ax1.legend(loc="lower left", frameon=False, fontsize=9)
    for xi, c in zip(x, cov):
        ax1.text(xi, c + 1, f"{c:.0f}%", ha="center", fontsize=9)

    width = [aware["mean_interval_width"] / 1000.0, blind["mean_interval_width"] / 1000.0]
    ax2.bar(x, width, color=[C_COUNTER, C_BAD], alpha=0.85, width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("mean 95% CI width (thousands of streams)")
    ax2.set_title("Interval sharpness", fontweight="bold", fontsize=11)
    for xi, wv in zip(x, width):
        ax2.text(xi, wv + max(width) * 0.01, f"{wv:,.0f}k", ha="center", fontsize=9)
    _save(fig, out_path)


def plot_annual_component(dates, observed, cf_mean, cf_lo, cf_hi, n_pre,
                          recovered_annual, injected_annual, out_path):
    """Long-history scenario: annual component included and recovered."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.4))
    cdate = dates[n_pre]

    ax1.fill_between(dates, cf_lo, cf_hi, color=C_BAND, alpha=0.5, linewidth=0,
                     label="counterfactual 95% CI")
    ax1.plot(dates, cf_mean, color=C_COUNTER, linewidth=1.6, label="posterior mean counterfactual")
    ax1.plot(dates, observed, color=C_OBSERVED, linewidth=1.2, label="observed streams")
    ax1.axvline(cdate, color="#9aa5b1", linestyle="--", linewidth=1.2)
    _fmt_dates(ax1)
    ax1.set_ylabel("daily streams")
    ax1.set_title("Long history: annual component included", fontweight="bold", fontsize=11)
    ax1.legend(loc="upper left", frameon=False, fontsize=8.5)

    ax2.plot(dates, injected_annual, color=C_TRUTH, linewidth=1.6, linestyle=":", label="injected annual (treated-specific)")
    ax2.plot(dates, recovered_annual, color=C_EFFECT, linewidth=1.5, label="recovered annual component")
    ax2.axhline(0, color="#9aa5b1", linewidth=1.0)
    _fmt_dates(ax2)
    ax2.set_ylabel("streams (deviation)")
    ax2.set_title("Recovered vs injected annual seasonality", fontweight="bold", fontsize=11)
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    _save(fig, out_path)


def plot_communication(dates, observed, cf_mean, cf_lo, cf_hi, n_pre, summary, out_path):
    """Declining-scenario figure framed as decline avoided / streams retained."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    cdate = dates[n_pre]
    post = slice(n_pre, len(observed))

    # Lift shading first, with a moderate alpha so the credible band shows
    # through, then the band, then its bounds as dashed lines on top so neither
    # the interval nor the shaded lift is hidden by the other.
    ax.fill_between(
        dates[post], cf_mean[post], observed[post],
        color=C_EFFECT_BAND, alpha=0.55, linewidth=0, zorder=1,
        label="streams retained by campaign",
    )
    ax.fill_between(dates, cf_lo, cf_hi, color=C_BAND, alpha=0.35, linewidth=0,
                    zorder=2, label="counterfactual 95% credible interval")
    ax.plot(dates, cf_lo, color=C_COUNTER, linewidth=1.0, linestyle="--", alpha=0.9, zorder=3)
    ax.plot(dates, cf_hi, color=C_COUNTER, linewidth=1.0, linestyle="--", alpha=0.9, zorder=3)
    ax.plot(dates, cf_mean, color=C_COUNTER, linewidth=1.8, zorder=4, label="without campaign (counterfactual)")
    ax.plot(dates, observed, color=C_OBSERVED, linewidth=1.7, zorder=5, label="observed (with campaign)")

    pre_level = summary["pre_daily_mean"]
    ax.axhline(pre_level, color="#9aa5b1", linewidth=1.0, linestyle="-.")
    ax.annotate("pre-campaign level", xy=(dates[2], pre_level), xytext=(0, 5),
                textcoords="offset points", color="#616e7c", fontsize=9)

    _campaign_marker(ax, cdate)
    _fmt_dates(ax)
    ax.set_ylabel("daily streams")
    ax.set_title("Declining series: the campaign avoided a deeper drop", fontweight="bold")

    retained = summary["streams_retained"]
    # Mathematical minus for the signed percentages, without touching hyphens.
    obs_mom = f"{summary['mom_change_observed'] * 100:+.0f}".replace("-", "−")
    cf_mom = f"{summary['mom_change_counterfactual'] * 100:+.0f}".replace("-", "−")
    note = (
        f"Observed is {obs_mom}% vs the pre-campaign level, so it still looks down.\n"
        f"Without the campaign it would have been {cf_mom}%.\n"
        f"The campaign retained about {retained:,.0f} streams."
    )
    ax.text(0.985, 0.03, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#3e4c59",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f7fa", edgecolor="#cbd2d9"))
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    _save(fig, out_path)


def plot_donor_selection(dates, treated, selected_donors, fanbase, n_pre, out_path):
    """Pre-period co-movement: good donors track the treated unit, fanbase is flat."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    pre = slice(0, n_pre)

    for k in range(selected_donors.shape[1]):
        ax.plot(dates[pre], selected_donors[pre, k], color=C_BAND, linewidth=0.9, alpha=0.8)
    ax.plot([], [], color=C_BAND, linewidth=0.9, label="selected donors (co-moving)")
    ax.plot(dates[pre], treated[pre], color=C_OBSERVED, linewidth=2.0, label="treated unit (target audience)")
    ax.plot(dates[pre], fanbase[pre], color=C_BAD, linewidth=1.8, linestyle="--",
            label="same-artist other region (fanbase, flat)")

    _fmt_dates(ax)
    ax.set_ylabel("daily streams")
    ax.set_title("Pre-period parallel-trends check (donor selection)", fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _save(fig, out_path)


def plot_reconciliation(result, out_path):
    """Base vs reconciled vs truth across the hierarchy (one campaign)."""
    names = ["Total"] + [n.replace("_", " ") for n in result["names"]]
    base_bottom = np.array(result["base"]["bottom"])
    base_top = result["base"]["top"]
    bottom_sum = result["base"]["bottom_sum"]
    mint_bottom = np.array(result["mint"]["bottom"])
    mint_top = result["mint"]["top"]
    true_bottom = np.array(result["true_bottom"])
    true_top = result["true_top"]

    base_vals = np.concatenate([[base_top], base_bottom]) / 1000.0
    mint_vals = np.concatenate([[mint_top], mint_bottom]) / 1000.0
    true_vals = np.concatenate([[true_top], true_bottom]) / 1000.0

    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.bar(x - w, base_vals, w, color=C_PLACEBO, label="base (independent Bayesian)")
    ax.bar(x, mint_vals, w, color=C_COUNTER, label="reconciled (MinT)")
    ax.bar(x + w, true_vals, w, color=C_TRUTH, label="true injected effect")

    ax.plot(0, bottom_sum / 1000.0, marker="D", color=C_BAD, markersize=9, linestyle="none",
            label="bottom-up sum (incoherent)")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("cumulative extra streams (thousands)")
    ax.set_title("Hierarchical reconciliation: base vs reconciled vs truth", fontweight="bold")
    gap = result["incoherence_gap"]
    ax.text(
        0.30, 0.72,
        f"At the Total node the base has two values:\n"
        f"aggregate estimate {base_top/1000:,.0f}k and bottom-up sum {bottom_sum/1000:,.0f}k.\n"
        f"Incoherence gap = {gap:,.0f}. Reconciliation removes it.",
        transform=ax.transAxes, fontsize=9, color="#3e4c59",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f7fa", edgecolor="#cbd2d9"),
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="x", visible=False)
    _save(fig, out_path)


def plot_naive_vs_counterfactual(naive_total, cf_total, out_path):
    """Contrast the naive pre-vs-post delta with the counterfactual attribution.

    The naive number credits the campaign with the entire rise above the
    pre-campaign level. The counterfactual attribution is only the gap above the
    projected no-ad trend, so the naive number overstates the effect.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    labels = ["naive pre vs post\n(entire rise)", "counterfactual\nattribution"]
    vals = np.array([naive_total, cf_total]) / 1000.0
    colors = [C_BAD, C_COUNTER]
    x = np.arange(2)
    ax.bar(x, vals, width=0.6, color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("extra streams credited to the campaign (thousands)")
    ax.set_title("The naive before-and-after number overstates the ad effect",
                 fontweight="bold", fontsize=11)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.01, f"{v * 1000:,.0f}", ha="center", fontsize=10)
    overstatement = (naive_total / cf_total - 1.0) * 100 if cf_total else float("nan")
    ax.text(
        0.5, 0.90,
        f"Naive overstates by {overstatement:,.0f}%: it credits the ad with the whole\n"
        f"rise, including the trend that would have happened anyway.",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#3e4c59",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f7fa", edgecolor="#cbd2d9"),
    )
    ax.grid(axis="x", visible=False)
    _save(fig, out_path)


def plot_decline_pretrend(dates, observed, n_pre, out_path):
    """Show the pre-campaign downward slope: the series was already falling."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    cdate = dates[n_pre]
    t = np.arange(len(observed))
    coef = np.polyfit(t[:n_pre], observed[:n_pre], 1)
    trend = np.polyval(coef, t)

    ax.plot(dates, observed, color=C_OBSERVED, linewidth=1.5, label="observed streams")
    ax.plot(dates[:n_pre], trend[:n_pre], color=C_EFFECT, linewidth=2.2,
            label="pre-campaign trend (fitted)")
    ax.plot(dates[n_pre:], trend[n_pre:], color=C_EFFECT, linewidth=1.6, linestyle="--",
            label="pre-campaign trend extrapolated")
    pre_level = float(observed[:n_pre].mean())
    ax.axhline(pre_level, color="#9aa5b1", linewidth=1.0, linestyle="-.")
    ax.annotate("pre-campaign average", xy=(dates[2], pre_level), xytext=(0, 5),
                textcoords="offset points", color="#616e7c", fontsize=9)

    slope_per_week = coef[0] * 7.0
    ax.text(
        0.015, 0.06,
        f"The series was already falling before the campaign: about\n"
        f"{slope_per_week:,.0f} streams per week over the pre-period.".replace("-", "−"),
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color="#3e4c59",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f7fa", edgecolor="#cbd2d9"),
    )
    _campaign_marker(ax, cdate)
    _fmt_dates(ax)
    ax.set_ylabel("daily streams")
    ax.set_title("Streams were already declining before the campaign",
                 fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    _save(fig, out_path)


def plot_regional_totals(names, mint_bottom, mint_top, out_path):
    """Per-region reconciled attributed streams that sum to the reconciled total."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [n.replace("_", " ") for n in names]
    vals = np.array(mint_bottom) / 1000.0
    x = np.arange(len(labels))
    ax.bar(x, vals, width=0.62, color=C_COUNTER, alpha=0.9)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.01, f"{v * 1000:,.0f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("reconciled attributed streams (thousands)")
    ax.set_title("Per-region attributed streams sum to the reconciled total",
                 fontweight="bold", fontsize=11)
    ax.text(
        0.5, 0.92,
        f"Reconciled total finance can trust: {mint_top:,.0f} extra streams.\n"
        f"The five regional figures add up to it exactly.",
        transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color="#3e4c59",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f7fa", edgecolor="#cbd2d9"),
    )
    ax.grid(axis="x", visible=False)
    _save(fig, out_path)


def plot_montecarlo(mc, out_path):
    """Monte Carlo distributions: incoherence gap and per-method absolute error."""
    gaps = np.array(mc["raw"]["gaps"]) / 1000.0
    methods = ["bottom_up", "aggregate_only", "mint"]
    labels = ["bottom-up", "aggregate-only", "reconciled\n(MinT)"]
    abserr = [np.array(mc["raw"]["abs_error"][m]) / 1000.0 for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    ax1.hist(gaps, bins=18, color=C_BAND, edgecolor="#5b7c95")
    ax1.axvline(0, color=C_BAD, linewidth=1.6, linestyle="--", label="coherent (gap = 0)")
    ax1.axvline(float(np.mean(gaps)), color=C_COUNTER, linewidth=1.6, label="mean gap")
    ax1.set_xlabel("incoherence gap: bottom-up sum minus aggregate (thousands)")
    ax1.set_ylabel("campaigns")
    ax1.set_title("Base estimates are incoherent", fontweight="bold", fontsize=11)
    ax1.legend(loc="upper left", frameon=False, fontsize=9)

    colors = ["#9aa5b1", "#c9a24b", C_COUNTER]
    bp = ax2.boxplot(abserr, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    for med in bp["medians"]:
        med.set_color("#1f2933")
    means = [float(np.mean(a)) for a in abserr]
    ax2.plot(np.arange(1, len(means) + 1), means, marker="o", linestyle="none",
             color="#1f2933", label="mean")
    ax2.set_ylabel("absolute error vs true total (thousands)")
    ax2.set_title("Reconciled has the lowest error", fontweight="bold", fontsize=11)
    ax2.legend(loc="upper right", frameon=False, fontsize=9)
    ax2.grid(axis="x", visible=False)

    _save(fig, out_path)
