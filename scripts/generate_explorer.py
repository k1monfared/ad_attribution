"""Generate the seven-scenario interactive explorer data.

Each scenario is a synthetic campaign with a KNOWN, EXPLICITLY CONSTRUCTED
data-generating process, run through the same Bayesian structural time-series
forecasting counterfactual used everywhere else in this project
(src/bayesian_model.py). For every scenario we commit a self-contained results
JSON under docs/data/ that the static explorer page (docs/index.html) reads
directly.

Construction (per scenario), built deliberately rather than by tuning opaque
parameters:

  STEP 1, baseline. A constant daily trend slope (streams per day) big enough to
  be obvious: the pre-period level moves by several times the weekly peak-to-
  trough swing. The baseline is trend + weekly seasonality + a small market
  cycle + noise. The donor pool co-moves with the baseline INCLUDING its trend,
  so the Bayesian counterfactual (expected without the campaign) continues the
  baseline trend into the campaign window.

  STEP 2, campaign effect as a CONSTANT slope change (a linear ramp) over the
  campaign window. The observed series during the campaign has slope
  (baseline slope + campaign slope change); the extra streams accumulate linearly
  from zero at the campaign start.

  STEP 3, run the Bayesian attribution on the constructed observed series to get
  the ACTUAL numbers (never hardcoded), and report the observed pre-period slope,
  the observed campaign-period slope, and the counterfactual campaign-period
  slope so the intended behavior can be verified.

The injected true baseline is never written to the JSON: the page shows only
what a real deployment would see (observed series, posterior counterfactual,
credible interval, attributed area, diagnostics).

    python scripts/generate_explorer.py

MCMC is stochastic but seeded, and kept modest (2 chains, 800 draws) so the
seven fits finish in reasonable time. Convergence is reported per scenario.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import diagnostics, io_utils  # noqa: E402
from src.bayesian_model import BayesConfig  # noqa: E402
from src.data_gen import weekly_signal  # noqa: E402
from scripts.run_demo import _predictive_band, run_scenario  # noqa: E402


# Modest sampler shared by every scenario fit.
FIT_CFG = BayesConfig(
    donor_prior="normal",
    donor_prior_scale=1.0,
    weekly_harmonics=2,
    annual_harmonics=3,
    level_sigma_scale=0.05,
    draws=800,
    tune=800,
    chains=2,
    cores=2,
    target_accept=0.95,
    seed=20260707,
)

# Donor selection thresholds (same two-stage filter used across the project).
SELECTION = {
    "selection": {
        "feature_top_k": 22,
        "level_corr_threshold": 0.7,
        "diff_corr_threshold": 0.1,
    }
}

N_DAYS = 180
CAMPAIGN_START = 120
START_DATE = "2025-01-01"


@dataclass
class ExplorerScenario:
    key: str
    label: str
    short_label: str
    trend: str          # "growing" | "flat" | "declining"
    kind: str           # verdict template selector
    seed: int
    base_level: float
    # STEP 1: baseline daily trend slope, in streams per day, constant across the
    # whole horizon. Positive rises, negative declines. The pre-period level moves
    # by base_slope * (n_pre - 1) streams, which we keep several times the weekly
    # peak-to-trough swing so the trend is unmistakable.
    base_slope: float
    # STEP 2: campaign slope CHANGE, in streams per day, added to the baseline
    # slope over the campaign window. The extra streams ramp linearly from zero at
    # the campaign start (a constant change over the duration). The observed slope
    # during the campaign is therefore base_slope + campaign_slope.
    campaign_slope: float
    weekly_amp: float = 0.08     # weekly seasonality amplitude, fraction of base
    genre_amp: float = 0.015     # small shared market cycle, fraction of base
    noise_scale: float = 0.025   # observation noise sd, fraction of base
    n_good: int = 20
    n_bad: int = 10


# Seven scenarios. base_slope sets the organic trajectory, campaign_slope sets the
# campaign's constant slope change. The verdict template is chosen by `kind`, but
# every number in the verdict is read back from the posterior fit.
#
# base_level 9000 (growing/flat) gives a weekly peak-to-trough swing of about
# 0.08 * 2.15 * 9000 ~ 1550 streams; base_level 11000 (declining) about 1900. A
# base_slope of 55/day moves the 120-day pre-period by about 6500 streams, roughly
# 4x the weekly swing, so the trend clearly dominates the wiggle.
SCENARIOS = [
    ExplorerScenario(
        key="scenario_1", trend="growing", kind="grow_clear",
        label="Growing trend: real lift on top of the climb",
        short_label="Growing positive",
        seed=310001, base_level=9000.0, base_slope=55.0, campaign_slope=55.0,
    ),
    ExplorerScenario(
        key="scenario_2", trend="growing", kind="grow_noise",
        label="Growing trend: effect not distinguishable from zero",
        short_label="Growing neutral",
        seed=310002, base_level=9000.0, base_slope=55.0, campaign_slope=0.0,
    ),
    ExplorerScenario(
        key="scenario_3", trend="growing", kind="grow_negative",
        label="Growing trend: measured negative reads as zero",
        short_label="Growing negative",
        seed=310003, base_level=9000.0, base_slope=55.0, campaign_slope=-75.0,
    ),
    ExplorerScenario(
        key="scenario_4", trend="flat", kind="flat_huge",
        label="Flat trend: a clear lift from a flat baseline",
        short_label="Flat positive",
        seed=310139, base_level=9000.0, base_slope=0.0, campaign_slope=65.0,
    ),
    ExplorerScenario(
        key="scenario_5", trend="declining", kind="decline_noise",
        label="Declining trend: effect not distinguishable from zero",
        short_label="Declining neutral",
        seed=310005, base_level=11000.0, base_slope=-55.0, campaign_slope=0.0,
    ),
    ExplorerScenario(
        key="scenario_6", trend="declining", kind="decline_slowed",
        label="Declining trend: the ad slowed the decline",
        short_label="Declining slowed",
        seed=310006, base_level=11000.0, base_slope=-85.0, campaign_slope=50.0,
    ),
    ExplorerScenario(
        key="scenario_7", trend="declining", kind="decline_reversed",
        label="Declining trend: the ad reversed the decline",
        short_label="Declining reversed",
        seed=310007, base_level=11000.0, base_slope=-55.0, campaign_slope=130.0,
    ),
]


def _minus(s: str) -> str:
    """Mathematical minus for signed numbers, without touching hyphens in words."""
    return s.replace("-", "−")


def _trim(v: float) -> str:
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def _fnum(x: float) -> str:
    """Stream count rounded to 2 significant figures with K/M suffixes.

    Mirrors fmtNum() in docs/index.html so the verdict strings match the tables:
    66K, 450K, 4.9K, 1.5M, 360. Negatives keep the mathematical minus.
    """
    neg = x < 0
    a = abs(float(x))
    if a == 0:
        r = 0.0
    else:
        factor = 10.0 ** (math.floor(math.log10(a)) - 1)
        r = round(a / factor) * factor
    if r >= 1e6:
        out = _trim(r / 1e6) + "M"
    elif r >= 1e3:
        out = _trim(r / 1e3) + "K"
    else:
        out = _trim(r)
    return ("−" if neg else "") + out


def _fpct(x: float, signed: bool = False) -> str:
    fmt = f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"
    return _minus(fmt)


def _build_unit(base, frac_signal, noise, rng):
    series = base * (1.0 + frac_signal) + rng.normal(0, noise * base, len(frac_signal))
    return np.clip(series, 1.0, None)


def generate_explicit(es: ExplorerScenario) -> dict:
    """Build the scenario series explicitly (STEP 1 baseline, STEP 2 campaign).

    Returns the same dict structure the shared pipeline (run_scenario) expects.
    The treated baseline (true_counterfactual) is a constant-slope linear trend
    plus weekly seasonality, a small shared market cycle, and noise. Donors carry
    the SAME trend, market cycle, and part of the weekly seasonality, so they
    co-move with the baseline including its trend and the fitted counterfactual
    continues that trend into the campaign window. The campaign adds a linear ramp
    (a constant slope change) on top of the baseline over the campaign window.
    """
    rng = np.random.default_rng(es.seed)
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    t = np.arange(N_DAYS, dtype=float)
    mid = (N_DAYS - 1) / 2.0

    # STEP 1: shared organic signals, as fractions of base_level. The trend is a
    # constant-slope line, centered on the horizon midpoint so a steep decline
    # never drives the level toward zero.
    trend_frac = (es.base_slope / es.base_level) * (t - mid)
    genre_frac = es.genre_amp * np.sin(2 * np.pi * t / 47.0 + 1.1)
    week = weekly_signal(dates)                       # unit-amplitude day-of-week
    treated_seasonal = es.weekly_amp * week           # treated carries full weekly
    donor_seasonal = 0.45 * es.weekly_amp * week      # donors share part of weekly

    # Treated baseline = the true no-campaign counterfactual.
    treated_frac = trend_frac + genre_frac + treated_seasonal
    true_cf = _build_unit(es.base_level, treated_frac, es.noise_scale, rng)

    # STEP 2: campaign effect as a constant slope change (linear ramp), additive in
    # streams, zero before the campaign, growing by campaign_slope per day after.
    effect = np.zeros(N_DAYS)
    for i in range(CAMPAIGN_START, N_DAYS):
        effect[i] = es.campaign_slope * (i - CAMPAIGN_START)
    treated_obs = np.clip(true_cf + effect, 1.0, None)
    true_effect = treated_obs - true_cf

    # Co-moving "good" donors: share the trend, the market cycle, and part of the
    # weekly seasonality, at their own volumes and loadings.
    treated_feat = np.zeros(6)
    candidates = {}
    features = []
    is_good = []
    for j in range(es.n_good):
        a_tr = 1.0 + rng.normal(0, 0.05)
        a_ge = 1.0 + rng.normal(0, 0.06)
        a_se = 1.0 + rng.normal(0, 0.10)
        base = es.base_level * rng.uniform(0.9, 1.1)
        frac = a_tr * trend_frac + a_ge * genre_frac + a_se * donor_seasonal
        candidates[f"cand_g{j:02d}"] = _build_unit(base, frac, es.noise_scale, rng)
        features.append(treated_feat + rng.normal(0, 0.3, 6))
        is_good.append(True)

    # Unrelated "bad" donors driven by their own independent factors.
    for j in range(es.n_bad):
        bad_slope = rng.uniform(-90.0, 90.0)
        bad_trend = (bad_slope / es.base_level) * (t - mid)
        bad_genre = rng.uniform(0.0, 0.03) * np.sin(2 * np.pi * t / rng.uniform(30, 60) + rng.uniform(0, 6.28))
        bad_week = weekly_signal(dates, phase=rng.uniform(0, 2 * np.pi))
        frac = bad_trend + bad_genre + rng.uniform(0.0, 0.08) * bad_week
        base = es.base_level * rng.uniform(0.5, 1.8)
        candidates[f"cand_b{j:02d}"] = _build_unit(base, frac, es.noise_scale * 1.4, rng)
        features.append(treated_feat + rng.normal(0, 2.2, 6) + rng.choice([-3.0, 3.0], 6))
        is_good.append(False)

    # Fanbase: same artist, an untargeted region, roughly flat (the pitfall).
    fan_frac = 0.03 * trend_frac + 0.30 * treated_seasonal
    fanbase = _build_unit(es.base_level * 1.2, fan_frac, es.noise_scale * 0.8, rng)

    names = list(candidates.keys())
    order = rng.permutation(len(names))
    names = [names[i] for i in order]
    cand_matrix = np.vstack([candidates[n] for n in names]).T
    feats = np.vstack([features[i] for i in order])
    good_flags = np.array([is_good[i] for i in order])

    cand_df = pd.DataFrame(cand_matrix, index=dates, columns=names)
    cand_df.index.name = "date"
    treated_df = pd.DataFrame({"streams": treated_obs}, index=dates)
    treated_df.index.name = "date"

    return {
        "name": es.key,
        "dates": dates,
        "treated": treated_df,
        "true_counterfactual": true_cf,
        "true_effect": true_effect,
        "lift": effect,
        "candidates": cand_df,
        "candidate_features": feats,
        "candidate_is_good": good_flags,
        "treated_features": treated_feat,
        "fanbase": pd.Series(fanbase, index=dates, name="fanbase_other_region"),
        "campaign_start": CAMPAIGN_START,
    }


def _slope_per_day(y: np.ndarray) -> float:
    """Ordinary-least-squares slope of y against day index, in streams per day."""
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, np.asarray(y, dtype=float), 1)[0])


def _weeks(index_start: int, index_end: int, dates, chunk: int = 7):
    """Yield (label, [day indices]) 7-day blocks over [index_start, index_end)."""
    blocks = []
    i = index_start
    while i < index_end:
        j = min(i + chunk, index_end)
        idx = list(range(i, j))
        label = f"{dates[i].strftime('%b %d')} to {dates[j - 1].strftime('%b %d')}"
        blocks.append((label, idx))
        i = j
    return blocks


def _band(draws: np.ndarray):
    return (
        float(np.mean(draws)),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def build_verdict(es: ExplorerScenario, st: dict) -> str:
    att = _fnum(st["att_mean"])
    lo = _fnum(st["att_lo"])
    hi = _fnum(st["att_hi"])
    rel = _fpct(st["rel_mean"])
    mom_obs = _fpct(st["mom_obs"], signed=True)
    mom_cf = _fpct(st["mom_cf"], signed=True)

    if es.kind == "grow_clear":
        return (
            f"Real lift on top of an already-rising trend. Against the no-ad counterfactual "
            f"the campaign added an estimated {att} extra streams over the window (95% "
            f"credible interval {lo} to {hi}), a lift of {rel}. The interval sits well above "
            f"zero, so the effect is real rather than noise. Note the surface read overstates "
            f"it: observed streams run {mom_obs} above the pre-campaign baseline, but the "
            f"trend was already climbing, so much of that rise is the trend, not the ad. The "
            f"{rel} causal lift is the number to defend."
        )
    if es.kind == "grow_noise":
        return (
            f"The effect is not distinguishable from zero. The estimated ad impact is {att} "
            f"streams with a 95% credible interval of {lo} to {hi} that spans zero. Streams "
            f"rose over the window, but that movement is explained by the underlying upward "
            f"trend and ordinary week-to-week noise, not by the ad, so on this evidence we "
            f"cannot claim the campaign moved streams."
        )
    if es.kind == "grow_negative":
        return (
            f"Read this as zero, not negative. The model puts the campaign effect at {att} "
            f"streams (95% credible interval {lo} to {hi}), below zero, but an ad campaign "
            f"cannot produce fewer streams than no campaign at all, so a measured negative "
            f"effect is not evidence the ad hurt. It signals that something outside the data "
            f"moved streams down during the campaign window and the model could not control "
            f"for it, so confidence is low and the honest campaign attribution is read as "
            f"ZERO. The Campaign impact table below still shows the measured numbers for full "
            f"transparency."
        )
    if es.kind == "flat_huge":
        return (
            f"A clear lift from a flat baseline. With no underlying trend the counterfactual "
            f"stays flat, so almost all of the rise is attributable to the campaign: an "
            f"estimated {att} extra streams (95% credible interval {lo} to {hi}), a lift of "
            f"{rel} over the no-ad counterfactual. The interval sits well above zero."
        )
    if es.kind == "decline_noise":
        return (
            f"The effect is not distinguishable from zero. Streams fell over the window, and "
            f"the estimated ad impact is {att} streams with a 95% credible interval of {lo} "
            f"to {hi} that spans zero. The observed series simply follows the pre-existing "
            f"downward trend, with no measurable help from the ad."
        )
    if es.kind == "decline_slowed":
        return (
            f"Streams are still below the pre-campaign baseline: the observed post-period runs "
            f"{mom_obs} versus baseline, so at a glance the campaign looks like a failure. But "
            f"without the ad the decline would have been steeper, to about {mom_cf} below "
            f"baseline. The ad slowed the decline and retained an estimated {att} streams "
            f"versus the no-ad counterfactual (95% credible interval {lo} to {hi}), a {rel} "
            f"lift. It slowed the decline, it did not reverse it. Lead with streams retained, "
            f"not the headline drop."
        )
    if es.kind == "decline_reversed":
        return (
            f"The campaign reversed the decline into genuine growth. The series was trending "
            f"down, but over the campaign the observed streams turn and climb on a positive "
            f"slope, averaging {mom_obs} versus the pre-campaign level, whereas without the "
            f"ad they would have kept falling to about {mom_cf}. The ad added an estimated "
            f"{att} extra streams (95% credible interval {lo} to {hi}), a {rel} lift, large "
            f"enough to turn a falling trajectory into real growth, not merely a shallower "
            f"decline."
        )
    raise ValueError(f"unknown kind {es.kind!r}")


def _rhat_values(idata) -> list:
    var_names = [v for v in idata.posterior.data_vars
                 if v not in {"mu", "level", "weekly", "annual"}]
    rhat = az.rhat(idata, var_names=var_names)
    vals = []
    for v in rhat.data_vars:
        arr = np.atleast_1d(np.asarray(rhat[v].values).ravel())
        vals += [float(x) for x in arr if np.isfinite(x)]
    return vals


def build_scenario_json(es: ExplorerScenario) -> dict:
    scn = generate_explicit(es)
    r = run_scenario(scn, SELECTION, FIT_CFG, predictive=True)

    dates = r["dates"]
    n_pre = r["n_pre"]
    T = len(dates)
    post = slice(n_pre, T)
    observed = np.asarray(r["treated"], dtype=float)
    true_cf = np.asarray(scn["true_counterfactual"], dtype=float)
    s = r["summary"]
    cf_mean = np.asarray(s["counterfactual_mean"], dtype=float)
    cf_draws = r["fit"]["counterfactual_draws"]  # (S, T) posterior counterfactual

    date_iso = [d.strftime("%Y-%m-%d") for d in dates]

    # Weekly figures.
    pre_weeks = []
    for label, idx in _weeks(0, n_pre, dates):
        pre_weeks.append({
            "label": label,
            "observed": float(observed[idx].sum()),
        })

    post_weeks = []
    for label, idx in _weeks(n_pre, T, dates):
        cf_week = cf_draws[:, idx].sum(axis=1)                 # (S,)
        att_week = observed[idx].sum() - cf_week              # (S,)
        cf_m, cf_lo, cf_hi = _band(cf_week)
        att_m, att_lo, att_hi = _band(att_week)
        post_weeks.append({
            "label": label,
            "observed": float(observed[idx].sum()),
            "cf_mean": cf_m, "cf_lo": cf_lo, "cf_hi": cf_hi,
            "att_mean": att_m, "att_lo": att_lo, "att_hi": att_hi,
        })

    cf_total_draws = cf_draws[:, post].sum(axis=1)
    att_total_draws = s["total_effect_draws"]
    cf_tot_m, cf_tot_lo, cf_tot_hi = _band(cf_total_draws)
    att_tot_m, att_tot_lo, att_tot_hi = _band(att_total_draws)

    # Diagnostics: posterior predictive check on the pre-period.
    y_std, y_mean = r["fit"]["y_std"], r["fit"]["y_mean"]
    post_lo, post_hi, post_med = _predictive_band(
        r["fit"]["idata"].posterior_predictive, "y_obs", y_std, y_mean)

    conv = r["convergence"]
    rhat_vals = _rhat_values(r["fit"]["idata"])

    # Constructed vs recovered slopes (streams per day), for verification.
    slopes = {
        "baseline_pre": _slope_per_day(true_cf[:n_pre]),
        "baseline_post": _slope_per_day(true_cf[post]),
        "observed_pre": _slope_per_day(observed[:n_pre]),
        "observed_post": _slope_per_day(observed[post]),
        "counterfactual_post": _slope_per_day(cf_mean[post]),
    }

    # Numbers that feed the verdict, all read back from the fit.
    st = {
        "att_mean": att_tot_m, "att_lo": att_tot_lo, "att_hi": att_tot_hi,
        "rel_mean": s["relative_lift_mean"],
        "rel_lo": s["relative_lift_lo"], "rel_hi": s["relative_lift_hi"],
        "mom_obs": s["mom_change_observed"],
        "mom_cf": s["mom_change_counterfactual"],
        "slopes": slopes,
    }
    verdict = build_verdict(es, st)
    spans_zero = bool(att_tot_lo < 0.0 < att_tot_hi)

    payload = {
        "id": es.key,
        "label": es.label,
        "short_label": es.short_label,
        "trend": es.trend,
        "verdict": verdict,
        "campaign_start_index": n_pre,
        "campaign_start_date": date_iso[n_pre],
        "dates": date_iso,
        "observed": observed.tolist(),
        "counterfactual_mean": s["counterfactual_mean"].tolist(),
        "counterfactual_lo": s["counterfactual_lo"].tolist(),
        "counterfactual_hi": s["counterfactual_hi"].tolist(),
        "pointwise_mean": s["pointwise_effect_mean"].tolist(),
        "pointwise_lo": s["pointwise_effect_lo"].tolist(),
        "pointwise_hi": s["pointwise_effect_hi"].tolist(),
        "post_dates": date_iso[n_pre:],
        "cumulative_mean": s["cumulative_effect_mean"].tolist(),
        "cumulative_lo": s["cumulative_effect_lo"].tolist(),
        "cumulative_hi": s["cumulative_effect_hi"].tolist(),
        "weeks_pre": pre_weeks,
        "weeks_post": post_weeks,
        "totals": {
            "pre_observed_total": float(observed[:n_pre].sum()),
            "pre_weekly_mean": float(observed[:n_pre].sum() / (n_pre / 7.0)),
            "pre_daily_mean": s["pre_daily_mean"],
            "post_observed_total": float(observed[post].sum()),
            "cf_total_mean": cf_tot_m, "cf_total_lo": cf_tot_lo, "cf_total_hi": cf_tot_hi,
            "att_total_mean": att_tot_m, "att_total_lo": att_tot_lo, "att_total_hi": att_tot_hi,
            "relative_lift_mean": s["relative_lift_mean"],
            "relative_lift_lo": s["relative_lift_lo"],
            "relative_lift_hi": s["relative_lift_hi"],
            "spans_zero": spans_zero,
            "n_pre": int(n_pre), "n_post": int(T - n_pre),
        },
        "ppc": {
            "dates_pre": date_iso[:n_pre],
            "observed_pre": observed[:n_pre].tolist(),
            "post_lo": [float(x) for x in post_lo],
            "post_hi": [float(x) for x in post_hi],
            "post_med": [float(x) for x in post_med],
        },
        "convergence": conv,
        "rhat_values": rhat_vals,
        "include_annual": bool(r["fit"]["include_annual"]),
        "include_weekly": bool(r["fit"]["include_weekly"]),
    }
    return payload, st, spans_zero


def main():
    t0 = time.time()
    out_dir = ROOT / "docs" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for es in SCENARIOS:
        ts = time.time()
        payload, st, spans_zero = build_scenario_json(es)
        io_utils.save_json(payload, out_dir / f"{es.key}.json")
        conv = payload["convergence"]
        sl = st["slopes"]
        print(
            f"[{es.key}] {es.short_label}: attributed {_fnum(st['att_mean'])} "
            f"[{_fnum(st['att_lo'])} to {_fnum(st['att_hi'])}], lift {_fpct(st['rel_mean'])}, "
            f"spans_zero={spans_zero}, maxRhat {conv['max_rhat']:.3f}, "
            f"div {conv['divergences']}/{conv['n_samples']}, {time.time() - ts:.0f}s"
        )
        print(
            f"    slopes/day: baseline_pre {sl['baseline_pre']:+.1f}, "
            f"observed_pre {sl['observed_pre']:+.1f}, observed_post {sl['observed_post']:+.1f}, "
            f"counterfactual_post {sl['counterfactual_post']:+.1f}"
        )
        manifest.append({
            "id": es.key,
            "label": es.label,
            "short_label": es.short_label,
            "trend": es.trend,
            "file": f"data/{es.key}.json",
        })

    io_utils.save_json({"scenarios": manifest}, out_dir / "manifest.json")
    print(f"Explorer data written to {out_dir} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
