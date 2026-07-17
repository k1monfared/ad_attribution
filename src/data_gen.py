"""Synthetic data generation for the marketing-attribution demos.

The generator builds, for a given scenario:

* a treated unit (an artist or song promoted to a target region or audience),
* a pool of candidate donors made of SIMILAR artists or songs whose audiences
  move together with the treated unit through purely organic market changes,
* a set of unrelated "bad" candidates that do not co-move, so donor selection
  has something real to filter out,
* a same-artist other-region series that is roughly FLAT. This is the fanbase
  pitfall: the artist's existing, loyal fanbase in an untargeted region does
  not track how the TARGET audience would have moved, so it is a biased
  control even though it is the same artist.

On top of the shared market structure the treated unit carries its own
SEASONAL structure that the Bayesian forecasting model has to handle:

* weekly seasonality (a day-of-week pattern),
* annual / day-of-year seasonality (a smooth yearly cycle plus a few holiday
  spikes anchored to calendar dates).

Donors share only PART of this seasonality (the market-wide part), so the
treated unit keeps a treated-specific seasonal signal that the donor regression
cannot absorb. A model that ignores seasonality therefore mis-forecasts the
counterfactual, which is exactly what the seasonality-blind comparison shows.

A campaign effect with a KNOWN shape and size is injected into the treated
unit over the post-campaign window, so every estimate can be scored against
ground truth.

Two headline scenarios ship with the demo:

* growth: a stable or rising series that the campaign lifts further,
* decline: a series trending down (often the very reason a campaign is run).

Everything operates on aggregated daily streams. There is no individual-level
data anywhere in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "genre_x",
    "genre_y",
    "tempo",
    "catalog_age",
    "momentum",
    "audience_age",
]

# Calendar holidays as (month, day, fractional bump). These are day-of-year
# effects: market-wide listening spikes anchored to the calendar, not to the
# campaign. Widths are a couple of days (a small Gaussian bump around the date).
HOLIDAYS = [
    (1, 1, 0.16),    # New Year
    (2, 14, 0.10),   # Valentine's
    (7, 4, 0.14),
    (10, 31, 0.11),
    (12, 25, 0.20),  # winter holiday
]


@dataclass
class ScenarioConfig:
    seed: int = 20260702
    name: str = "growth"
    n_days: int = 180
    campaign_start: int = 120  # first post-campaign day index (0-based)
    start_date: str = "2025-01-01"
    n_good: int = 12  # co-moving similar-artist candidates
    n_bad: int = 10  # unrelated candidates that should be filtered out
    trend_total: float = 0.15  # fractional level change over the horizon
    peak_lift: float = 0.12  # sustained campaign lift at the plateau
    ramp_days: int = 10
    noise_scale: float = 0.035
    base_level: float = 9000.0
    # Seasonality. The treated unit carries the full amplitude; donors share
    # only a fraction, leaving a treated-specific residual the model must fit.
    weekly_amp: float = 0.08
    annual_amp: float = 0.05
    holiday_amp: float = 1.0  # global multiplier on the HOLIDAYS bumps
    donor_weekly_frac: float = 0.45
    # Donors carry most of the annual / holiday structure. With a short pre-period
    # the annual component is not fitted (see the inclusion rule), so donors must
    # carry it or the omission would bias the counterfactual. Weekly seasonality
    # is deliberately only partly shared, so it stays the differentiator in the
    # seasonality-aware vs seasonality-blind comparison.
    donor_annual_frac: float = 0.85
    donor_holiday_frac: float = 0.70


def lift_profile(n_days, campaign_start, peak_lift, ramp_days):
    """Multiplicative lift per day, zero before the campaign, ramped then held."""
    lift = np.zeros(n_days)
    for t in range(campaign_start, n_days):
        frac = min(1.0, (t - campaign_start + 1) / max(1, ramp_days))
        lift[t] = peak_lift * frac
    return lift


def weekly_signal(dates: pd.DatetimeIndex, phase: float = -0.6) -> np.ndarray:
    """Unit-amplitude day-of-week pattern (two harmonics)."""
    dow = dates.dayofweek.to_numpy()
    return (
        np.sin(2 * np.pi * dow / 7 + phase)
        + 0.4 * np.cos(2 * np.pi * dow / 7)
    )


def annual_signal(dates: pd.DatetimeIndex, phase: float = 1.1) -> np.ndarray:
    """Unit-amplitude smooth annual cycle keyed to day-of-year."""
    doy = dates.dayofyear.to_numpy()
    return np.sin(2 * np.pi * doy / 365.25 + phase)


def holiday_signal(dates: pd.DatetimeIndex, amp: float = 1.0) -> np.ndarray:
    """Sum of small Gaussian bumps anchored to calendar holidays."""
    out = np.zeros(len(dates))
    doy = dates.dayofyear.to_numpy().astype(float)
    for month, day, bump in HOLIDAYS:
        anchor = pd.Timestamp(year=int(dates[0].year), month=month, day=day).dayofyear
        # Distance on the yearly circle so December and January are close.
        d = np.abs(doy - anchor)
        d = np.minimum(d, 365.25 - d)
        out += amp * bump * np.exp(-0.5 * (d / 1.5) ** 2)
    return out


def _market_factors(rng, n_days, trend_total):
    """Shared organic market signals expressed as fractional deviations.

    A slow trend (deterministic drift plus a random walk) and a mid-frequency
    genre cycle. Weekly and annual seasonality are added per unit so their
    amplitude can differ between the treated unit and the donors.
    """
    t = np.arange(n_days)
    frac = t / (n_days - 1)
    trend = trend_total * frac + np.cumsum(rng.normal(0, 0.004, n_days))
    trend = trend - trend[0]
    genre = 0.07 * np.sin(2 * np.pi * t / 47 + 1.1)
    return trend, genre


def _build_unit(rng, base, frac_signal, noise_scale):
    level = base * (1.0 + frac_signal)
    series = level + rng.normal(0, noise_scale * base, len(frac_signal))
    return np.clip(series, 1.0, None)


def generate_scenario(cfg: ScenarioConfig):
    rng = np.random.default_rng(cfg.seed)
    n_days = cfg.n_days
    dates = pd.date_range(cfg.start_date, periods=n_days, freq="D")

    trend, genre = _market_factors(rng, n_days, cfg.trend_total)
    week = weekly_signal(dates)
    ann = annual_signal(dates)
    hol = holiday_signal(dates, amp=cfg.holiday_amp)

    # Treated unit and its true (no-campaign) counterfactual. The treated unit
    # carries the full seasonal amplitude.
    treated_seasonal = cfg.weekly_amp * week + cfg.annual_amp * ann + hol
    treated_frac = trend + genre + treated_seasonal
    true_cf = _build_unit(rng, cfg.base_level, treated_frac, cfg.noise_scale)

    lift = lift_profile(n_days, cfg.campaign_start, cfg.peak_lift, cfg.ramp_days)
    treated_obs = true_cf * (1.0 + lift)
    true_effect = treated_obs - true_cf

    treated_feat = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    candidates = {}
    features = []
    is_good = []

    # Donors share the trend, the genre cycle, and PART of the seasonality.
    donor_seasonal_base = (
        cfg.donor_weekly_frac * cfg.weekly_amp * week
        + cfg.donor_annual_frac * cfg.annual_amp * ann
        + cfg.donor_holiday_frac * hol
    )

    for j in range(cfg.n_good):
        a_tr = 1.0 + rng.normal(0, 0.05)
        a_ge = 1.0 + rng.normal(0, 0.06)
        a_se = 1.0 + rng.normal(0, 0.10)
        base = cfg.base_level * rng.uniform(0.9, 1.1)
        frac = a_tr * trend + a_ge * genre + a_se * donor_seasonal_base
        series = _build_unit(rng, base, frac, cfg.noise_scale)
        candidates[f"cand_g{j:02d}"] = series
        features.append(treated_feat + rng.normal(0, 0.3, 6))
        is_good.append(True)

    # Bad donors: unrelated units driven by their own independent factors.
    for j in range(cfg.n_bad):
        bad_trend, bad_genre = _market_factors(rng, n_days, trend_total=rng.uniform(-0.4, 0.4))
        bad_week = weekly_signal(dates, phase=rng.uniform(0, 2 * np.pi))
        frac = (
            rng.uniform(0.3, 1.2) * bad_trend
            + rng.uniform(0.0, 1.0) * bad_genre
            + rng.uniform(0.0, 0.08) * bad_week
        )
        base = cfg.base_level * rng.uniform(0.5, 1.8)
        series = _build_unit(rng, base, frac, cfg.noise_scale * 1.4)
        candidates[f"cand_b{j:02d}"] = series
        features.append(treated_feat + rng.normal(0, 2.2, 6) + rng.choice([-3.0, 3.0], 6))
        is_good.append(False)

    # Fanbase series: same artist, an untargeted region, roughly flat.
    fan_frac = 0.03 * trend + 0.30 * (cfg.weekly_amp * week)
    fanbase = _build_unit(rng, cfg.base_level * 1.2, fan_frac, cfg.noise_scale * 0.8)

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
        "name": cfg.name,
        "dates": dates,
        "treated": treated_df,
        "true_counterfactual": true_cf,
        "true_effect": true_effect,
        "lift": lift,
        "candidates": cand_df,
        "candidate_features": feats,
        "candidate_is_good": good_flags,
        "treated_features": treated_feat,
        "fanbase": pd.Series(fanbase, index=dates, name="fanbase_other_region"),
        "campaign_start": cfg.campaign_start,
    }


@dataclass
class RegionsConfig:
    """Multi-region campaign with deliberate heterogeneity across regions.

    Regions differ in volume, true campaign lift, donor availability, and donor
    quality. Donor-poor regions (few donors, looser fit) have high-variance
    per-region estimates. The aggregate is estimated from a separate, rich,
    clean national donor pool. Because the per-region pools and the national
    pool are different and of uneven quality, the independent per-region and
    aggregate estimates disagree (incoherence), and the noisy regions distort
    the bottom-up sum. Reconciliation repairs both.
    """

    seed: int = 909090
    n_days: int = 180
    campaign_start: int = 120
    ramp_days: int = 10
    start_date: str = "2025-01-01"
    trend_total: float = -0.20
    donor_noise: float = 0.02
    base_idio: float = 0.005  # small region-specific component present everywhere
    weekly_amp: float = 0.06
    mc_runs: int = 40
    region_names: list = field(
        default_factory=lambda: ["region_1", "region_2", "region_3", "region_4", "region_5"]
    )
    volumes: list = field(default_factory=lambda: [12000.0, 9000.0, 7000.0, 5000.0, 3000.0])
    true_lifts: list = field(default_factory=lambda: [0.10, 0.14, 0.12, 0.18, 0.25])
    donor_counts: list = field(default_factory=lambda: [16, 16, 4, 14, 4])
    donor_quality: list = field(default_factory=lambda: [0.05, 0.05, 0.22, 0.06, 0.26])
    national_donors: int = 10
    national_quality: float = 0.10


def _region_idio(rng, n_days, amp):
    """Region-specific component (not shared with the donor pool)."""
    t = np.arange(n_days)
    comp = np.sin(2 * np.pi * t / rng.uniform(28, 64) + rng.uniform(0, 2 * np.pi))
    comp = comp + 0.6 * np.cumsum(rng.normal(0, 0.06, n_days))
    comp = comp - comp.mean()
    comp = comp / (comp.std() + 1e-9)
    return amp * comp


def _make_donor_pool(rng, n_donors, base_level, g, quality, base_idio, donor_noise):
    """A donor pool whose loadings spread around the treated loadings by `quality`."""
    g_trend, g_genre, g_week = g
    n_days = len(g_trend)
    cols = {}
    for a in range(n_donors):
        a_tr = 1.0 + rng.normal(0, quality)
        a_ge = 1.0 + rng.normal(0, quality)
        a_se = 1.0 + rng.normal(0, quality)
        d_base = base_level * rng.uniform(0.9, 1.1)
        d_idio = _region_idio(rng, n_days, base_idio)
        signal = a_tr * g_trend + a_ge * g_genre + a_se * g_week + d_idio
        series = d_base * (1.0 + signal) + rng.normal(0, donor_noise * d_base, n_days)
        cols[f"arch_{a:02d}"] = np.clip(series, 1.0, None)
    return cols


def generate_region_campaign(cfg: RegionsConfig, seed: int):
    """One multi-region campaign realization.

    Global market factors (trend, genre cycle, weekly seasonality) are shared by
    all regions and all donors. Each region has its own donor pool whose quality
    varies. The aggregate is estimated from a separate, clean national donor
    pool. Because the per-region pools and the national pool differ, the
    independent per-region and aggregate estimates disagree.
    """
    rng = np.random.default_rng(seed)
    n_days, cs = cfg.n_days, cfg.campaign_start
    dates = pd.date_range(cfg.start_date, periods=n_days, freq="D")

    g_trend, g_genre = _market_factors(rng, n_days, cfg.trend_total)
    g_week = cfg.weekly_amp * weekly_signal(dates)
    g = (g_trend, g_genre, g_week)
    g_signal = g_trend + g_genre + g_week  # treated loadings all 1

    per_region = {}
    true_effects = {}
    names = []

    for r, name in enumerate(cfg.region_names):
        vol = cfg.volumes[r]
        lift_r = cfg.true_lifts[r]

        idio = _region_idio(rng, n_days, cfg.base_idio)
        true_cf = vol * (1.0 + g_signal + idio) + rng.normal(0, cfg.donor_noise * vol, n_days)
        true_cf = np.clip(true_cf, 1.0, None)
        lift = lift_profile(n_days, cs, lift_r, cfg.ramp_days)
        treated_obs = true_cf * (1.0 + lift)

        donor_cols = _make_donor_pool(
            rng, cfg.donor_counts[r], vol, g, cfg.donor_quality[r], cfg.base_idio, cfg.donor_noise
        )

        post = slice(cs, n_days)
        true_effects[name] = float((treated_obs[post] - true_cf[post]).sum())
        per_region[name] = {
            "treated": treated_obs,
            "donors": pd.DataFrame(donor_cols, index=dates),
            "true_counterfactual": true_cf,
            "true_effect": treated_obs - true_cf,
            "dates": dates,
            "campaign_start": cs,
        }
        names.append(name)

    total_vol = float(sum(cfg.volumes))
    national_cols = _make_donor_pool(
        rng, cfg.national_donors, total_vol, g, cfg.national_quality, cfg.base_idio, cfg.donor_noise
    )
    national_donors = pd.DataFrame(national_cols, index=dates)

    return {
        "per_region": per_region,
        "names": names,
        "national_donors": national_donors,
        "true_effects": true_effects,
        "true_total": float(sum(true_effects.values())),
        "campaign_start": cs,
    }
