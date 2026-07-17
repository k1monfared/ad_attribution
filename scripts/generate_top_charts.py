"""Generate the three extra charts for the top README answer blocks.

Everything here is computed from the COMMITTED run, no new MCMC and no injected
true line:

* growth block: the naive pre-vs-post delta (which credits the ad with the whole
  rise) against the counterfactual attribution (only the gap above the projected
  trend), so the naive number is visibly an overstatement,
* decline block: the observed series with its fitted pre-campaign downward trend,
  so the reader sees streams were already falling before the campaign,
* regional block: a bar chart of the per-region reconciled attributed streams
  that sum to the reconciled total finance can trust.

    python scripts/generate_top_charts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import plotting  # noqa: E402


def _load_series(name: str):
    df = pd.read_csv(ROOT / "data" / name / "treated_streams.csv", parse_dates=["date"])
    return df["date"], df["streams"].to_numpy()


def main():
    import json

    with open(ROOT / "outputs" / "attribution_results.json") as f:
        res = json.load(f)
    cfg = res["config"]
    n_pre = cfg["common"]["campaign_start"]
    img = ROOT / "docs" / "images"

    # Block 1: growth, naive pre-post vs counterfactual attribution.
    _, growth = _load_series("growth")
    n_post = len(growth) - n_pre
    pre_daily_mean = float(growth[:n_pre].mean())
    observed_post_total = float(growth[n_pre:].sum())
    naive_total = observed_post_total - pre_daily_mean * n_post
    cf_total = float(res["growth"]["cumulative_effect_mean"])
    plotting.plot_naive_vs_counterfactual(naive_total, cf_total, img / "top_growth_naive_vs_cf.png")

    # Block 2: decline, pre-campaign downward trend.
    ddates, decline = _load_series("decline")
    plotting.plot_decline_pretrend(ddates, decline, n_pre, img / "top_decline_pretrend.png")

    # Block 3: regional reconciled totals.
    il = res["multi_region"]["illustrative"]
    plotting.plot_regional_totals(
        il["names"], il["mint"]["bottom"], il["mint"]["top"], img / "top_regional_totals.png"
    )

    print(
        f"Top-section charts written to {img}: "
        f"naive {naive_total:,.0f} vs counterfactual {cf_total:,.0f}, "
        f"reconciled total {il['mint']['top']:,.0f}"
    )


if __name__ == "__main__":
    main()
