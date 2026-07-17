"""Run the multi-region reconciliation Monte Carlo on its own and print results.

Fits a Bayesian counterfactual per region and for the national aggregate over
many simulated campaigns, then reports, for bottom-up, aggregate-only, and MinT
reconciliation, the incoherence gap of the base estimates and each method's
error against the known true total.

    python scripts/run_montecarlo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import aggregate, io_utils  # noqa: E402
from scripts.generate_data import build_all  # noqa: E402
from scripts.run_demo import make_bayes_cfg  # noqa: E402


def main():
    cfg, _, regions_cfg, _ = build_all()
    fit_cfg = make_bayes_cfg(cfg, draws=400, tune=400)

    il = aggregate.run_illustrative(regions_cfg, fit_cfg)
    print("=== illustrative campaign (fixed seed) ===")
    print(f"  true total            : {il['true_top']:,.0f}")
    print(f"  aggregate estimate    : {il['base']['top']:,.0f}")
    print(f"  bottom-up sum         : {il['base']['bottom_sum']:,.0f}")
    print(f"  incoherence gap       : {il['incoherence_gap']:,.0f}")
    for m, label in (("bottom_up", "bottom-up"), ("aggregate_only", "aggregate-only"), ("mint", "reconciled MinT")):
        print(f"  {label:16s} total {il[m]['top']:,.0f}  error {il[m]['error_vs_true_top']:+,.0f}")

    mc = aggregate.monte_carlo(regions_cfg, regions_cfg.mc_runs, fit_cfg)
    print(f"\n=== Monte Carlo over {mc['n_runs']} campaigns (true total {mc['true_top']:,.0f}) ===")
    g = mc["incoherence_gap"]
    print(f"  incoherence gap: mean {g['mean']:,.0f}, std {g['std']:,.0f}, "
          f"mean abs {np.mean(np.abs(mc['raw']['gaps'])):,.0f}")
    for m, label in (("bottom_up", "bottom-up"), ("aggregate_only", "aggregate-only"), ("mint", "reconciled MinT")):
        print(f"  {label:16s} mean|err| {mc['abs_error_mean'][m]:,.0f}  "
              f"signed {mc['signed_error'][m]['mean']:+,.0f}  std {mc['signed_error'][m]['std']:,.0f}")

    out = {k: v for k, v in mc.items() if k != "raw"}
    io_utils.save_json(
        {"illustrative": il, "monte_carlo": out},
        ROOT / "outputs" / "reconciliation_montecarlo.json",
    )
    print(f"\nWrote {ROOT / 'outputs' / 'reconciliation_montecarlo.json'}")


if __name__ == "__main__":
    main()
