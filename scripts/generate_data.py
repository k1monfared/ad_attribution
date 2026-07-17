"""Generate and commit the synthetic inputs for the demo.

Writes the treated series, candidate donor pools, and fanbase series for the
scenarios (growth, decline, and the long-history annual-demo), plus the
multi-region panels. Run standalone or via run_demo.py.

    python scripts/generate_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import io_utils  # noqa: E402
from src.data_gen import (  # noqa: E402
    RegionsConfig,
    ScenarioConfig,
    generate_region_campaign,
    generate_scenario,
)


def _scenario_cfg(cfg, key, name):
    c = cfg[key]
    common = cfg["common"]
    n_days = c.get("n_days", common["n_days"])
    campaign_start = c.get("campaign_start", common["campaign_start"])
    base = ScenarioConfig()
    return ScenarioConfig(
        seed=c["seed"], name=name, n_days=n_days,
        campaign_start=campaign_start, n_good=c["n_good"], n_bad=c["n_bad"],
        trend_total=c["trend_total"], peak_lift=c["peak_lift"], ramp_days=common["ramp_days"],
        noise_scale=c["noise_scale"], base_level=c["base_level"],
        annual_amp=c.get("annual_amp", base.annual_amp),
        donor_annual_frac=c.get("donor_annual_frac", base.donor_annual_frac),
        donor_holiday_frac=c.get("donor_holiday_frac", base.donor_holiday_frac),
    )


def _regions_cfg(cfg) -> RegionsConfig:
    r = cfg["regions"]
    return RegionsConfig(
        seed=r["seed"], n_days=cfg["common"]["n_days"],
        campaign_start=cfg["common"]["campaign_start"], ramp_days=cfg["common"]["ramp_days"],
        trend_total=r["trend_total"], donor_noise=r["donor_noise"], base_idio=r["base_idio"],
        weekly_amp=r.get("weekly_amp", 0.06),
        mc_runs=r["mc_runs"], region_names=list(r["region_names"]), volumes=list(r["volumes"]),
        true_lifts=list(r["true_lifts"]), donor_counts=list(r["donor_counts"]),
        donor_quality=list(r["donor_quality"]), national_donors=r["national_donors"],
        national_quality=r["national_quality"],
    )


def build_all(config_path: Path | None = None):
    cfg = io_utils.load_config(config_path or ROOT / "configs" / "demo.yaml")
    growth = generate_scenario(_scenario_cfg(cfg, "growth", "growth"))
    decline = generate_scenario(_scenario_cfg(cfg, "decline", "decline"))
    annual_demo = generate_scenario(_scenario_cfg(cfg, "annual_demo", "annual_demo"))

    regions_cfg = _regions_cfg(cfg)
    campaign = generate_region_campaign(regions_cfg, regions_cfg.seed)
    scenarios = {"growth": growth, "decline": decline, "annual_demo": annual_demo}
    return cfg, scenarios, regions_cfg, campaign


def _write_scenario(scn, data_dir: Path):
    sub = data_dir / scn["name"]
    sub.mkdir(parents=True, exist_ok=True)
    scn["treated"].to_csv(sub / "treated_streams.csv")
    scn["candidates"].to_csv(sub / "candidate_donors.csv")
    scn["fanbase"].to_frame().to_csv(sub / "fanbase_other_region.csv")

    feats = pd.DataFrame(scn["candidate_features"], index=scn["candidates"].columns)
    feats.index.name = "candidate"
    feats.columns = [f"f{i}" for i in range(feats.shape[1])]
    feats["is_good"] = scn["candidate_is_good"]
    feats.to_csv(sub / "candidate_features.csv")

    truth = pd.DataFrame(
        {"true_counterfactual": scn["true_counterfactual"], "true_effect": scn["true_effect"]},
        index=scn["dates"],
    )
    truth.index.name = "date"
    truth.to_csv(sub / "ground_truth.csv")


def write_data(scenarios, campaign, data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    for scn in scenarios.values():
        _write_scenario(scn, data_dir)

    region_dir = data_dir / "regions"
    region_dir.mkdir(parents=True, exist_ok=True)
    for name, rp in campaign["per_region"].items():
        pd.DataFrame({"streams": rp["treated"]}, index=rp["dates"]).to_csv(
            region_dir / f"{name}_treated.csv"
        )
        rp["donors"].to_csv(region_dir / f"{name}_donors.csv")
    campaign["national_donors"].to_csv(region_dir / "national_donors.csv")
    io_utils.save_json(
        {
            "true_effects": campaign["true_effects"],
            "true_total": campaign["true_total"],
            "campaign_start": campaign["campaign_start"],
        },
        region_dir / "region_truth.json",
    )


def main():
    cfg, scenarios, regions_cfg, campaign = build_all()
    write_data(scenarios, campaign, ROOT / "data")
    print(
        f"Wrote scenarios {list(scenarios.keys())} and "
        f"{len(campaign['per_region'])} regional panels to {ROOT / 'data'}"
    )


if __name__ == "__main__":
    main()
