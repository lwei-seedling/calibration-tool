#!/usr/bin/env python3
"""End-to-end runner for the Catalytic Capital Calibration Tool.

Supports three input modes:
  1. Built-in sample data (default, no arguments needed)
  2. CSV files in a directory (--csv <dir>)
  3. JSON portfolio spec (--json <file>)

Usage:
    python run_e2e.py
    python run_e2e.py --csv examples/
    python run_e2e.py --json examples/portfolio.json
    python run_e2e.py --json examples/portfolio.json --sims 5000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Imports from the calibration package
# ---------------------------------------------------------------------------
from calibration.portfolio.models import PortfolioInputs, PortfolioResult
from calibration.portfolio.optimizer import PortfolioOptimizer
from calibration.project.models import ProjectInputs
from calibration.vehicle.calibration import CalibratorConfig
from calibration.vehicle.models import VehicleInputs


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _sep(char="─", width=70):
    print(char * width)


def _header(title: str):
    _sep("═")
    print(f"  {title}")
    _sep("═")


def _section(title: str):
    print()
    _sep()
    print(f"  {title}")
    _sep()


def print_results(result: PortfolioResult, inputs: PortfolioInputs, vehicle_names: list[str]):
    _header("CATALYTIC CAPITAL CALIBRATION — RESULTS")

    _section("Portfolio Summary")
    total_catalytic = sum(result.catalytic_allocations.values())
    total_commercial = sum(result.commercial_allocations.values())
    print(f"  Solver status          : {result.status}")
    print(f"  Total budget           : ${inputs.total_budget:>12,.0f}")
    print(f"  Total catalytic capital: ${total_catalytic:>12,.0f}  ({total_catalytic/inputs.total_budget:.1%})")
    print(f"  Total commercial capital: ${total_commercial:>11,.0f}  ({total_commercial/inputs.total_budget:.1%})")
    print(f"  Portfolio leverage ratio: {result.leverage_ratio:>10.2f}x  (commercial per catalytic $)")
    print(f"  Portfolio CVaR (95%)   : {result.cvar_95:>11.1%}  of deployed capital")

    irr_clean = result.portfolio_irr_distribution[np.isfinite(result.portfolio_irr_distribution)]
    if len(irr_clean) > 0:
        print(f"  Portfolio median IRR   : {float(np.median(irr_clean)):>11.1%}")
        print(f"  Portfolio IRR p10/p90  :  {float(np.percentile(irr_clean, 10)):.1%}  /  {float(np.percentile(irr_clean, 90)):.1%}")

    _section("Per-Vehicle Breakdown")
    rows = []
    for v_idx, name in enumerate(vehicle_names):
        alpha = result.catalytic_fractions[v_idx]
        alloc = result.allocations[v_idx]
        cat = result.catalytic_allocations[v_idx]
        com = result.commercial_allocations[v_idx]
        eff = result.marginal_catalytic_efficiency[v_idx]
        rows.append({
            "Vehicle": name,
            "Allocation $": f"{alloc:,.0f}",
            "Alpha (cat %)": f"{alpha:.1%}",
            "Catalytic $": f"{cat:,.0f}",
            "Commercial $": f"{com:,.0f}",
            "Leverage": f"{eff:.1f}x",
        })
    df = pd.DataFrame(rows).set_index("Vehicle")
    print(df.to_string())

    _section("Loss Distribution (portfolio loss rate)")
    loss = result.portfolio_loss_distribution
    print(f"  Mean loss rate  : {float(np.mean(loss)):.2%}")
    print(f"  Median loss rate: {float(np.median(loss)):.2%}")
    print(f"  p95 loss rate   : {float(np.percentile(loss, 95)):.2%}")
    print(f"  p99 loss rate   : {float(np.percentile(loss, 99)):.2%}")

    _sep("═")
    print()


# ---------------------------------------------------------------------------
# Sample data (built-in, used when no --csv / --json supplied)
# ---------------------------------------------------------------------------

def _built_in_inputs(n_sims: int, seed: int | None) -> PortfolioInputs:
    """Two-vehicle, three-project-each sample portfolio."""
    east_africa_projects = [
        ProjectInputs(capex=2_000_000, opex_annual=80_000, price=45.0, yield_=60_000,
                      lifetime_years=15, price_vol=0.12, yield_vol=0.08,
                      inflation_rate=0.04, fx_vol=0.06, delay_prob=0.03),
        ProjectInputs(capex=800_000,   opex_annual=35_000, price=12.0, yield_=80_000,
                      lifetime_years=10, price_vol=0.18, yield_vol=0.15,
                      inflation_rate=0.05, fx_vol=0.08, delay_prob=0.07),
        ProjectInputs(capex=1_200_000, opex_annual=50_000, price=25.0, yield_=55_000,
                      lifetime_years=12, price_vol=0.20, yield_vol=0.12,
                      inflation_rate=0.04, fx_vol=0.10, delay_prob=0.05),
    ]
    west_africa_projects = [
        ProjectInputs(capex=3_000_000, opex_annual=120_000, price=55.0, yield_=70_000,
                      lifetime_years=20, price_vol=0.10, yield_vol=0.07,
                      inflation_rate=0.03, fx_vol=0.07, delay_prob=0.02),
        ProjectInputs(capex=600_000,   opex_annual=28_000,  price=8.0,  yield_=100_000,
                      lifetime_years=8,  price_vol=0.25, yield_vol=0.20,
                      inflation_rate=0.05, fx_vol=0.09, delay_prob=0.10),
        ProjectInputs(capex=900_000,   opex_annual=40_000,  price=18.0, yield_=65_000,
                      lifetime_years=12, price_vol=0.22, yield_vol=0.14,
                      inflation_rate=0.04, fx_vol=0.11, delay_prob=0.06),
    ]
    vehicles = [
        VehicleInputs(
            projects=east_africa_projects,
            correlation_matrix=[[1.00, 0.35, 0.20],
                                 [0.35, 1.00, 0.40],
                                 [0.20, 0.40, 1.00]],
            total_capital=4_000_000,
            guarantee_coverage=0.30,
            grant_reserve=200_000,
            mezzanine_fraction=0.10,
            senior_coupon=0.08,
            mezzanine_coupon=0.13,
        ),
        VehicleInputs(
            projects=west_africa_projects,
            correlation_matrix=[[1.00, 0.25, 0.30],
                                 [0.25, 1.00, 0.35],
                                 [0.30, 0.35, 1.00]],
            total_capital=6_000_000,
            guarantee_coverage=0.25,
            grant_reserve=300_000,
            mezzanine_fraction=0.15,
            senior_coupon=0.075,
            mezzanine_coupon=0.12,
        ),
    ]
    return PortfolioInputs(
        vehicles=vehicles,
        calibrator_config=CalibratorConfig(
            investor_hurdle_irr=0.07,
            max_loss_probability=0.08,
        ),
        total_budget=10_000_000,
        max_allocation_fraction=0.70,
        cvar_confidence=0.95,
        cvar_max=0.35,
        n_sims=n_sims,
        seed=seed,
    )


_BUILT_IN_NAMES = ["East Africa Nature Fund", "West Africa Clean Energy Fund"]


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def _is_cashflow_format(df: pd.DataFrame) -> bool:
    """Return True if the CSV uses the cashflow format (year + cashflow columns).

    Option A — cashflow format:  columns are 'year' and 'cashflow'
    Option B — parametric format: columns include 'capex', 'price', etc.
    """
    cols = {c.strip().lower() for c in df.columns}
    return "year" in cols and "cashflow" in cols and "capex" not in cols


def _project_from_cashflow_csv(df: pd.DataFrame, price_vol: float = 0.15) -> ProjectInputs:
    """Build a ProjectInputs from a cashflow-format CSV.

    Expected columns:
        year     : integer period (0 = capex outflow, 1..T = operating CFs)
        cashflow : net cashflow for that period

    If year=0 is present with a negative value, it is treated as capex.
    All other rows (year >= 1) are treated as base_cashflows.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.sort_values("year").reset_index(drop=True)

    capex = 0.0
    if df["year"].min() == 0:
        t0_val = float(df.loc[df["year"] == 0, "cashflow"].iloc[0])
        if t0_val < 0:
            capex = abs(t0_val)
        df = df[df["year"] > 0]

    base_cashflows = df["cashflow"].tolist()
    if not base_cashflows:
        raise ValueError("Cashflow CSV has no operating-period rows (year >= 1).")

    return ProjectInputs(
        capex=capex if capex > 0 else None,
        base_cashflows=base_cashflows,
        price_vol=price_vol,
    )


def _load_csv_inputs(csv_dir: Path, n_sims: int, seed: int | None) -> tuple[PortfolioInputs, list[str]]:
    """Load all projects_vehicle_N.csv files from a directory.

    Auto-detects the CSV format per file:

    **Option A — cashflow format** (columns: year, cashflow):
        Each row is a single period. Year 0 (if present and negative) is the
        capex outflow. Years 1..T are base operating cashflows for one project.
        A single cashflow-format file per vehicle defines ONE project.

    **Option B — parametric format** (columns: capex, opex_annual, price, yield_, ...):
        Each row is one project. Multiple rows = multiple projects in one vehicle.

    Vehicle-level settings (correlation_matrix, total_capital, etc.) are inferred
    and set to conservative defaults. For full control, use --json.

    Returns:
        (PortfolioInputs, list of vehicle names)
    """
    csv_files = sorted(csv_dir.glob("projects_vehicle_*.csv"))
    if not csv_files:
        print(f"ERROR: No 'projects_vehicle_N.csv' files found in {csv_dir}")
        sys.exit(1)

    vehicles = []
    names = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        if _is_cashflow_format(df):
            # ----- Option A: cashflow format — one project per file -----
            print(f"  {csv_path.name}: detected cashflow format (year, cashflow)")
            try:
                project = _project_from_cashflow_csv(df)
            except ValueError as exc:
                print(f"ERROR parsing {csv_path.name}: {exc}")
                sys.exit(1)
            projects = [project]
        else:
            # ----- Option B: parametric format — one project per row -----
            required = {"capex", "opex_annual", "price", "yield_", "lifetime_years"}
            missing = required - set(df.columns)
            if missing:
                print(f"ERROR: {csv_path.name} is missing columns: {missing}")
                sys.exit(1)

            projects = []
            for _, row in df.iterrows():
                projects.append(ProjectInputs(
                    capex=float(row["capex"]),
                    opex_annual=float(row["opex_annual"]),
                    price=float(row["price"]),
                    yield_=float(row["yield_"]),
                    lifetime_years=int(row["lifetime_years"]),
                    price_vol=float(row.get("price_vol", 0.15)),
                    yield_vol=float(row.get("yield_vol", 0.10)),
                    inflation_rate=float(row.get("inflation_rate", 0.03)),
                    fx_vol=float(row.get("fx_vol", 0.05)),
                    delay_prob=float(row.get("delay_prob", 0.05)),
                ))

        J = len(projects)
        # Default correlation: moderate positive (0.30 off-diagonal)
        corr = [[1.0 if i == j else 0.30 for j in range(J)] for i in range(J)]
        # For cashflow-mode projects capex may be 0; estimate total capital from CFs
        capex_sum = sum(
            (p.capex or abs(min(p.base_cashflows or [0]))) for p in projects
        )
        total_capital = max(capex_sum, 1.0) * 1.2  # 20% buffer

        vehicles.append(VehicleInputs(
            projects=projects,
            correlation_matrix=corr,
            total_capital=total_capital,
            guarantee_coverage=0.25,
            grant_reserve=total_capital * 0.05,
            mezzanine_fraction=0.10,
            senior_coupon=0.08,
            mezzanine_coupon=0.12,
        ))
        vname = csv_path.stem.replace("projects_", "").replace("_", " ").title()
        names.append(vname)

    total_budget = sum(v.total_capital for v in vehicles)
    inputs = PortfolioInputs(
        vehicles=vehicles,
        calibrator_config=CalibratorConfig(investor_hurdle_irr=0.07, max_loss_probability=0.08),
        total_budget=total_budget,
        max_allocation_fraction=0.70,
        cvar_confidence=0.95,
        cvar_max=0.35,
        n_sims=n_sims,
        seed=seed,
    )
    return inputs, names


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------

def _load_json_inputs(json_path: Path, n_sims_override: int | None, seed_override: int | None) -> tuple[PortfolioInputs, list[str]]:
    """Load a full portfolio spec from a JSON file.

    The JSON schema mirrors PortfolioInputs. Vehicle names are read from an
    optional top-level 'vehicle_names' list or from each vehicle's 'name' field.
    """
    with open(json_path) as f:
        raw = json.load(f)

    # Extract vehicle names before Pydantic parsing (not part of VehicleInputs)
    names = raw.pop("vehicle_names", None)
    for i, v in enumerate(raw.get("vehicles", [])):
        n = v.pop("name", None)
        if names is None:
            names = []
        if n and i >= len(names):
            names.append(n)

    if n_sims_override is not None:
        raw["n_sims"] = n_sims_override
    if seed_override is not None:
        raw["seed"] = seed_override

    inputs = PortfolioInputs.model_validate(raw)
    if not names or len(names) < len(inputs.vehicles):
        names = [f"Vehicle {i+1}" for i in range(len(inputs.vehicles))]
    return inputs, names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Catalytic Capital Calibration Tool — end-to-end runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--csv", metavar="DIR", type=Path,
                       help="Directory containing projects_vehicle_N.csv files.")
    group.add_argument("--json", metavar="FILE", type=Path,
                       help="JSON file with full portfolio specification.")
    parser.add_argument("--sims", type=int, default=None,
                        help="Override number of Monte Carlo simulations (default: 1000).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    n_sims = args.sims or 1000
    seed = args.seed

    # ---- Load inputs -------------------------------------------------------
    if args.csv:
        if not args.csv.is_dir():
            print(f"ERROR: {args.csv} is not a directory.")
            sys.exit(1)
        print(f"Loading projects from CSV files in: {args.csv}")
        inputs, vehicle_names = _load_csv_inputs(args.csv, n_sims, seed)

    elif args.json:
        if not args.json.exists():
            print(f"ERROR: {args.json} not found.")
            sys.exit(1)
        print(f"Loading portfolio from JSON: {args.json}")
        inputs, vehicle_names = _load_json_inputs(args.json, args.sims, seed)

    else:
        print("Using built-in sample data (East Africa + West Africa funds).")
        print("Run with --csv or --json to load your own data.")
        inputs = _built_in_inputs(n_sims, seed)
        vehicle_names = _BUILT_IN_NAMES

    # ---- Summary of what we're running ------------------------------------
    print()
    _header("INPUT SUMMARY")
    print(f"  Vehicles       : {len(inputs.vehicles)}")
    for i, (v, name) in enumerate(zip(inputs.vehicles, vehicle_names)):
        print(f"    [{i}] {name}  |  {len(v.projects)} projects  |  "
              f"capital=${v.total_capital:,.0f}  |  "
              f"guarantee={v.guarantee_coverage:.0%}  |  reserve=${v.grant_reserve:,.0f}")
    print(f"  Total budget   : ${inputs.total_budget:,.0f}")
    print(f"  Simulations    : {inputs.n_sims:,}")
    print(f"  Seed           : {inputs.seed}")
    print(f"  Hurdle IRR     : {inputs.calibrator_config.investor_hurdle_irr:.1%}")
    print(f"  Max loss prob  : {inputs.calibrator_config.max_loss_probability:.1%}")
    print(f"  CVaR limit     : {inputs.cvar_max:.1%} @ {inputs.cvar_confidence:.0%}")
    print()

    # ---- Run ---------------------------------------------------------------
    print("Running calibration and portfolio optimisation...")
    t0 = time.time()
    optimizer = PortfolioOptimizer(inputs)
    result = optimizer.run()
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s\n")

    # ---- Print results -----------------------------------------------------
    print_results(result, inputs, vehicle_names)


if __name__ == "__main__":
    main()
