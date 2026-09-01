"""Regenerate the UI sample CSVs from documented unit economics.

Every figure traces to a public source listed in docs/SAMPLE_DATA_SOURCES.md.
Run: python scripts/build_sample_data.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "examples" / "ui_sample"

# --- Carbon prices, $/tCO2e (see docs/SAMPLE_DATA_SOURCES.md) ------------------
# A fund financing new, rated projects prices off the rated tier, not the
# all-vintage market average ($6.53/t in 2024).
PRICE_ARR      = 26.0    # ARR rated BBB or higher, 2025 offtakes (unrated ~$14)
PRICE_AGRO     = 20.0    # agroforestry, below rated ARR
PRICE_REDD     = 6.0     # REDD+ average, 2026
PRICE_BIOCHAR  = 130.0   # biochar CDR, low end of the $130-200 range

# --- Spot price volatility, annual --------------------------------------------
# Market-wide VCM average price ran $4.04 (2021) -> $7.37 (2022) -> $6.97 (2023)
# -> $6.53 (2024); the sample sd of those log returns is 0.38. REDD+ alone lost
# 62% of its value in a single year, and N-GEO futures fell from double digits to
# well under a dollar. Biochar prices have been notably stable by comparison.
SPOT_VOL = {
    "arr_core":  0.35,
    "arr_low":   0.28,
    "arr_high":  0.45,
    "agro":      0.40,
    "agro_high": 0.50,
    "redd":      0.55,
    "biochar":   0.20,
}

# --- Offtake coverage ----------------------------------------------------------
# `price_vol` is the volatility of project REVENUE, not of the spot carbon price.
# Project-financed carbon projects forward-sell a large share of production —
# $12.25bn of offtake deals were signed in 2025, up from ~$4bn in 2024 — and
# lenders generally require it. With a fraction f of volume sold forward at a
# fixed price, revenue volatility is (1 - f) x spot volatility.
#
# This is the single most consequential assumption in the sample data, and it is
# set conservatively: only 30% of nature-based volume forward-sold, well below
# what a project-finance lender would normally require, reflecting how immature
# forward contracting still is for new ARR and agroforestry supply. Biochar is
# largely pre-sold; REDD+ is the hardest to forward-sell.
#
# Raise these and alpha falls sharply - it is worth re-running the model with
# your own coverage assumptions before reading anything into the numbers.
OFFTAKE = {
    "arr_core": 0.30, "arr_low": 0.30, "arr_high": 0.30,
    "agro": 0.30, "agro_high": 0.30,
    "redd": 0.25, "biochar": 0.70,
}


def vol(key: str) -> float:
    """Revenue volatility = (1 - offtake coverage) x spot price volatility."""
    return round(SPOT_VOL[key] * (1.0 - OFFTAKE[key]), 3)


VOL_ARR_CORE   = vol("arr_core")
VOL_ARR_LOW    = vol("arr_low")
VOL_ARR_HIGH   = vol("arr_high")
VOL_AGRO       = vol("agro")
VOL_AGRO_HIGH  = vol("agro_high")
VOL_REDD       = vol("redd")
VOL_BIOCHAR    = vol("biochar")

DRIFT = 0.03      # modest real price appreciation; prices were flat 2022-2024
DRIFT_LOW = 0.02

# Fund hold, in operating years. Projects carry a 20-year VCS crediting period;
# a fund vehicle exits well before the tail rather than holding to year 20.
OPERATING_YEARS = 15


def ramp(n_years: int, plateau_at: int = 8) -> list[float]:
    """Sequestration ramp for a planted project: young trees fix less carbon."""
    out = []
    for t in range(1, n_years + 1):
        out.append(round(min(1.0, 0.15 + 0.85 * (t - 1) / max(1, plateau_at - 1)), 4))
    return out


def planted_project(hectares, capex_per_ha, capex_split, opex_per_ha,
                    tco2_per_ha, price, vol, drift=DRIFT, operating_years=OPERATING_YEARS,
                    revenue_type="carbon", start_year=2025):
    """Rows for a planting project: construction years, then a ramping yield."""
    rows, year = [], start_year
    total_capex = hectares * capex_per_ha
    for share in capex_split:
        rows.append({"year": year, "yield": 0, "capex": round(total_capex * share),
                     "opex": round(hectares * opex_per_ha * 0.35),
                     "revenue_type": revenue_type, "base_price": price,
                     "price_growth_rate": drift, "price_vol": vol})
        year += 1
    for frac in ramp(operating_years):
        rows.append({"year": year, "yield": round(hectares * tco2_per_ha * frac),
                     "capex": 0, "opex": round(hectares * opex_per_ha),
                     "revenue_type": revenue_type, "base_price": price,
                     "price_growth_rate": drift, "price_vol": vol})
        year += 1
    return pd.DataFrame(rows)


def flat_project(capex_schedule, annual_opex, annual_tco2, price, vol,
                 drift=DRIFT, operating_years=OPERATING_YEARS, revenue_type="carbon",
                 start_year=2025, ramp_first_year=1.0):
    """Rows for a project with a flat output profile (REDD+, biochar)."""
    rows, year = [], start_year
    for amount in capex_schedule:
        rows.append({"year": year, "yield": 0, "capex": round(amount),
                     "opex": round(annual_opex * 0.3), "revenue_type": revenue_type,
                     "base_price": price, "price_growth_rate": drift, "price_vol": vol})
        year += 1
    for t in range(operating_years):
        frac = ramp_first_year if t == 0 else 1.0
        rows.append({"year": year, "yield": round(annual_tco2 * frac), "capex": 0,
                     "opex": round(annual_opex), "revenue_type": revenue_type,
                     "base_price": price, "price_growth_rate": drift, "price_vol": vol})
        year += 1
    return pd.DataFrame(rows)


SPECS = {
    # ---- Vehicle 1: East Africa reforestation (ARR) ----
    "vehicle_1_forestry/project_forestry_arr.csv": planted_project(
        hectares=1600, capex_per_ha=1800, capex_split=(0.60, 0.40),
        opex_per_ha=60, tco2_per_ha=14, price=PRICE_ARR, vol=VOL_ARR_CORE),
    "vehicle_1_forestry/project_forestry_conservative.csv": planted_project(
        hectares=900, capex_per_ha=1500, capex_split=(0.65, 0.35),
        opex_per_ha=50, tco2_per_ha=12, price=PRICE_ARR, vol=VOL_ARR_LOW),
    "vehicle_1_forestry/project_forestry_risky.csv": planted_project(
        hectares=1200, capex_per_ha=2300, capex_split=(0.45, 0.35, 0.20),
        opex_per_ha=70, tco2_per_ha=13, price=PRICE_ARR, vol=VOL_ARR_HIGH,
        operating_years=OPERATING_YEARS - 1),

    # ---- Vehicle 2: West Africa agroforestry ----
    "vehicle_2_agroforestry/project_agro_standard.csv": planted_project(
        hectares=3000, capex_per_ha=600, capex_split=(0.55, 0.45),
        opex_per_ha=40, tco2_per_ha=7, price=PRICE_AGRO, vol=VOL_AGRO),
    "vehicle_2_agroforestry/project_agro_conservative.csv": planted_project(
        hectares=2500, capex_per_ha=500, capex_split=(0.60, 0.40),
        opex_per_ha=28, tco2_per_ha=5, price=18.0, vol=0.32),
    "vehicle_2_agroforestry/project_agro_volatile.csv": planted_project(
        hectares=1800, capex_per_ha=850, capex_split=(0.55, 0.45),
        opex_per_ha=45, tco2_per_ha=9, price=PRICE_AGRO, vol=VOL_AGRO_HIGH),

    # ---- Vehicle 3: Mixed (REDD+, biochar, hybrid) ----
    "vehicle_3_mixed/project_mixed_redd.csv": flat_project(
        capex_schedule=(450_000, 250_000), annual_opex=150_000,
        annual_tco2=39_000, price=PRICE_REDD, vol=VOL_REDD, drift=DRIFT_LOW,
        ramp_first_year=0.7),
    "vehicle_3_mixed/project_mixed_biochar.csv": flat_project(
        capex_schedule=(1_600_000, 1_000_000), annual_opex=806_000,
        annual_tco2=10_752, price=PRICE_BIOCHAR, vol=VOL_BIOCHAR, drift=DRIFT_LOW,
        operating_years=10, ramp_first_year=0.6),
    "vehicle_3_mixed/project_mixed_hybrid.csv": planted_project(
        hectares=1400, capex_per_ha=1300, capex_split=(0.60, 0.40),
        opex_per_ha=50, tco2_per_ha=11, price=21.0, vol=0.38),
}


if __name__ == "__main__":
    print(f"{'file':<38} {'capex':>10} {'lifetime rev':>13} {'net/capex':>10}")
    for rel, df in SPECS.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        capex = df["capex"].sum()
        rev = (df["yield"] * df["base_price"]).sum()
        net = (rev - df["opex"].sum() - capex) / capex
        print(f"{rel.split('/')[-1]:<38} {capex:>10,.0f} {rev:>13,.0f} {net:>10.2f}")
