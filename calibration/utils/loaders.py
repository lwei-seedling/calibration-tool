"""Utility loaders for importing project data from external file formats."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from calibration.project.models import ProjectInputs


def load_project_from_excel(
    file_path: str | Path,
    price_vol: float = 0.15,
) -> ProjectInputs:
    """Load a ProjectInputs from an Excel file.

    Expected columns (case-insensitive, trimmed):
        year         : integer period (0 = t=0 outflow, 1..T = operating periods)
        cashflow / net_cashflow / net cashflow : net cashflow per period (preferred)
        revenue      : per-period revenue (t=1..T); used for price-shocked mode
        cost / costs : per-period costs (t=1..T); unshocked
        capex        : capital expenditure per period (subtracted from net CF)

    Logic:
        1. If a net cashflow column is present: build base_cashflows directly.
        2. Else if revenue column is present: populate base_revenue + base_costs,
           and derive CF[0] from capex (year=0 row or capex column sum).
        3. Else: build net CF as revenue - cost - capex per period.

    The returned ProjectInputs uses cashflow mode with the provided price_vol.
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path, engine="openpyxl")
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Column aliases: accept natural field names used in project finance models
    if "yield" in df.columns and "revenue" not in df.columns:
        df = df.rename(columns={"yield": "revenue"})
    if "opex" in df.columns and "cost" not in df.columns and "costs" not in df.columns:
        df = df.rename(columns={"opex": "cost"})

    # Sort by year if present
    if "year" in df.columns:
        df = df.sort_values("year").reset_index(drop=True)

    # ---- Detect net cashflow column ---- #
    net_cf_col: str | None = None
    for candidate in ("cashflow", "net_cashflow", "net cashflow"):
        if candidate in df.columns:
            net_cf_col = candidate
            break

    if net_cf_col is not None:
        # Option A: net cashflow provided directly → base_cashflows (full lifecycle)
        cashflows = df[net_cf_col].fillna(0).tolist()
        if len(cashflows) < 2:
            raise ValueError(
                f"Excel file must contain at least 2 rows (t=0 and t=1). "
                f"Got {len(cashflows)} rows."
            )
        return ProjectInputs(base_cashflows=cashflows, price_vol=price_vol)

    # ---- Revenue/cost split ---- #
    has_revenue = "revenue" in df.columns
    has_cost = "cost" in df.columns or "costs" in df.columns
    cost_col = "cost" if "cost" in df.columns else "costs"
    has_capex_col = "capex" in df.columns

    # Calendar-year files (Year >= 1900) cannot use the t=0 split logic.
    # Route them to the net-CF fallback instead of Option B.
    _uses_calendar_years = (
        "year" in df.columns and int(df["year"].iloc[0]) >= 1900
    )

    if has_revenue and not _uses_calendar_years:
        # Option B: separate revenue and cost columns
        # Identify t=0 row vs operating rows
        if "year" in df.columns:
            op_rows = df[df["year"] > 0].reset_index(drop=True)
            t0_rows = df[df["year"] == 0]
        else:
            # Assume first row = t=0, rest = operating
            t0_rows = df.iloc[:1]
            op_rows = df.iloc[1:].reset_index(drop=True)

        # CF[0]: negative capex at t=0
        if has_capex_col and not t0_rows.empty:
            cf0 = -float(t0_rows["capex"].fillna(0).iloc[0])
        elif has_capex_col:
            cf0 = -float(df["capex"].fillna(0).sum())
        else:
            cf0 = 0.0

        revenue = op_rows["revenue"].fillna(0).tolist()
        costs = op_rows[cost_col].fillna(0).tolist() if has_cost else [0.0] * len(revenue)

        if len(revenue) == 0:
            raise ValueError("Excel file contains no operating-period rows (year > 0).")

        return ProjectInputs(
            base_cashflows=[cf0],
            base_revenue=revenue,
            base_costs=costs,
            price_vol=price_vol,
        )

    # ---- Fallback: compute net CF from available columns ---- #
    revenue_arr = df.get("revenue", pd.Series(np.zeros(len(df)))).fillna(0).to_numpy(dtype=float)
    cost_arr = df.get(cost_col, pd.Series(np.zeros(len(df)))).fillna(0).to_numpy(dtype=float) if has_cost else np.zeros(len(df))
    capex_arr = df.get("capex", pd.Series(np.zeros(len(df)))).fillna(0).to_numpy(dtype=float) if has_capex_col else np.zeros(len(df))

    net_cf = (revenue_arr - cost_arr - capex_arr).tolist()
    if len(net_cf) < 2:
        raise ValueError(
            f"Excel file must contain at least 2 rows. Got {len(net_cf)} rows."
        )
    return ProjectInputs(base_cashflows=net_cf, price_vol=price_vol)
