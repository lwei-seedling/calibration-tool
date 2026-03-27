"""Utility loaders for importing project data from external file formats."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from calibration.project.models import ProjectInputs


def load_price_series(
    file_path: str | Path,
) -> tuple[float, float, float]:
    """Load a price-series CSV and return (base_price, annual_drift, annual_vol).

    Expected columns (case-insensitive): Date, Price

    The observation frequency is inferred from average days between dates:
      - ≤ 35 days  → monthly  (annualise ×12 / ×√12)
      - ≤ 100 days → quarterly (annualise ×4  / ×√4)
      - else       → annual   (no scaling)

    Returns:
        base_price   : last observed price in the series
        annual_drift : annualised mean log-return (GBM μ)
        annual_vol   : annualised std  log-return (GBM σ)
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    price_col = next(
        (c for c in df.columns if c.startswith("price") or c == "close"),
        None,
    )
    if price_col is None:
        raise ValueError(
            f"Price series file {file_path.name!r} must have a 'price' column. "
            f"Found: {list(df.columns)}"
        )

    date_col = next((c for c in df.columns if "date" in c), None)
    prices = df[price_col].dropna().astype(float).to_numpy()
    if len(prices) < 2:
        raise ValueError(f"Price series {file_path.name!r} must contain at least 2 observations.")

    log_returns = np.diff(np.log(prices))
    period_mean = float(np.mean(log_returns))
    period_std  = float(np.std(log_returns, ddof=1))

    # Infer observation frequency
    periods_per_year = 1.0
    if date_col is not None:
        try:
            dates = pd.to_datetime(df[date_col].dropna())
            if len(dates) >= 2:
                avg_days = float((dates.max() - dates.min()).days / (len(dates) - 1))
                if avg_days <= 35:
                    periods_per_year = 12.0   # monthly
                elif avg_days <= 100:
                    periods_per_year = 4.0    # quarterly
                # else annual → 1.0
        except Exception:
            pass  # can't parse dates; leave as annual

    annual_drift = period_mean * periods_per_year
    annual_vol   = period_std  * float(np.sqrt(periods_per_year))
    base_price   = float(prices[-1])

    return base_price, annual_drift, annual_vol


def load_project_from_excel(
    file_path: str | Path,
    price_vol: float = 0.15,
) -> ProjectInputs:
    """Load a ProjectInputs from a CSV or Excel file.

    Supported column sets (case-insensitive, trimmed):

    **Format 1 — pre-computed net cashflow (simplest)**
        year, cashflow

    **Format 2 — revenue/cost split with embedded price params**
        year, yield, capex, opex [, project_lifetime, base_price,
        price_growth_rate, price_vol]

        Yield is physical units per year. base_price ($/unit) scales to revenue.
        GBM drift = price_growth_rate; GBM vol = price_vol column (overrides arg).
        Construction rows (yield == 0) become base_cashflows; operating rows
        become base_revenue + base_costs (GBM applied to revenue only).

    **Format 3 — revenue/cost split with external price series**
        year, yield, capex, opex [, project_lifetime], price_file

        price_file column names a CSV in the same folder (without .csv suffix,
        without "price_" prefix — e.g. "carbon_credits" → "price_carbon_credits.csv").
        base_price, annual_drift, and annual_vol are derived from that series.

    **Legacy format — revenue + cost columns (period-index years)**
        year (0-based), revenue, cost, capex
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path, engine="openpyxl")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Column aliases
    if "yield" in df.columns and "revenue" not in df.columns:
        df = df.rename(columns={"yield": "revenue"})
    if "opex" in df.columns and "cost" not in df.columns and "costs" not in df.columns:
        df = df.rename(columns={"opex": "cost"})

    # Sort by year if present
    if "year" in df.columns:
        df = df.sort_values("year").reset_index(drop=True)

    # ---- Validate Project_Lifetime if present ---- #
    if "project_lifetime" in df.columns:
        stated = int(df["project_lifetime"].dropna().iloc[0])
        if stated != len(df):
            warnings.warn(
                f"{file_path.name}: project_lifetime={stated} but file has {len(df)} rows. "
                "Using actual row count.",
                stacklevel=2,
            )

    # ---- Detect net cashflow column (Format 1) ---- #
    net_cf_col: str | None = None
    for candidate in ("cashflow", "net_cashflow", "net_cashflow"):
        if candidate in df.columns:
            net_cf_col = candidate
            break

    if net_cf_col is not None:
        cashflows = df[net_cf_col].fillna(0).tolist()
        if len(cashflows) < 2:
            raise ValueError(
                f"{file_path.name}: must have at least 2 rows. Got {len(cashflows)}."
            )
        return ProjectInputs(base_cashflows=cashflows, price_vol=price_vol)

    # ---- Shared column flags ---- #
    has_revenue   = "revenue" in df.columns
    has_cost      = "cost" in df.columns or "costs" in df.columns
    cost_col      = "cost" if "cost" in df.columns else "costs"
    has_capex_col = "capex" in df.columns
    _uses_calendar_years = (
        "year" in df.columns and int(df["year"].iloc[0]) >= 1900
    )

    # ---- Format 2 / 3 — Yield-based with base_price (calendar-year files) ---- #
    if has_revenue and _uses_calendar_years:
        # Resolve price parameters
        _p_vol: float = price_vol
        _p_drift: float | None = None

        has_price_file_col = "price_file" in df.columns
        has_base_price_col = "base_price" in df.columns

        if has_price_file_col:
            # Format 3: load external price series
            price_file_name = str(df["price_file"].dropna().iloc[0]).strip()
            price_csv = file_path.parent / f"price_{price_file_name}.csv"
            if not price_csv.exists():
                raise FileNotFoundError(
                    f"Price series file {price_csv} not found "
                    f"(referenced by {file_path.name} price_file={price_file_name!r})."
                )
            base_price, _p_drift, _p_vol = load_price_series(price_csv)
        elif has_base_price_col:
            # Format 2: embedded price params
            base_price = float(df["base_price"].dropna().iloc[0])
            if "price_growth_rate" in df.columns:
                _p_drift = float(df["price_growth_rate"].dropna().iloc[0])
            if "price_vol" in df.columns:
                _p_vol = float(df["price_vol"].dropna().iloc[0])
        else:
            base_price = 1.0  # Yield already in $; treat as pass-through

        # Split construction rows (revenue == 0) from operating rows (revenue > 0)
        revenue_raw = df["revenue"].fillna(0).to_numpy(dtype=float)
        capex_arr   = df["capex"].fillna(0).to_numpy(dtype=float) if has_capex_col else np.zeros(len(df))
        cost_arr    = df[cost_col].fillna(0).to_numpy(dtype=float) if has_cost else np.zeros(len(df))

        op_mask = revenue_raw > 0
        if not op_mask.any():
            raise ValueError(f"{file_path.name}: no operating rows found (all Yield == 0).")

        construction_rows = ~op_mask
        const_cfs = (-(capex_arr + cost_arr)[construction_rows]).tolist()
        if not const_cfs:
            # No construction rows — treat the entire t=0 row as initial outflow
            const_cfs = [-(capex_arr[0] + cost_arr[0])]
            op_mask = np.ones(len(df), dtype=bool)
            op_mask[0] = False

        base_revenue_vals = (revenue_raw[op_mask] * base_price).tolist()
        base_costs_vals   = cost_arr[op_mask].tolist()

        return ProjectInputs(
            base_cashflows=const_cfs,
            base_revenue=base_revenue_vals,
            base_costs=base_costs_vals,
            price_vol=_p_vol,
            price_drift=_p_drift,
        )

    # ---- Legacy: period-index revenue/cost split (Format B) ---- #
    if has_revenue and not _uses_calendar_years:
        if "year" in df.columns:
            op_rows = df[df["year"] > 0].reset_index(drop=True)
            t0_rows = df[df["year"] == 0]
        else:
            t0_rows = df.iloc[:1]
            op_rows = df.iloc[1:].reset_index(drop=True)

        if has_capex_col and not t0_rows.empty:
            cf0 = -float(t0_rows["capex"].fillna(0).iloc[0])
        elif has_capex_col:
            cf0 = -float(df["capex"].fillna(0).sum())
        else:
            cf0 = 0.0

        revenue = op_rows["revenue"].fillna(0).tolist()
        costs = op_rows[cost_col].fillna(0).tolist() if has_cost else [0.0] * len(revenue)

        if len(revenue) == 0:
            raise ValueError(f"{file_path.name}: no operating-period rows (year > 0).")

        return ProjectInputs(
            base_cashflows=[cf0],
            base_revenue=revenue,
            base_costs=costs,
            price_vol=price_vol,
        )

    # ---- Fallback: compute net CF from available columns ---- #
    revenue_arr2 = df.get("revenue", pd.Series(np.zeros(len(df)))).fillna(0).to_numpy(dtype=float)
    cost_arr2    = df.get(cost_col, pd.Series(np.zeros(len(df)))).fillna(0).to_numpy(dtype=float) if has_cost else np.zeros(len(df))
    capex_arr2   = df.get("capex", pd.Series(np.zeros(len(df)))).fillna(0).to_numpy(dtype=float) if has_capex_col else np.zeros(len(df))

    net_cf = (revenue_arr2 - cost_arr2 - capex_arr2).tolist()
    if len(net_cf) < 2:
        raise ValueError(f"{file_path.name}: must have at least 2 rows. Got {len(net_cf)}.")
    return ProjectInputs(base_cashflows=net_cf, price_vol=price_vol)
