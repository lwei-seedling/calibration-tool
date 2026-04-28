"""Streamlit demo — Catalytic Capital Optimizer."""
from __future__ import annotations

import io
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calibration.portfolio.models import PortfolioInputs
from calibration.portfolio.optimizer import PortfolioOptimizer
from calibration.project.models import ProjectInputs
from calibration.utils.loaders import load_project_from_excel
from calibration.vehicle.calibration import CalibratorConfig
from calibration.vehicle.models import VehicleInputs

from auth import check_auth, logout

try:
    import openai as _openai_mod  # noqa: F401
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_DIR = Path(__file__).parent / "examples" / "ui_sample"
REQUIRED_COLS = {"year", "yield", "capex", "opex", "base_price", "price_growth_rate", "price_vol"}
MAX_VEHICLES = 3
MAX_PROJECTS_PER_VEHICLE = 5
MAX_ROWS = 30

TEMPLATE_CARBON = """\
year,yield,capex,opex,revenue_type,base_price,price_growth_rate,price_vol
2025,0,3000000,80000,carbon,15.0,0.05,0.30
2026,0,1500000,80000,carbon,15.0,0.05,0.30
2027,20000,0,120000,carbon,15.0,0.05,0.30
2028,35000,0,120000,carbon,15.0,0.05,0.30
2029,50000,0,120000,carbon,15.0,0.05,0.30
2030,50000,0,120000,carbon,15.0,0.05,0.30
2031,50000,0,120000,carbon,15.0,0.05,0.30
2032,50000,0,120000,carbon,15.0,0.05,0.30
2033,50000,0,120000,carbon,15.0,0.05,0.30
2034,50000,0,120000,carbon,15.0,0.05,0.30
2035,50000,0,120000,carbon,15.0,0.05,0.30
2036,50000,0,120000,carbon,15.0,0.05,0.30
2037,50000,0,120000,carbon,15.0,0.05,0.30
2038,50000,0,120000,carbon,15.0,0.05,0.30
2039,50000,0,120000,carbon,15.0,0.05,0.30
2040,50000,0,120000,carbon,15.0,0.05,0.30
"""

TEMPLATE_COMMODITY = """\
year,yield,capex,opex,revenue_type,base_price,price_growth_rate,price_vol
2025,0,4000000,100000,commodity,1800.0,0.03,0.22
2026,0,2000000,100000,commodity,1800.0,0.03,0.22
2027,0,500000,150000,commodity,1800.0,0.03,0.22
2028,5000,0,200000,commodity,1800.0,0.03,0.22
2029,10000,0,200000,commodity,1800.0,0.03,0.22
2030,14000,0,200000,commodity,1800.0,0.03,0.22
2031,17000,0,200000,commodity,1800.0,0.03,0.22
2032,19000,0,200000,commodity,1800.0,0.03,0.22
2033,20000,0,200000,commodity,1800.0,0.03,0.22
2034,20000,0,200000,commodity,1800.0,0.03,0.22
2035,20000,0,200000,commodity,1800.0,0.03,0.22
2036,20000,0,200000,commodity,1800.0,0.03,0.22
2037,20000,0,200000,commodity,1800.0,0.03,0.22
2038,20000,0,200000,commodity,1800.0,0.03,0.22
2039,20000,0,200000,commodity,1800.0,0.03,0.22
2040,20000,0,200000,commodity,1800.0,0.03,0.22
"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_df(df: pd.DataFrame, filename: str) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for an uploaded Format-2 dataframe."""
    errors: list[str] = []
    warns: list[str] = []
    cols = {str(c).strip().lower() for c in df.columns}
    missing = REQUIRED_COLS - cols
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}. Download a template to see the required format.")
        return errors, warns

    if len(df) > MAX_ROWS:
        errors.append(f"Max {MAX_ROWS}-year horizon. File has {len(df)} rows.")

    df2 = df.copy()
    df2.columns = [str(c).strip().lower() for c in df2.columns]

    num_cols = ["yield", "capex", "opex", "base_price", "price_growth_rate", "price_vol"]
    for col in num_cols:
        bad = pd.to_numeric(df2[col], errors="coerce").isna()
        if bad.any():
            rows = (df2.index[bad] + 2).tolist()
            errors.append(f"Non-numeric values in '{col}' at row(s): {rows}.")

    if errors:
        return errors, warns

    df2[num_cols] = df2[num_cols].apply(pd.to_numeric, errors="coerce")

    if (df2["yield"] <= 0).all():
        errors.append("No operating rows found (all yield=0). At least one year must have yield>0.")
    if (df2["base_price"] <= 0).any():
        errors.append("base_price must be positive ($/unit).")

    if errors:
        return errors, warns

    if df2["capex"].iloc[0] == 0 and df2["yield"].iloc[0] == 0:
        warns.append("First year has no capex — unusual for a construction-phase project.")
    rev = (df2["yield"] * df2["base_price"]).sum()
    costs = (df2["capex"] + df2["opex"]).sum()
    if rev < costs:
        warns.append("At base assumptions total revenues < total costs. May require significant support.")
    return errors, warns



# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _capex_sum(projects: list[ProjectInputs]) -> float:
    total = 0.0
    for p in projects:
        if p.base_cashflows is not None:
            total += abs(sum(cf for cf in p.base_cashflows if cf < 0))
        elif p.capex:
            total += p.capex
    return total


def _build_vehicle(
    projects: list[ProjectInputs],
    guarantee_coverage: float,
    grant_reserve_pct: float,
    mezzanine_fraction: float,
    senior_coupon: float,
    off_diag_corr: float,
) -> VehicleInputs:
    J = len(projects)
    corr = [[1.0 if i == j else off_diag_corr for j in range(J)] for i in range(J)]
    total_capital = max(_capex_sum(projects), 1.0) * 1.2
    return VehicleInputs(
        projects=projects,
        correlation_matrix=corr,
        total_capital=total_capital,
        guarantee_coverage=guarantee_coverage,
        grant_reserve=total_capital * grant_reserve_pct,
        mezzanine_fraction=mezzanine_fraction,
        senior_coupon=senior_coupon,
        mezzanine_coupon=min(senior_coupon + 0.04, 0.20),
    )


def _build_portfolio(
    vehicles: list[VehicleInputs],
    n_sims: int,
    hurdle_irr: float,
    max_loss_prob: float,
    cvar_max: float,
    seed: int,
    catalytic_budget: float | None,
) -> PortfolioInputs:
    total_budget = sum(v.total_capital for v in vehicles)
    return PortfolioInputs(
        vehicles=vehicles,
        calibrator_config=CalibratorConfig(
            investor_hurdle_irr=hurdle_irr,
            max_loss_probability=max_loss_prob,
        ),
        total_budget=total_budget,
        catalytic_budget=catalytic_budget if catalytic_budget and catalytic_budget > 0 else None,
        min_deployment=total_budget * 0.3,
        max_allocation_fraction=0.70,
        cvar_confidence=0.95,
        cvar_max=cvar_max,
        n_sims=n_sims,
        seed=seed,
    )



# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

def _rebuild_sensitivity(
    base_inputs: PortfolioInputs,
    test_id: str,
    value: float,
) -> PortfolioInputs:
    """Return a modified PortfolioInputs for sensitivity test A/B/C/D."""
    new_vehicles: list[VehicleInputs] = []
    for v in base_inputs.vehicles:
        if test_id == "A":
            new_vehicles.append(v.model_copy(update={"guarantee_coverage": value}))
        elif test_id == "B":
            new_projects = [
                p.model_copy(update={"price_vol": min(p.price_vol * value, 1.0)})
                for p in v.projects
            ]
            new_vehicles.append(v.model_copy(update={"projects": new_projects}))
        else:  # C or D — scale base_revenue; fall back to scaling positive base_cashflows
            new_projects = []
            for p in v.projects:
                if p.base_revenue is not None:
                    new_projects.append(
                        p.model_copy(update={"base_revenue": [r * value for r in p.base_revenue]})
                    )
                elif p.base_cashflows is not None:
                    scaled = [
                        cf * value if (i > 0 and cf > 0) else cf
                        for i, cf in enumerate(p.base_cashflows)
                    ]
                    new_projects.append(p.model_copy(update={"base_cashflows": scaled}))
                else:
                    new_projects.append(p)
            new_vehicles.append(v.model_copy(update={"projects": new_projects}))
    return base_inputs.model_copy(update={"vehicles": new_vehicles})



# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_irr_histogram(irr_dist: np.ndarray) -> go.Figure:
    clean = irr_dist[np.isfinite(irr_dist)] * 100
    nan_pct = (len(irr_dist) - len(clean)) / max(len(irr_dist), 1) * 100
    median = float(np.median(clean)) if len(clean) else 0.0
    fig = go.Figure(go.Histogram(x=clean, nbinsx=50, marker_color="#2E86AB", opacity=0.85))
    fig.add_vline(x=median, line_dash="dash", line_color="#E84855",
                  annotation_text=f"Median {median:.1f}%", annotation_position="top right")
    annotations = []
    if nan_pct > 0.5:
        annotations.append(dict(text=f"Loss/NaN: {nan_pct:.1f}% of paths",
                                 x=0.98, y=0.95, xref="paper", yref="paper",
                                 showarrow=False, font=dict(size=11, color="gray"),
                                 xanchor="right"))
    fig.update_layout(title="Portfolio IRR Distribution", xaxis_title="IRR (%)",
                      yaxis_title="Scenarios", showlegend=False,
                      height=320, margin=dict(l=40, r=20, t=40, b=40),
                      annotations=annotations)
    return fig


def chart_capital_stack(result, names: list[str]) -> go.Figure:
    idxs = sorted(result.allocations.keys())
    order = sorted(idxs, key=lambda i: result.catalytic_fractions.get(i, 0), reverse=True)
    vnames = [names[i] for i in order]
    cat = [result.catalytic_allocations.get(i, 0) / 1e6 for i in order]
    com = [result.commercial_allocations.get(i, 0) / 1e6 for i in order]
    fig = go.Figure([
        go.Bar(name="Catalytic", y=vnames, x=cat, orientation="h",
               marker_color="#1565C0", text=[f"${v:.1f}M" for v in cat], textposition="inside"),
        go.Bar(name="Commercial", y=vnames, x=com, orientation="h",
               marker_color="#2E7D32", text=[f"${v:.1f}M" for v in com], textposition="inside"),
    ])
    fig.update_layout(barmode="stack", title="Capital Stack by Vehicle",
                      xaxis_title="Capital ($M)", height=360,
                      margin=dict(l=20, r=20, t=40, b=70),
                      legend=dict(orientation="h", yanchor="top", y=-0.20,
                                  xanchor="center", x=0.5))
    return fig


def chart_alpha(result, names: list[str]) -> go.Figure:
    idxs = sorted(result.allocations.keys())
    alphas = [result.catalytic_fractions.get(i, 0) * 100 for i in idxs]
    vnames = [names[i] for i in idxs]
    colors = ["#2E7D32" if a < 40 else "#F9A825" if a < 60 else "#C62828" for a in alphas]
    fig = go.Figure(go.Bar(x=vnames, y=alphas, marker_color=colors,
                            text=[f"{a:.1f}%" for a in alphas], textposition="outside"))
    fig.update_layout(title="Alpha (Catalytic Fraction) by Vehicle",
                      yaxis_title="Alpha (%)",
                      yaxis=dict(range=[0, max(alphas + [10]) * 1.35]),
                      height=300, margin=dict(l=40, r=20, t=40, b=50),
                      annotations=[dict(text="Lower alpha = more efficient use of catalytic capital",
                                        x=0.5, y=-0.20, xref="paper", yref="paper",
                                        showarrow=False, font=dict(size=11, color="gray"))])
    return fig


def chart_cashflows(inputs, names: list[str]) -> list[go.Figure]:
    """One Plotly figure per vehicle showing base-case annual cashflow breakdown.

    Bars show capital costs (construction), operating costs, and revenue.
    A line overlay shows the net cashflow per period.
    x-axis uses relative period labels (Yr 0, Yr 1, …) because calendar years
    are not preserved through the ProjectInputs data model.
    """
    figs = []
    for vehicle, vname in zip(inputs.vehicles, names):
        max_periods = 0
        for p in vehicle.projects:
            if p.base_cashflows and p.base_revenue:
                max_periods = max(max_periods, len(p.base_cashflows) + len(p.base_revenue))
            elif p.base_cashflows:
                max_periods = max(max_periods, len(p.base_cashflows))
            elif p.lifetime_years:
                max_periods = max(max_periods, p.lifetime_years + 1)
        if max_periods == 0:
            continue

        labels = [f"Yr {i}" for i in range(max_periods)]
        agg_capex = np.zeros(max_periods)
        agg_opex  = np.zeros(max_periods)
        agg_rev   = np.zeros(max_periods)

        for p in vehicle.projects:
            if p.base_cashflows:
                n_const = len(p.base_cashflows)
                for t, cf in enumerate(p.base_cashflows):
                    if t < max_periods and cf < 0:
                        agg_capex[t] += cf   # already negative
                for t, rev in enumerate(p.base_revenue or []):
                    idx = n_const + t
                    if idx < max_periods:
                        agg_rev[idx] += rev
                for t, cost in enumerate(p.base_costs or []):
                    idx = n_const + t
                    if idx < max_periods:
                        agg_opex[idx] -= cost  # flip to negative

        net = agg_capex + agg_opex + agg_rev
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Capital costs", x=labels, y=agg_capex.tolist(),
                             marker_color="#B71C1C"))
        fig.add_trace(go.Bar(name="Operating costs", x=labels, y=agg_opex.tolist(),
                             marker_color="#EF9A9A"))
        fig.add_trace(go.Bar(name="Revenue (base)", x=labels, y=agg_rev.tolist(),
                             marker_color="#2E7D32"))
        fig.add_trace(go.Scatter(name="Net CF", x=labels, y=net.tolist(),
                                 mode="lines+markers",
                                 line=dict(color="#1565C0", width=2),
                                 marker=dict(size=5)))
        fig.update_layout(
            title=f"{vname} — Base-Case Cashflows (pre-Monte Carlo)",
            barmode="relative",
            xaxis_title="Period",
            yaxis_title="Cashflow (USD)",
            height=400,
            margin=dict(l=50, r=20, t=45, b=80),
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        )
        figs.append((vname, fig))
    return figs


# ---------------------------------------------------------------------------
# Results commentary helpers
# ---------------------------------------------------------------------------

def _effective_horizon(inputs) -> int:
    """Max project lifetime (years) across all vehicles."""
    max_lt = 0
    for v in inputs.vehicles:
        for p in v.projects:
            lt = p.lifetime_years
            if lt is None:
                if p.base_cashflows and p.base_revenue:
                    lt = len(p.base_cashflows) + len(p.base_revenue) - 1
                elif p.base_cashflows:
                    lt = len(p.base_cashflows) - 1
                else:
                    lt = 10
            max_lt = max(max_lt, lt)
    return max_lt


def _vehicle_commentary(name: str, alpha: float, leverage: float, hurdle: float,
                         median_irr: float, cvar: float, n_sims: int) -> str:
    conc_pct = f"{alpha:.0%}"
    lev_str  = f"{leverage:.1f}\u00d7"

    if alpha < 0.20:
        eff_note = ("This is a capital-efficient structure — most of the return comes from "
                    "commercial sources with limited concessional support.")
    elif alpha < 0.40:
        eff_note = "A reasonable catalytic requirement for this asset class and risk profile."
    else:
        eff_note = ("This vehicle has a high concessional requirement. Consider increasing "
                    "guarantee coverage or grant reserves to reduce the catalytic fraction.")

    if np.isfinite(median_irr):
        verdict = "meets" if median_irr >= hurdle else "falls short of"
        irr_line = (f"The modelled median senior IRR is **{median_irr:.1%}**, which {verdict} "
                    f"the **{hurdle:.1%}** commercial hurdle rate.")
    else:
        irr_line = ("Insufficient profitable simulation paths to report a reliable median IRR — "
                    "review the project revenue and cost assumptions.")

    return (
        f"**{name}** requires **{conc_pct}** of its capital to be concessional "
        f"(first-loss, grants, or guarantees), attracting **{lev_str} of commercial capital** "
        f"per $1 of catalytic funding. "
        f"{irr_line} "
        f"Tail-loss risk (CVaR\u2085) is **{cvar:.1%}** of vehicle value in the worst 5% of "
        f"{n_sims:,} simulated scenarios. {eff_note}"
    )


def _portfolio_commentary(result, inputs, names: list[str]) -> str:
    total_cat = sum(result.catalytic_allocations.values())
    total_com = sum(result.commercial_allocations.values())
    n_active  = sum(1 for a in result.allocations.values() if a > 0)
    lev       = result.leverage_ratio
    cvar      = result.cvar_95
    hurdle    = inputs.calibrator_config.investor_hurdle_irr
    horizon   = _effective_horizon(inputs)

    if cvar < 0.01:
        risk_note = (
            "Near-zero senior CVaR confirms the capital structure is working as intended — "
            "catalytic layers are absorbing tail losses so commercial investors are fully protected."
        )
    elif cvar < 0.15:
        risk_note = "Portfolio tail risk is within conservative bounds."
    elif cvar < 0.30:
        risk_note = "Portfolio tail risk is moderate — within typical blended-finance tolerances."
    else:
        risk_note = ("Portfolio tail risk is elevated. Consider tightening the CVaR constraint "
                     "or adjusting the vehicle mix.")

    return (
        f"Across {n_active} active vehicle(s), **${total_cat/1e6:.1f}M of catalytic capital "
        f"mobilises ${total_com/1e6:.1f}M of commercial investment** (portfolio leverage: "
        f"**{lev:.1f}\u00d7**). "
        f"All senior tranches were calibrated to a **{hurdle:.1%} IRR hurdle** over a "
        f"**{horizon}-year** investment horizon. "
        f"The blended portfolio CVaR\u2085 is **{cvar:.1%}** — the expected loss rate "
        f"in the worst 5% of simulated scenarios. {risk_note}"
    )


def _sensitivity_commentary(base, mod, test_id: str, label: str) -> str:
    """Plain-language interpretation of a sensitivity result."""
    base_alpha = float(np.mean(list(base.catalytic_fractions.values())))
    mod_alpha  = float(np.mean(list(mod.catalytic_fractions.values())))
    d_alpha = mod_alpha - base_alpha
    d_lev   = mod.leverage_ratio - base.leverage_ratio
    d_cvar  = mod.cvar_95 - base.cvar_95

    alpha_dir = "rises" if d_alpha > 0 else "falls"
    lev_dir   = "falls" if d_lev   < 0 else "rises"
    cvar_dir  = "rises" if d_cvar  > 0 else "falls"

    alpha_note = (
        f"The average catalytic fraction {alpha_dir} by **{abs(d_alpha):.1%}** "
        f"({base_alpha:.1%} \u2192 {mod_alpha:.1%})."
    )
    lev_note = (
        f"Portfolio leverage {lev_dir} to **{mod.leverage_ratio:.1f}\u00d7** "
        f"(from {base.leverage_ratio:.1f}\u00d7)."
    )
    cvar_note = (
        f"Senior tail-loss risk (CVaR\u2085) {cvar_dir} to **{mod.cvar_95:.1%}** "
        f"(from {base.cvar_95:.1%})."
    )

    if test_id == "A" and d_alpha <= 0:
        interp = (
            f"**{label}** reduces the catalytic capital needed — the larger guarantee "
            "absorbs more senior risk directly, so the first-loss tranche can be smaller."
        )
    elif test_id == "A" and d_alpha > 0:
        interp = (
            f"**{label}** raises the guarantee but alpha still increases, suggesting "
            "the portfolio's risk profile is not fully offset by the additional coverage."
        )
    elif test_id == "B" and d_alpha > 0:
        interp = (
            f"**{label}** increases price uncertainty across projects, requiring more "
            "catalytic protection to keep senior lenders at the hurdle IRR."
        )
    elif test_id in ("C", "D") and d_alpha > 0:
        interp = (
            f"**{label}** reduces project revenues, increasing default risk and requiring "
            "a larger catalytic buffer to protect senior lenders."
        )
    else:
        direction = "improves" if d_alpha <= 0 else "worsens"
        interp = f"**{label}** {direction} the capital efficiency of the portfolio."

    return f"{interp}\n\n{alpha_note} {lev_note} {cvar_note}"


# ---------------------------------------------------------------------------
# Page 1: Setup
# ---------------------------------------------------------------------------

def page_setup(cfg: dict) -> None:
    st.header("Portfolio Setup")

    src = st.radio("Data source", ["Sample Portfolio", "Upload Your Own Data"], horizontal=True)

    if src == "Sample Portfolio":
        st.success("3 vehicles pre-loaded: **Forestry** | **Agroforestry** | **Mixed**")
        all_projects: list[list[ProjectInputs]] = []
        all_names: list[str] = []
        for subdir in sorted(p for p in SAMPLE_DIR.iterdir() if p.is_dir()):
            files = sorted(subdir.glob("project_*.csv"))[:MAX_PROJECTS_PER_VEHICLE]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                projs = [load_project_from_excel(f) for f in files]
            vname = "_".join(subdir.name.split("_")[2:]).replace("_", " ").title()
            all_projects.append(projs)
            all_names.append(vname)
            with st.expander(f"Preview: {vname}"):
                for f in files:
                    st.caption(f.name)
                    st.dataframe(pd.read_csv(f).head(5))
        st.session_state["_projects"] = all_projects
        st.session_state["_names"] = all_names
        st.session_state["_data_ok"] = True

    else:
        c1, c2 = st.columns(2)
        c1.download_button("Download Carbon Template", TEMPLATE_CARBON,
                           "template_carbon.csv", "text/csv")
        c2.download_button("Download Commodity Template", TEMPLATE_COMMODITY,
                           "template_commodity.csv", "text/csv")
        st.caption("Name files **vehiclename_projectname.csv** "
                   "(e.g. `forestry_arr.csv` + `forestry_redd.csv` → **Forestry** vehicle)")
        uploaded = st.file_uploader("Upload project CSVs", type=["csv"],
                                    accept_multiple_files=True)
        if uploaded:
            groups: dict[str, list[ProjectInputs]] = {}
            rows = []
            for uf in uploaded:
                vname = Path(uf.name).stem.split("_")[0].title()
                try:
                    df = pd.read_csv(io.BytesIO(uf.read()))
                except Exception as exc:
                    rows.append({"File": uf.name, "Vehicle": vname,
                                 "Status": "❌ Error", "Notes": str(exc)})
                    continue
                errs, wrns = validate_df(df, uf.name)
                if errs:
                    rows.append({"File": uf.name, "Vehicle": vname,
                                 "Status": "❌ Error", "Notes": "; ".join(errs)})
                    continue
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    df.to_csv(tmp.name, index=False)
                    tmp_path = tmp.name
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        proj = load_project_from_excel(tmp_path)
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception as exc:
                    rows.append({"File": uf.name, "Vehicle": vname,
                                 "Status": "❌ Error", "Notes": f"Load error: {exc}"})
                    continue
                groups.setdefault(vname, []).append(proj)
                note = "; ".join(wrns) if wrns else ""
                rows.append({"File": uf.name, "Vehicle": vname,
                             "Status": "⚠ Warning" if wrns else "✓ OK", "Notes": note})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            vnames = list(groups.keys())[:MAX_VEHICLES]
            if len(groups) > MAX_VEHICLES:
                st.warning(f"First {MAX_VEHICLES} vehicles used (MVP limit).")
            all_projects = [groups[n][:MAX_PROJECTS_PER_VEHICLE] for n in vnames]
            st.session_state["_projects"] = all_projects
            st.session_state["_names"] = vnames
            st.session_state["_data_ok"] = bool(all_projects)
            if all_projects:
                st.info("Vehicles: " + " | ".join(
                    f"**{n}** ({len(groups[n][:MAX_PROJECTS_PER_VEHICLE])} project(s))"
                    for n in vnames))
        else:
            st.session_state["_data_ok"] = False

    st.divider()
    with st.expander("Vehicle Configuration (all vehicles)", expanded=True):
        c1, c2, c3 = st.columns(3)
        cfg["guarantee"] = c1.slider("Guarantee Coverage", 0, 60, 25, format="%d%%") / 100
        cfg["reserve_pct"] = c1.slider("Grant Reserve", 0, 15, 5, format="%d%% of capital") / 100
        cfg["mezz_frac"] = c2.slider("Mezzanine Fraction", 0, 30, 10, format="%d%%") / 100
        cfg["senior_coupon"] = c2.slider("Senior Coupon", 5, 15, 8, format="%d%%") / 100
        cfg["corr"] = c3.slider("Off-diagonal Correlation", 0.0, 0.8, 0.30, step=0.05)

    c1, c2, c3 = st.columns(3)
    projs_stored = st.session_state.get("_projects", [])
    auto_budget = sum(_capex_sum(p) for p in projs_stored) * 1.2 / 1e6 if projs_stored else 30.0
    cfg["cat_budget"] = c2.number_input("Catalytic Budget $M (optional)", 0.0, step=0.5,
                                         help="Leave 0 for no separate catalytic cap") * 1e6
    cfg["cvar_max"] = c3.slider("CVaR Limit", 10, 60, 35, format="%d%%") / 100

    data_ok = st.session_state.get("_data_ok", False)
    if st.button("Run Calibration", type="primary", disabled=not data_ok):
        stored_projects = st.session_state["_projects"]
        stored_names = st.session_state["_names"]
        vehicles = [
            _build_vehicle(projs, cfg["guarantee"], cfg["reserve_pct"],
                           cfg["mezz_frac"], cfg["senior_coupon"], cfg["corr"])
            for projs in stored_projects
        ]
        inputs = _build_portfolio(vehicles, cfg["n_sims"], cfg["hurdle_irr"],
                                   cfg["max_loss_prob"], cfg["cvar_max"],
                                   cfg["seed"], cfg["cat_budget"] or None)
        with st.spinner("Calibrating vehicles and optimising portfolio… (30–90 s)"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = PortfolioOptimizer(inputs).run()
                st.session_state.update({"result": result, "inputs": inputs,
                                         "names": stored_names})
                st.success("Done — open the Results page.")
            except Exception as exc:
                st.error(f"Calibration failed: {exc}")



# ---------------------------------------------------------------------------
# Page 2: Results
# ---------------------------------------------------------------------------

def _export_csv(result, names: list[str]) -> str:
    rows = []
    for i, name in enumerate(names):
        rows.append({"vehicle": name,
                     "allocation_usd": result.allocations.get(i, 0),
                     "catalytic_usd": result.catalytic_allocations.get(i, 0),
                     "commercial_usd": result.commercial_allocations.get(i, 0),
                     "alpha": result.catalytic_fractions.get(i, 0),
                     "leverage_x": result.marginal_catalytic_efficiency.get(i, 0)})
    rows.append({"vehicle": "PORTFOLIO",
                 "allocation_usd": sum(result.allocations.values()),
                 "catalytic_usd": sum(result.catalytic_allocations.values()),
                 "commercial_usd": sum(result.commercial_allocations.values()),
                 "alpha": "", "leverage_x": result.leverage_ratio})
    return pd.DataFrame(rows).to_csv(index=False)


def page_results() -> None:
    st.header("Results Dashboard")
    result = st.session_state.get("result")
    inputs = st.session_state.get("inputs")
    names: list[str] = st.session_state.get("names", [])
    if result is None:
        st.info("No results yet — go to **Setup** and run calibration first.")
        return

    # --- Calibration context bar (Enhancement B) ---
    hurdle    = inputs.calibrator_config.investor_hurdle_irr
    max_loss  = inputs.calibrator_config.max_loss_probability
    horizon   = _effective_horizon(inputs)
    st.info(
        f"**Calibration parameters:** {hurdle:.0%} IRR hurdle for senior investors  ·  "
        f"≤{max_loss:.0%} max senior loss probability  ·  "
        f"{inputs.n_sims:,} Monte Carlo simulations  ·  "
        f"**Investment horizon:** {horizon} years"
    )

    ch, ci, ce = st.columns([4, 2, 1])
    icon = "✓" if result.status == "optimal" else "⚠"
    ch.subheader(f"{icon} {result.status.title()}  ·  seed {inputs.seed}")
    ce.download_button("Export CSV", _export_csv(result, names),
                       "results.csv", "text/csv")

    total_cat = sum(result.catalytic_allocations.values())
    total_dep = sum(result.allocations.values())
    irr_clean = result.portfolio_irr_distribution[np.isfinite(result.portfolio_irr_distribution)]
    median_irr = float(np.median(irr_clean)) if len(irr_clean) else float("nan")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Catalytic Capital",
              f"${total_cat/1e6:.1f}M",
              f"{total_cat/max(total_dep,1):.1%} of deployed")
    k2.metric("Portfolio Leverage", f"{result.leverage_ratio:.2f}×",
              "commercial per catalytic $")
    k3.metric(f"Median IRR  (hurdle {hurdle:.0%})",
              f"{median_irr:.1%}" if np.isfinite(median_irr) else "N/A",
              f"{'above' if np.isfinite(median_irr) and median_irr >= hurdle else 'below'} hurdle"
              if np.isfinite(median_irr) else None)
    k4.metric(
        "CVaR (95%)",
        f"{result.cvar_95:.1%}",
        help=(
            "Expected senior-tranche loss rate in the worst 5% of scenarios. "
            "Near-zero values are the intended outcome of blended finance — "
            "it means the first-loss, mezzanine, and guarantee layers have "
            "successfully absorbed all tail losses before the senior tranche "
            "is touched. The catalytic capital IS taking risk; the senior "
            "tranche is being protected."
        ),
    )

    st.divider()
    st.subheader("Vehicle Breakdown")
    rows = []
    for i, name in enumerate(names):
        cat = result.catalytic_allocations.get(i, 0)
        com = result.commercial_allocations.get(i, 0)
        rows.append({"Vehicle": name,
                     "Allocation": f"${result.allocations.get(i,0)/1e6:.1f}M",
                     "Alpha": f"{result.catalytic_fractions.get(i,0):.1%}",
                     "Catalytic": f"${cat/1e6:.1f}M",
                     "Commercial": f"${com/1e6:.1f}M",
                     "Leverage": f"{com/max(cat,1):.1f}×",
                     "Marg. Eff.": f"{result.marginal_catalytic_efficiency.get(i,0):.1f}×"})
    rows.sort(key=lambda r: float(r["Leverage"].replace("×", "")), reverse=True)
    st.dataframe(pd.DataFrame(rows).set_index("Vehicle"))

    # --- Plain language commentary (Enhancement C) ---
    with st.expander("Commentary", expanded=False):
        st.markdown(_portfolio_commentary(result, inputs, names))
        st.divider()
        for i, name in enumerate(names):
            alpha   = result.catalytic_fractions.get(i, 0)
            cat     = result.catalytic_allocations.get(i, 0)
            com     = result.commercial_allocations.get(i, 0)
            lev     = com / max(cat, 1)
            cvar    = result.cvar_95  # vehicle-level CVaR not separately stored; use portfolio
            # Compute median senior IRR if available via tranche results
            # (VehicleResult tranche IRRs not in PortfolioResult — use portfolio-level as proxy)
            irr_proxy = median_irr
            st.markdown(_vehicle_commentary(name, alpha, lev, hurdle,
                                            irr_proxy, cvar, inputs.n_sims))
            if i < len(names) - 1:
                st.divider()

    cl, cr = st.columns(2)
    with cl:
        st.plotly_chart(chart_irr_histogram(result.portfolio_irr_distribution))
    with cr:
        st.plotly_chart(chart_capital_stack(result, names))
    st.plotly_chart(chart_alpha(result, names))

    # --- Base-case cashflow charts (Enhancement A) ---
    cf_figs = chart_cashflows(inputs, names)
    if cf_figs:
        with st.expander("Underlying Cash Flows — base-case inputs to Monte Carlo", expanded=False):
            st.caption(
                "These charts show the **deterministic base-case cashflows** loaded from your "
                "project files — the starting point before Monte Carlo price and yield shocks "
                "are applied. Revenue bars are pre-shock (expected value); actual simulated "
                "paths will fan out around these figures."
            )
            for vname, fig in cf_figs:
                st.plotly_chart(fig, use_container_width=True)



# ---------------------------------------------------------------------------
# Page 3: Sensitivity
# ---------------------------------------------------------------------------

def _sens_comparison(base, mod, names: list[str], label: str) -> None:
    b_irr = base.portfolio_irr_distribution
    m_irr = mod.portfolio_irr_distribution
    bm = float(np.median(b_irr[np.isfinite(b_irr)])) if np.any(np.isfinite(b_irr)) else float("nan")
    mm = float(np.median(m_irr[np.isfinite(m_irr)])) if np.any(np.isfinite(m_irr)) else float("nan")
    ba = float(np.mean(list(base.catalytic_fractions.values())))
    ma = float(np.mean(list(mod.catalytic_fractions.values())))
    rows = [
        {"Metric": "Mean Alpha", "Base": f"{ba:.1%}", "Modified": f"{ma:.1%}",
         "Δ": f"{(ma-ba)*100:+.1f} pp"},
        {"Metric": "Portfolio Leverage", "Base": f"{base.leverage_ratio:.2f}×",
         "Modified": f"{mod.leverage_ratio:.2f}×",
         "Δ": f"{mod.leverage_ratio-base.leverage_ratio:+.2f}×"},
        {"Metric": "Median IRR",
         "Base": f"{bm:.1%}" if np.isfinite(bm) else "N/A",
         "Modified": f"{mm:.1%}" if np.isfinite(mm) else "N/A",
         "Δ": f"{(mm-bm)*100:+.1f} pp" if np.isfinite(bm) and np.isfinite(mm) else "—"},
        {"Metric": "CVaR 95%", "Base": f"{base.cvar_95:.1%}",
         "Modified": f"{mod.cvar_95:.1%}",
         "Δ": f"{(mod.cvar_95-base.cvar_95)*100:+.1f} pp"},
    ]
    st.dataframe(pd.DataFrame(rows).set_index("Metric"))
    idxs = sorted(base.catalytic_fractions.keys())
    fig = go.Figure([
        go.Bar(name="Base", x=[names[i] for i in idxs],
               y=[base.catalytic_fractions[i]*100 for i in idxs], marker_color="#1565C0"),
        go.Bar(name=label, x=[names[i] for i in idxs],
               y=[mod.catalytic_fractions[i]*100 for i in idxs], marker_color="#EF6C00"),
    ])
    fig.update_layout(barmode="group", yaxis_title="Alpha (%)", height=300,
                      title=f"Alpha: Base vs {label}",
                      margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig)


def page_sensitivity() -> None:
    st.header("Sensitivity Analysis")
    base_result = st.session_state.get("result")
    base_inputs = st.session_state.get("inputs")
    names: list[str] = st.session_state.get("names", [])
    if base_result is None:
        st.info("Run calibration first (Setup page).")
        return

    test = st.radio("Select test", [
        "A: Higher guarantee coverage",
        "B: Higher price volatility",
        "C: Lower operating cashflows",
        "D: Lower base price",
    ])
    tid = test[0]

    ca, cb = st.columns([1, 2])
    with ca:
        if tid == "A":
            base_g = base_inputs.vehicles[0].guarantee_coverage
            new_g = st.slider("New guarantee (%)", 0, 60, min(int(base_g*100)+20, 60), 5) / 100
            val, lbl = new_g, f"{base_g:.0%} → {new_g:.0%}"
        elif tid == "B":
            mult = st.slider("Vol multiplier", 1.0, 3.0, 2.0, 0.1)
            avg = float(np.mean([p.price_vol for v in base_inputs.vehicles for p in v.projects]))
            val, lbl = mult, f"{avg:.0%} avg → {avg*mult:.0%} avg (×{mult:.1f})"
        else:
            scale = st.slider("Scale factor", 0.50, 1.00, 0.80, 0.05)
            val, lbl = scale, f"100% → {scale:.0%}"
    with cb:
        st.info(f"**{lbl}**")

    if st.button("Run Sensitivity", type="primary"):
        mod_inputs = _rebuild_sensitivity(base_inputs, tid, val)
        with st.spinner("Running modified scenario…"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod_result = PortfolioOptimizer(mod_inputs).run()
                st.subheader(f"Comparison: Base vs {lbl}")
                _sens_comparison(base_result, mod_result, names, lbl)
                with st.expander("Commentary", expanded=True):
                    st.markdown(_sensitivity_commentary(base_result, mod_result, tid, lbl))
            except Exception as exc:
                st.error(f"Sensitivity run failed: {exc}")


# ---------------------------------------------------------------------------
# Page 4: How It Works
# ---------------------------------------------------------------------------

def page_how_it_works() -> None:
    st.header("How It Works")
    st.markdown("""
## What is this tool for?

This tool helps foundations and development finance practitioners answer a practical
question before committing capital: **how much concessional funding does a blended-finance
deal actually need?**

Rather than guessing, you describe your projects (revenues, costs, price risk), set the
return your commercial investors require, and the tool calculates the *minimum* grant,
first-loss, and guarantee support needed to make those investors whole — across all the
market conditions your projects might face.

Use it to:
- **Size your catalytic budget** before structuring a vehicle or approaching co-investors
- **Compare alternative structures** (more guarantee vs. more first-loss vs. larger reserve)
- **Stress-test assumptions** — see how results change if prices fall or volatility rises
- **Communicate to boards and donors** how efficiently each concessional dollar is being used

---

## Why catalytic capital matters

Private investors — pension funds, banks, insurance companies — will only invest in
emerging-market or nature-based projects if the risk-adjusted return meets their threshold.
Many high-impact projects are too uncertain on their own to clear that bar.

**Catalytic capital** (grants, first-loss equity, DFI guarantees) changes that equation by
absorbing the first share of any losses. Think of it like an insurance deductible: a
senior lender who knows the first 20% of losses will be covered by someone else faces a
much less risky investment. The commercial investor's downside is protected; the foundation
or DFI takes the concentrated risk in exchange for the impact.

The key metric is **α (alpha)** — the share of a vehicle's total capital that must be
concessional. If α = 25%, then for every $75 a commercial bank puts in, a foundation
or DFI provides $25 of first-loss/guarantee support. A lower α means each catalytic
dollar is doing more work.

---

## How the tool works — three steps

**Step 1 — Describe your projects.**
Upload a CSV for each investment: how many tonnes of carbon or commodity it will produce
each year, what price you expect, how risky that price is, and what it costs to build
and run. The tool runs thousands of simulated futures (high prices, low prices, average
prices) to map out the range of outcomes each project might deliver.

**Step 2 — Size the catalytic stack.**
You set the return commercial investors require (the hurdle IRR) and the maximum chance
of loss they'll accept. The tool then searches for the smallest catalytic fraction α
that still meets both conditions across all those simulated scenarios. It tests
combinations of first-loss equity, a cash reserve, a DFI guarantee on senior debt, and
a mezzanine layer — all configurable in the Setup panel.

**Step 3 — Allocate across your portfolio.**
If you have several vehicles competing for a fixed catalytic budget, the tool uses a
mathematical optimisation to find the allocation that mobilises the most commercial
capital while keeping the portfolio's tail risk within bounds you set.

---

## How to read the results

| Metric | What it means |
|--------|---------------|
| **Catalytic fraction (α)** | Share of total vehicle capital that must be concessional. α = 20% means $1 of catalytic unlocks $4 of commercial. |
| **Leverage ratio** | Commercial capital mobilised per catalytic dollar. 4× means each $1 of grant/first-loss brings in $4 from banks. |
| **Senior IRR** | The expected annual return for commercial investors, at the median of all simulated scenarios. Should meet or exceed the hurdle you set. |
| **CVaR (95%)** | The expected senior-tranche loss rate in the worst 5% of scenarios. Near-zero means the catalytic structure is working — commercial investors are protected even in bad outcomes. |

> **CVaR near zero is the goal, not a sign something is wrong.** It means the first-loss,
> mezzanine, and guarantee layers have absorbed all tail losses before the senior tranche
> is touched. The catalytic capital *is* taking risk — it's just doing so on behalf of
> commercial investors, as intended.

---

## CSV format reference

| Column | Required | Description |
|--------|----------|-------------|
| `year` | ✓ | Calendar year. Construction rows: yield = 0. |
| `yield` | ✓ | Physical output / year (tCO2e, tons, m³). 0 = construction. |
| `capex` | ✓ | Capital expenditure (positive = outflow). 0 during operations. |
| `opex` | ✓ | Operating cost / year (positive = outflow). |
| `revenue_type` | optional | "carbon" or "commodity" — label only, same math. |
| `base_price` | ✓ | Current price per unit (USD). |
| `price_growth_rate` | ✓ | Annual log-price drift (e.g. 0.05 = 5 %/yr). |
| `price_vol` | ✓ | Annual price volatility (e.g. 0.30 = 30 %). |

**Revenue types:** Commodity (cocoa, timber) and Carbon (REDD+, ARR, biochar) use the
same columns and the same price-simulation math. The label is for your reference only.

## Typical project parameters

| Project type | Yield units | Base price | Growth | Vol |
|-------------|------------|-----------|--------|-----|
| Forestry ARR | tCO2e / yr | $15 | 5 % | 30 % |
| REDD+ | tCO2e / yr | $12 | 6 % | 35 % |
| Biochar | tons / yr | $130–200 | 4 % | 28 % |
| Agroforestry (cocoa) | tons / yr | $1,800 | 3 % | 22 % |

## Run time
~30–90 s for 1,000 simulations with 3 vehicles (3–5 projects each).
""")
    c1, c2 = st.columns(2)
    c1.download_button("Carbon Template (REDD+, ARR, biochar)",
                       TEMPLATE_CARBON, "template_carbon.csv", "text/csv")
    c2.download_button("Commodity Template (agroforestry, timber)",
                       TEMPLATE_COMMODITY, "template_commodity.csv", "text/csv")


# ---------------------------------------------------------------------------
# Code Review page
# ---------------------------------------------------------------------------

def page_code_review() -> None:
    st.header("AI Code Review")
    st.markdown(
        "Uses the OpenAI API to review the calibration tool's core source files "
        "for mathematical errors, edge cases, and numerical stability issues — "
        "a second model's perspective on the implementation."
    )

    try:
        from calibration.plugins.openai_codex import CodexReviewer  # noqa: PLC0415
    except ImportError:
        st.error(
            "The `openai` package is not installed. "
            "Run: `pip install -e \".[codex]\"` (from repo root) or `pip install openai`."
        )
        return

    with st.form("codex_form"):
        try:
            _default_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            _default_key = ""
        _default_key = _default_key or os.environ.get("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=_default_key,
            help="Or set OPENAI_API_KEY in the environment / Streamlit Cloud secrets.",
        )
        model = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"], index=0)
        files_to_review = st.multiselect(
            "Files to review",
            options=CodexReviewer.DEFAULT_FILES,
            default=CodexReviewer.DEFAULT_FILES,
        )
        submitted = st.form_submit_button("Run Code Review", type="primary")

    if not submitted:
        return

    if not api_key:
        st.error("An OpenAI API key is required.")
        return

    if not files_to_review:
        st.warning("No files selected.")
        return

    with st.spinner(f"Reviewing {len(files_to_review)} file(s) via OpenAI API…"):
        try:
            reviewer = CodexReviewer(model=model, files=files_to_review, api_key=api_key)
            result = reviewer.review()
        except RuntimeError as exc:
            st.error(str(exc))
            return

    criticals = [f for f in result.findings if f.severity == "critical"]
    warnings_  = [f for f in result.findings if f.severity == "warning"]
    infos      = [f for f in result.findings if f.severity == "info"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files reviewed", len(result.files_reviewed))
    c2.metric("Critical", len(criticals))
    c3.metric("Warnings", len(warnings_))
    c4.metric("Info", len(infos))

    if result.errors:
        with st.expander(f"Errors ({len(result.errors)})", expanded=False):
            for err in result.errors:
                st.warning(err)

    if not result.findings:
        st.success("No issues found.")
        return

    for label, group, color in (
        ("Critical", criticals, "red"),
        ("Warnings", warnings_, "orange"),
        ("Info", infos, "blue"),
    ):
        if not group:
            continue
        with st.expander(f"{label} ({len(group)})", expanded=(label == "Critical")):
            for finding in group:
                hint = f" — line {finding.line_hint}" if finding.line_hint else ""
                st.markdown(
                    f"**`{finding.file}`{hint}** `[{finding.category}]`\n\n"
                    f"{finding.description}"
                )
                st.divider()

    st.caption(
        f"Model: {result.model_used}  |  Tokens used: {result.total_tokens_used:,}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Catalytic Capital Optimizer",
                       page_icon="🌿", layout="wide",
                       initial_sidebar_state="expanded")

    if not check_auth():
        st.stop()
        return

    with st.sidebar:
        st.title("🌿 Catalytic Capital")
        st.divider()
        _pages = ["Setup", "Results", "Sensitivity", "How It Works"]
        _page_labels = {
            "Setup": "📁  Setup",
            "Results": "📊  Results",
            "Sensitivity": "🔬  Sensitivity",
            "How It Works": "ℹ️  How It Works",
        }
        if _OPENAI_AVAILABLE:
            _pages.append("Code Review")
            _page_labels["Code Review"] = "🤖  Code Review"
        page = st.radio("Navigate", _pages, format_func=lambda p: _page_labels[p])
        st.divider()
        st.subheader("Run Settings")
        n_sims = st.slider("Simulations", 100, 2000, 500, 100)
        hurdle = st.slider("Hurdle IRR", 4, 15, 7, format="%d%%") / 100
        max_loss = st.slider("Max Loss Prob", 2, 20, 8, format="%d%%") / 100
        seed = int(st.number_input("Seed", value=42, min_value=0, max_value=99999))
        if st.session_state.get("result"):
            st.divider()
            status = st.session_state["result"].status
            st.success(f"Last run: {status}") if status == "optimal" else st.warning(f"Last run: {status}")
        st.divider()
        if st.button("Logout"):
            logout()
            st.rerun()

    cfg = {"n_sims": n_sims, "hurdle_irr": hurdle, "max_loss_prob": max_loss, "seed": seed,
           "guarantee": 0.25, "reserve_pct": 0.05, "mezz_frac": 0.10,
           "senior_coupon": 0.08, "corr": 0.30, "cvar_max": 0.35, "cat_budget": None}

    if page == "Setup":
        page_setup(cfg)
    elif page == "Results":
        page_results()
    elif page == "Sensitivity":
        page_sensitivity()
    elif page == "Code Review":
        page_code_review()
    else:
        page_how_it_works()


if __name__ == "__main__":
    main()







