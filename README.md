# Calibration Tool

**Solve for the minimum catalytic capital a blended-finance vehicle needs, then allocate it across a portfolio.**

Blended-finance vehicles use concessional money (grants, first-loss equity,
guarantees) to make deals investable for commercial lenders. The open question
is always *how much* concessional capital is actually needed. This tool answers
it by Monte Carlo simulation rather than rule of thumb:

- **Project** — simulate each project's cashflows under price uncertainty.
- **Vehicle** — pool projects into a tranched fund and solve for the minimum
  catalytic fraction **α** at which senior lenders meet their hurdle IRR and
  loss-probability limit.
- **Portfolio** — allocate a foundation's catalytic budget across vehicles to
  maximise commercial capital mobilised, subject to a CVaR tail-risk limit.

Catalytic capital is never an input — it is always a solved output.

> **Disclaimer — illustrative model, not investment advice.** This is a
> simplified reference implementation for exploring blended-finance structures.
> It is not investment, legal, or financial advice, and its outputs must not be
> the sole basis for any capital commitment. All bundled sample data is
> synthetic. See [Model limitations](#model-limitations) and [LICENSE](LICENSE)
> (Apache-2.0, provided "as is" without warranty).

---

## Key concepts

| Term | Plain-English meaning |
|---|---|
| **Catalytic capital** | Concessional money (grants, first-loss equity, guarantees) that takes the most risk so private investors will join |
| **α (alpha)** | Share of a vehicle's capital that must be catalytic. The tool solves for the *minimum* α |
| **Tranche** | A layer of a fund with a defined risk/return priority. Senior = safest, first-loss = riskiest |
| **Waterfall** | Who gets paid first (cashflow waterfall) and who absorbs losses first (loss waterfall) |
| **IRR** | Internal Rate of Return — the annualised yield an investor earns |
| **CVaR (95%)** | Expected loss rate across the worst 5% of simulated scenarios — tail risk |
| **Leverage** | Commercial capital mobilised per catalytic dollar, `(1−α)/α`. 3× = $3 private per $1 concessional |

Losses are absorbed junior-first: **grant reserve → first-loss → mezzanine →
guarantee (wraps senior) → senior**. The guarantee protects senior lenders, not
catalytic capital, matching DFI practice (MIGA, DFC, GuarantCo).

---

## Quick start

Requires Python 3.11+.

```bash
pip install -e ".[dev,ui]"
```

**Web demo** — opens at http://localhost:8501 with sample data pre-loaded; click **Run Calibration**:

```bash
streamlit run app.py
```

**Command line** — runs the built-in two-vehicle sample:

```bash
python run_e2e.py --seed 42
```

**Tests** (about a minute):

```bash
pytest
```

---

## Example output

From `python run_e2e.py --seed 42` (built-in sample, 1,000 simulations):

```
  Solver status          : optimal
  Total budget           : $  10,000,000
  Total catalytic capital: $   1,580,693  (15.8% of deployed)
  Total commercial capital: $  8,419,307  (84.2% of deployed)
  Portfolio leverage ratio:       5.33x  (commercial per catalytic $)
  Portfolio CVaR (95%)   :        0.0%  of deployed capital

                              Allocation $ Alpha (cat %) Catalytic $ Commercial $ Leverage (x)
East Africa Nature Fund          4,000,000         14.3%     573,517    3,426,483         6.0x
West Africa Clean Energy Fund    6,000,000         16.8%   1,007,176    4,992,824         5.0x
```

- **Alpha** — minimum share of the vehicle that must be first-loss + grant reserve.
- **Leverage** — commercial dollars mobilised per catalytic dollar.
- **CVaR** — expected senior loss rate in the worst 5% of scenarios; must stay below `cvar_max`.
- **Solver status** — `optimal` / `optimal_inaccurate` are usable. `calibration_infeasible`
  means at least one vehicle cannot meet investor targets at any α (see
  `result.infeasible_vehicles`; those vehicles are excluded). If
  `result.lp_status` is anything else, the allocation LP failed and allocations are zero. See [Troubleshooting](#troubleshooting).

---

## Bringing your own data

| Route | Best for | Details |
|---|---|---|
| **Streamlit upload** | Non-technical users; one CSV per project (`year,yield,capex,opex,base_price,price_growth_rate,price_vol`) | [Format 2](docs/DATA_FORMATS.md#streamlit-upload-format-format-2) |
| **`--folder DIR`** | Project financial models exported per project, optional historical price series | [Option C](docs/DATA_FORMATS.md#option-c--folder-per-vehicle-with-yearyieldcapexopex-files) |
| **`--json FILE`** | Full control over vehicle structure (guarantee, reserve, coupons, correlation) | [Option B](docs/DATA_FORMATS.md#option-b--json-portfolio-spec) |
| **Python API** | Notebooks and integration | [below](#python-api) |

Templates can be downloaded from the app's **Setup** and **How It Works** pages.
Upload limits for the demo: 3 vehicles, 5 projects per vehicle, 30 years, CSV only.

---

## Python API

```python
from calibration.project.models import ProjectInputs
from calibration.project.simulation import ProjectSimulator
from calibration.vehicle.calibration import CalibratorConfig, CatalyticCalibrator
from calibration.vehicle.capital_stack import CapitalStack
from calibration.vehicle.risk_mitigants import Guarantee, GrantReserve, CoverageType

# 1. Simulate a single project
project = ProjectInputs(
    capex=1_000_000, opex_annual=50_000,
    price=12.0, yield_=22_000, lifetime_years=10, price_vol=0.25,
)
sim_result = ProjectSimulator(project).run(n_sims=2000, seed=42)
print(f"Loss probability: {sim_result.loss_probability:.1%}")

# 2. Build a capital stack and calibrate
stack = CapitalStack(
    total_capital=1_000_000,
    grant_reserve=GrantReserve(50_000),
    guarantee=Guarantee(0.30, CoverageType.PERCENTAGE),
    senior_coupon=0.08, mezzanine_coupon=0.12,
    mezzanine_fraction=0.10, lifetime_years=10,
)
calibrator = CatalyticCalibrator(
    capital_stack=stack,
    vehicle_cashflows=sim_result.cashflows,
    config=CalibratorConfig(investor_hurdle_irr=0.07, max_loss_probability=0.08),
)
alpha_star = calibrator.calibrate()
print(f"Minimum catalytic fraction: {alpha_star:.1%}")
print(f"Catalytic capital: ${alpha_star * 1_000_000:,.0f}")

# 3. Run the full portfolio pipeline
from calibration.portfolio.models import PortfolioInputs
from calibration.portfolio.optimizer import PortfolioOptimizer
from calibration.vehicle.models import VehicleInputs

vehicle = VehicleInputs(
    projects=[project],
    correlation_matrix=[[1.0]],
    total_capital=1_000_000,
    guarantee_coverage=0.30,
    grant_reserve=50_000,
)
portfolio_inputs = PortfolioInputs(
    vehicles=[vehicle],
    total_budget=1_000_000,
    n_sims=1000,
    seed=42,
    calibrator_config=CalibratorConfig(investor_hurdle_irr=0.07, max_loss_probability=0.08),
)
result = PortfolioOptimizer(portfolio_inputs).run()
print(f"Solver status:    {result.status}")
print(f"Leverage ratio:   {result.leverage_ratio:.2f}x")
print(f"CVaR (95%):       {result.cvar_95:.1%}")
# Per-vehicle breakdown (keyed by vehicle index)
for v_idx, alloc in result.allocations.items():
    alpha = result.catalytic_fractions[v_idx]
    marg  = result.marginal_catalytic_efficiency[v_idx]
    print(f"  Vehicle {v_idx}: ${alloc:,.0f} total, α={alpha:.1%}, marg.eff={marg:.2f}x")
# Full distributions for further analysis
# result.portfolio_irr_distribution  → np.ndarray shape (n_sims,)
# result.portfolio_loss_distribution → np.ndarray shape (n_sims,)
```

---

---

## Model limitations

Read these before relying on any number the tool produces:

- **Price risk only (CSV formats).** Revenue is shocked by a GBM price path;
  yield, cost overrun, delay, FX and counterparty risk are not modelled unless
  you use the parametric format.
- **Bullet maturity.** Senior and mezzanine principal is repaid entirely at the
  final period; there is no amortisation, refinancing, or reserve account.
- **Two waterfall bases.** Loss probability comes from a terminal NPV loss
  waterfall; IRR comes from a per-period cashflow waterfall. The guarantee
  reduces senior loss probability but does not add cash to senior IRR.
- **Correlation is approximate.** Projects are correlated by rank-reordering
  lifetime cashflow totals (Iman-Conover style), not by correlating the
  underlying risk drivers. Vehicles are treated as independent of each other.
- **Monte Carlo noise.** At 100–2,000 paths, tail metrics (CVaR, loss
  probability near the threshold) move with the seed. Re-run with several seeds
  before quoting a figure.
- **Infeasible results are flagged, not solved.** If no catalytic fraction
  meets the investor constraints, the vehicle is listed in
  `PortfolioResult.infeasible_vehicles`, gets zero allocation, has α reported
  as `NaN`, and `status` is `calibration_infeasible`. If the portfolio LP
  fails, allocations are zero and `lp_status` gives the solver status.

---

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md) | All input formats, CLI options, changing constraints |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Validation protocol, mock portfolio, expected sensitivity directions |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Optional password gate and Streamlit Cloud deployment |
| [CLAUDE.md](CLAUDE.md) | Architecture, design decisions, and developer notes (also used by Claude Code) |

## Project structure

```
calibration/
├── project/     # Monte Carlo project simulation (GBM price paths)
├── vehicle/     # Capital stack, dual waterfall, catalytic calibration
├── portfolio/   # CVaR-constrained allocation LP (cvxpy)
└── utils/       # IRR, CVaR, correlated draws, file loaders
app.py           # Streamlit demo
auth.py          # Optional password gate for hosted demos
run_e2e.py       # Command-line runner
validate_e2e.py  # Analyst validation report
examples/        # Synthetic sample data
tests/           # pytest suite
docs/            # Detailed documentation
```

---

## Troubleshooting

**Vehicle flagged infeasible** (`status == "calibration_infeasible"`, or `ValueError: Constraints infeasible` from `CatalyticCalibrator`)
No `alpha` meets both the hurdle IRR and max loss probability. Options:
- Lower `investor_hurdle_irr` (e.g. from 0.10 to 0.07)
- Raise `max_loss_probability` (e.g. from 0.03 to 0.08)
- Add a guarantee (`guarantee_coverage > 0`)
- Increase `grant_reserve`

**LP solver returns `infeasible`**
The portfolio allocation LP has no feasible solution; allocations are returned as zero. Options:
- Raise `cvar_max` (e.g. from 0.20 to 0.35)
- Lower `min_expected_return`
- Raise `max_allocation_fraction` (allow more concentration)
- Lower `min_deployment` if it was set too high relative to `total_budget`

**LP solver returns all-zero weights**
No capital is deployed. Set `min_deployment > 0` in `PortfolioInputs` (or `"min_deployment"` in JSON) to enforce a minimum total deployment floor.

**`RuntimeWarning: overflow encountered in power / multiply`**
Harmless. Newton's method for IRR evaluation hits overflow on high-return outlier paths. The sentinel cap (`10.0` = 1000% IRR) is applied by `clean_irr()` afterwards. Results are unaffected.

**`UserWarning: Base-case lifetime revenue is less than capex`**
The project has negative expected NPV at base-case assumptions. This is a warning, not an error — the Monte Carlo will still run. Review your `price`, `yield_`, and `lifetime_years` inputs.

**Slow runtime**
Default is 1000 simulations. For faster iteration during model development:
```bash
python run_e2e.py --sims 200
```
For production / final results, use 5000–10000:
```bash
python run_e2e.py --json examples/portfolio.json --sims 10000
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
