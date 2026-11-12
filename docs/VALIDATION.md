# Validation

The validation protocol, bundled mock portfolio, and expected sensitivity directions.

[← Back to README](../README.md)

---

A structured 6-step validation protocol verifies that the system produces economically
correct, numerically stable, and decision-useful outputs.

### Mock dataset

A ready-to-use mock portfolio is included at `examples/mock_portfolio/` with 3 vehicles
× 3 projects representing nature-based investments:

```
examples/mock_portfolio/
  vehicle_1_forestry/
    forestry_arr.csv           ARR/timber, $8M capex, 3yr construction
    forestry_conservative.csv  Lower-yield ARR, $7M capex
    forestry_risky.csv         High capex ($9M), delayed income — tests negative-NPV flow-through
  vehicle_2_agroforestry/
    agro_standard.csv          J-curve (multi-year negative before positive)
    agro_conservative.csv      Lower-yield agroforestry
    agro_volatile.csv          Alternating cashflow pattern (high dispersion)
  vehicle_3_mixed/
    redd_credits.csv           REDD+, cash-generating from year 1
    biochar_volatile.csv       Biochar with alternating high/low cashflows
    hybrid_diversified.csv     Hybrid with 2yr construction then steady growth
```

All files use `year,cashflow` CSV format (years 0–15). The loader auto-detects this format.

### Run the mock portfolio through the main CLI

```bash
python run_e2e.py --folder examples/mock_portfolio/ --sims 1000 --seed 42
```

### Automated validation (pytest — for CI)

```bash
# Full e2e test suite (35 tests covering all 6 steps)
pytest tests/test_e2e.py -v

# Run a specific step
pytest tests/test_e2e.py -v -k Step2    # deterministic validation
pytest tests/test_e2e.py -v -k Step3    # Monte Carlo
pytest tests/test_e2e.py -v -k Step4    # sensitivity checks

# Full test suite (unit + integration + e2e)
pytest -v
```

### Interactive validation script (for analysts)

```bash
python validate_e2e.py                     # 1 000 sims, seed 42
python validate_e2e.py --sims 500          # faster run
python validate_e2e.py --sims 2000 --seed 7
python validate_e2e.py --charts            # also produce matplotlib charts
```

The script prints a structured report with pass/fail for 18 checks:

| Step | What is checked |
|------|----------------|
| 1 Data loading | 3 vehicles × 3 projects loaded; CF lengths; capex sign |
| 2 Deterministic | Capex determinism; multi-year construction; negative-NPV flow-through; waterfall ordering |
| 3 Monte Carlo | Solver status; no overflow warnings; IRR percentiles; leverage; loss bounds |
| 4 Sensitivity | Higher guarantee → alpha ↓; higher vol → dispersion ↑; lower CFs → alpha ↑ |
| 5 Diagnostics | NaN ratio bounds; loss statistics; IrrDiagnostics API |
| 6 Charts | IRR histogram; loss distribution (optional, requires matplotlib) |

### Sensitivity tests: expected directions

| Change | Expected direction | Economic rationale |
|--------|-------------------|-------------------|
| Guarantee coverage 25% → 50% | Alpha decreases | DFI covers more senior losses; less first-loss needed |
| Price volatility 0.15 → 0.40 | IRR dispersion increases | Wider revenue swings → wider return distribution |
| Operating cashflows × 0.80 | Alpha increases | Weaker economics → more concessional capital required |

### Diagnostic thresholds

| Metric | Flag if... | Likely cause |
|--------|-----------|-------------|
| NaN IRR ratio | > 40% | Cashflow assumptions too aggressive or project too short |
| NaN IRR ratio | = 0% | Possible over-filtering; verify edge cases are reaching solver |
| All loss rates = 0% | Always | Capital stack may be over-sized; consider tighter guarantee |

---
