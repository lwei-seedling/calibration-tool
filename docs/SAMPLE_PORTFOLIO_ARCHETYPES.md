# Sample Portfolio Archetypes — Draft for Review

**Status:** parameter draft, not yet implemented.
**Purpose:** replace the current synthetic built-in sample (East/West Africa generic) with three sector-grounded archetypes whose calibrated α we can defend to a foundation principal.

The numbers below are **parameter ranges drawn from public sources**, not specific deals. Once you approve the ranges (or override with figures from your own deal sheets), I'll generate the Format-2 CSVs in `examples/ui_sample/` and update `run_e2e.py`'s built-in fixture.

---

## Why three archetypes (not three randomised vehicles)

The current built-in is two near-identical synthetic geographies. A foundation reviewer can't tell what they're looking at. Three sector archetypes do three things:

1. Give visibly different risk profiles → the LP allocates *differently* between them, demonstrating its value
2. Anchor `price_vol`, `capex`, and `yield` to public benchmarks reviewers will recognise
3. Stress different parts of the model (carbon price collapse vs FX vs weather/yield)

Cross-check the calibrated α against **Convergence's State of Blended Finance** (free annual report) — typical median concessional share in climate-focused vehicles is **25–40%**, with first-of-kind structures running 40–60%. The current sample's 17% is on the low end and not reflective of the sector.

---

## Archetype 1 — VCM Forestry / ARR (Afforestation, Reforestation, Revegetation)

**Vehicle thesis:** pool 3 nature-based carbon projects in Sub-Saharan Africa or Latin America, sell credits into the voluntary carbon market.

**Why this archetype:** highest volatility and integrity-risk sector in blended finance. Stresses the carbon-price GBM and the loss waterfall (catalytic capital absorbs methodology/price-shock losses).

### Vehicle-level parameters

| Parameter | Value | Rationale |
|---|---|---|
| `total_capital` | $8M | Mid-size NbS fund (Mirova Land Degradation Neutrality, Althelia Climate Fund range: $5–25M per vehicle) |
| `guarantee_coverage` | 0.15 | DFI guarantees for VCM are still rare; partial cover via Article 6 / sovereign agreements |
| `grant_reserve` | 5% ($400k) | Project preparation, MRV cost reserve |
| `senior_coupon` | 0.05 | DFI senior pricing for nature funds (e.g. AGRI3 senior tranche) |
| `mezzanine_fraction` | 0.15 | Foundation-DFI mezz layer |
| `investor_hurdle_irr` (calibrator) | 0.07 | Mission-aligned senior — IFC/FMO patient capital |
| `max_loss_probability` | 0.10 | Higher than infra; market accepts more tail risk |

### Per-project parameters (3 projects)

| Project | Location archetype | `lifetime_years` | `capex` | Year-1 `yield` (tCO₂e/yr) | `base_price` ($/tCO₂e) | `price_vol` | Notes |
|---|---|---|---|---|---|---|---|
| ARR-1 (mature) | Kenya / Tanzania mixed | 30 | $2.2M (yrs 0–3) | 18,000 (from yr 5) | 14 | 0.35 | Largest, longest crediting period |
| ARR-2 (early-stage) | West Africa | 25 | $1.5M (yrs 0–4) | 9,000 (from yr 6) | 12 | 0.40 | Higher uncertainty, longer construction |
| ARR-3 (mangrove / blue-carbon premium) | Coastal Madagascar | 30 | $1.8M (yrs 0–3) | 11,000 (from yr 5) | 22 | 0.30 | Premium credit type, lower vol due to integrity premium |

**Construction period:** 3–5 years (planting + early growth, ~0 credits issued).
**Operating period:** issuance begins year 5–6, continues to year 25–30.

### Source anchors

| Parameter | Source |
|---|---|
| Capex $1.5–4k/ha for ARR | IPCC AFOLU Guidelines 2019 refinement; Verra VM0047 cost benchmarks |
| Yield 5–15 tCO₂e/ha/yr | IPCC AFOLU Tier-1 defaults for tropical/sub-tropical |
| Base price (VCM ARR credit) $8–22 | Trove Research VCM market reports 2023–2024; Xpansiv CBL N-GEO closing 2024 |
| `price_vol` 30–40% annual | Xpansiv N-GEO historical 2020–2024 log-return vol (the 2023 REDD+ price collapse drives the upper end) |
| Buffer pool / non-permanence | Apply 15–20% to issuance (Verra AFOLU non-permanence risk tool) |

---

## Archetype 2 — DFI-Backed Solar IPP (Sub-Saharan Africa)

**Vehicle thesis:** finance 2 utility-scale solar PV plants under USD-denominated PPAs with sovereign-backed offtakers.

**Why this archetype:** lowest volatility but highest capex intensity and FX/counterparty concentration. Stresses the senior tranche IRR mechanics (long-life bullet maturity, contracted cashflows) and the guarantee wrap.

### Vehicle-level parameters

| Parameter | Value | Rationale |
|---|---|---|
| `total_capital` | $60M | 2 plants × 30 MW × ~$1k/kW capex |
| `guarantee_coverage` | 0.50 | MIGA / ATI political-risk + partial-credit wrap is standard for African IPPs |
| `grant_reserve` | 2% ($1.2M) | Lower than NbS — costs are mostly EPC, not preparation |
| `senior_coupon` | 0.07 | Hard-currency project finance senior in SSA |
| `mezzanine_fraction` | 0.10 | Sponsor equity / sub-debt |
| `investor_hurdle_irr` (calibrator) | 0.09 | Commercial DFI senior |
| `max_loss_probability` | 0.05 | Infrastructure tolerance is tight |

### Per-project parameters (2 projects)

| Project | Plant size | `lifetime_years` | Construction yrs | `capex` total | Year-1 operating revenue | `price_vol` | Notes |
|---|---|---|---|---|---|---|---|
| Solar-1 | 30 MW PV | 25 | 2 | $30M (yrs 0–1) | $2.6M (~52,500 MWh × $0.05/kWh) | 0.08 | USD PPA, MIGA-wrapped |
| Solar-2 | 25 MW PV | 25 | 2 | $24M (yrs 0–1) | $2.2M (~44,000 MWh × $0.05/kWh) | 0.10 | Hybrid USD/local PPA, slightly higher vol |

**Capacity factor:** 18–22% (PVGIS Africa estimates for typical sites).
**O&M:** 1.5–2% of capex per year (~$500k–$600k/plant).

### Source anchors

| Parameter | Source |
|---|---|
| Utility-scale solar capex $900–1,400/kW (Africa) | IRENA "Renewable Power Generation Costs 2023" |
| PPA tariffs $0.04–$0.08/kWh | Scaling Solar (Senegal $0.038, Zambia $0.06); REIPPPP rounds 4–5; DFC project disclosures |
| Capacity factor 18–22% | IRENA Africa solar atlas; PVGIS |
| `price_vol` 5–10% (USD PPA) | Contracted revenue — vol comes from offtaker performance, curtailment, plant availability, not energy price |
| Construction delay risk | Add 5–10% capex contingency reserve; DFC/IFC project disclosures show ~15% of solar IPPs hit 6+ month delays |

**Note on FX:** if you want to model FX risk explicitly, switch Solar-2 to local-currency PPA and raise `price_vol` to 0.20–0.25. The current `ProjectInputs.fx_vol` field is already there but is folded into a single multiplier — a future improvement is to separate price and FX shocks.

---

## Archetype 3 — Smallholder Agroforestry / Outgrower Scheme

**Vehicle thesis:** finance an aggregator that supplies inputs to and offtakes from 2,000+ smallholders growing perennial cash crops (coffee, cocoa, cashew).

**Why this archetype:** combines weather/yield risk, soft-commodity price risk, and long time-to-first-revenue. Stresses both the yield-shock multiplier and the multi-year capex / J-curve handling.

### Vehicle-level parameters

| Parameter | Value | Rationale |
|---|---|---|
| `total_capital` | $4M | AgDevCo, Root Capital, Acumen ticket size for African agribusiness |
| `guarantee_coverage` | 0.25 | USAID DCA / AGRA guarantee programs typically wrap 25–50% |
| `grant_reserve` | 8% ($320k) | Higher reserve — extension services, technical assistance baked in |
| `senior_coupon` | 0.07 | Impact-aligned senior |
| `mezzanine_fraction` | 0.15 | First-loss / patient mezz |
| `investor_hurdle_irr` (calibrator) | 0.06 | Patient impact capital — lower hurdle |
| `max_loss_probability` | 0.12 | Highest tolerance — historical default rates in smallholder agri 5–15% |

### Per-project parameters (2 projects)

| Project | Crop / model | `lifetime_years` | Construction yrs | `capex` | Year-N operating revenue | `price_vol` | `yield_vol` | Notes |
|---|---|---|---|---|---|---|---|---|
| Agro-1 | Robusta coffee, 1,500 outgrowers | 20 | 4 | $1.6M (yrs 0–3 nursery + processing) | $1.2M (yr 6+, 1.0 t/ha × 1,500 ha × $2.5/kg × ~30% aggregator margin) | 0.28 | 0.20 | Coffee NY-C robusta vol; weather/leaf-rust yield risk |
| Agro-2 | Cocoa, 800 outgrowers | 18 | 5 | $1.2M (yrs 0–4) | $0.9M (yr 7+) | 0.25 | 0.22 | Cocoa London terminal vol; swollen-shoot virus / Ghana-CI policy risk |

**Construction period:** 4–7 years (perennial tree crops — coffee first cherries year 3–4, peak year 6–8; cocoa similar).
**Operating period:** 15–20 years productive lifetime.

### Source anchors

| Parameter | Source |
|---|---|
| Smallholder coffee/cocoa yield 0.4–1.5 t/ha | FAO STAT; ICO (International Coffee Org) productivity data |
| Coffee farmgate price $1.5–4/kg robusta, $3–6/kg arabica | ICO monthly composite indicator 2020–2024 |
| Cocoa farmgate $1.5–3/kg | ICCO daily prices; Ghana/Côte d'Ivoire cocoa boards |
| `price_vol` 25–35% (arabica), 20–30% (cocoa) | NY-C and London terminal price series 2015–2024 log-return vol |
| `yield_vol` 15–25% | FAO STAT historical yield CV by country; ENSO impact studies |
| Aggregator economics (margin ~25–35%, capex profile) | AgDevCo annual reports; Root Capital portfolio data; FMO impact reports |
| Default rates 5–15% | AGRA / USAID DCA guarantee program loss histories |

---

## Combined portfolio characteristics — what you'd expect

If we run the calibrator on these three vehicles together, the *qualitative* prediction (subject to MC noise):

| Vehicle | Expected α range | Expected leverage | What drives it |
|---|---|---|---|
| ARR forestry | 35–50% | 1.0–1.9× | High `price_vol` and long time-to-revenue dominate |
| Solar IPP | 10–20% | 4–9× | Guarantee wrap + contracted revenue → low senior risk |
| Smallholder agroforestry | 30–45% | 1.2–2.3× | Both yield and price vol bite; longer J-curve |
| **Portfolio (CVaR-constrained LP)** | weighted ~25–35% | ~2–3× | LP will likely overweight Solar IPP up to its allocation cap |

If the **calibrated numbers come out very different**, that's diagnostic — either the vehicle params are unrealistic, or the calibration constraints are too loose/tight. Either way you have something concrete to discuss with reviewers.

---

## Implementation plan (after your review)

When you've approved or adjusted the parameter table:

1. Generate 3 × 8 (≈24) Format-2 CSV files under `examples/ui_sample/{1_forestry,2_solar,3_agroforestry}/`
   - Each project: 30 rows (years 0–29), columns per Format-2 spec
2. Update `run_e2e.py` `_BUILT_IN_NAMES` and the inline portfolio definition to point at the new archetypes (or load from the CSVs)
3. Update README "Built-in sample data" section + `app.py` Setup page samples
4. Re-run `/calibrate-smoke` — confirm α / leverage falls in the expected ranges above
5. Add a `tests/test_archetypes.py` that loads each CSV and asserts α is within a defensible band

Time estimate: 2–3 hours once parameters are locked.

---

## What I need from you before generating CSVs

1. **Confirm or adjust** the `total_capital`, `guarantee_coverage`, and IRR hurdle values per vehicle — these reflect a generic blended-finance fund, but if you have a specific LP/donor profile in mind (foundation vs DFI vs sovereign-backed), the catalytic structure changes
2. **Geography preference** — Sub-Saharan Africa across all three? Or mix in LatAm / Southeast Asia? (Affects price benchmarks and yield distributions)
3. **Carbon price assumption** — keep at $12–22 (current VCM) or stress with a $5 collapse / $40 rebound scenario as a sensitivity?
4. Any of your real project sheets where the headline numbers (capex, lifetime, expected revenue) are public and I should anchor a project on, even if I don't use the underlying P&L verbatim?

Once those four are answered I can have the CSVs and updated `run_e2e.py` on the branch in one commit.
