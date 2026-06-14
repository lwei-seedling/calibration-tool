# Sample Portfolio Archetypes — Draft for Review

**Status:** parameter draft, not yet implemented.
**Purpose:** replace the current synthetic built-in sample (East/West Africa generic) with three sector-grounded archetypes whose calibrated α we can defend to a foundation principal.

**Audience context (drives the design choices below):**
- Upcoming conversations: **FMO, Terratai, OECD, revalue.earth**
- Deliverable target: **4–5 page white paper by late August**
- MVP also needs to function as a **portfolio testing sandbox** for refining the calibration framework against real deals

What each audience will probe — and how the archetypes need to answer:

| Audience | What they'll push on | What the archetypes must show |
|---|---|---|
| **FMO** | Realistic DFI senior pricing, guarantee structures, IRR hurdles by ticket size | Solar IPP archetype with MIGA-style 50% wrap; senior coupon 7%; sovereign offtaker risk |
| **Terratai** | NbS economics, VCM price reality, MRV/buffer costs | ARR archetype with 30–40% price vol from N-GEO, ≥15% buffer pool, 3–5y construction |
| **OECD DAC** | Public-private mobilisation ratio, concessional share by sector | Per-vehicle leverage `(1−α)/α` directly comparable to OECD's "Mobilisation Effect" metric |
| **revalue.earth** | Carbon price modelling assumptions, vol calibration source | GBM with vol calibrated from public price series; documented stress scenarios |

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

## Geography update (Q2 answered: include LatAm and SE Asia)

The single-region SSA sample is too narrow for the white paper's audience. Final geography mix:

| Archetype | Region split | Why |
|---|---|---|
| VCM Forestry / ARR | 1 SSA, 1 LatAm (Andean), 1 SE Asia (Pakistan-adjusted, see anonymisation §) | Captures the three dominant ARR jurisdictions; reviewers will expect to see Pacific Rim and Amazonian projects |
| DFI Solar IPP | 1 SSA, 1 SE Asia (Vietnam/Indonesia archetype) | Adds Asian PPA structures (LCOE benchmarks, FIT-to-auction transition) |
| Smallholder Agroforestry | 1 SSA (cocoa/coffee), 1 LatAm (cocoa/Andean coffee) | Avoids over-indexing on West Africa cocoa |

LatAm carbon-price benchmarks: same global N-GEO series — VCM is globally priced. LatAm-specific yield benchmarks: CIFOR-ICRAF agroforestry productivity database; PROCAFE / Cenicafé (Colombia) for coffee.

SE Asia adjustments: lower labour cost component in ARR opex (~25% below SSA average), higher delivery risk (monsoon delays, land-tenure complexity).

---

## Stress scenario framework (Q3 answered: include stress)

### Answering your question — is "collapse/rebound" the industry norm?

**Short answer: no, not as a framing.** The industry default is parallel ±% shocks on a handful of key drivers, not regime-shift narratives. But the carbon market specifically had a real collapse episode (N-GEO −60% peak-to-trough 2022→2024) that makes a regime-shift scenario *empirically defensible* for ARR archetypes. So we use both:

| Convention | Source | When to use |
|---|---|---|
| **Parallel ±10/20/30% shocks** on revenue, costs, vol | World Bank PSIA sensitivity framework; IFC project finance review templates | Default for every model parameter; cheap and universally legible to DFI reviewers |
| **NGFS climate scenarios** (Net Zero 2050, Delayed Transition, Disorderly, Current Policies) | NGFS Phase IV scenarios (free) | When the audience is climate-aware (FMO, OECD climate-finance teams) — frames the story as "what if policy environment shifts" |
| **Empirical regime-shift** (VCM 2023 collapse, 2008 cocoa shock, 2014 oil-price collapse) | Public price series | When the audience is sector-specialist (Terratai for VCM, ag-focused investors for cocoa) |

### Proposed stress matrix for the tool

Five named scenarios, each implementable as either a parameter override on the base archetypes or a multiplicative shock in `app.py` Sensitivity page:

| Scenario | Carbon price | PPA tariff | Soft-commodity price | Yield | What it tests |
|---|---|---|---|---|---|
| **Base case** | $14 ARR / $22 mangrove | $0.05/kWh | as base | as base | The headline result |
| **VCM collapse** (empirical 2023) | −50% to $7 ARR / $11 mangrove | unchanged | unchanged | unchanged | NbS-only portfolio resilience; α reaction in forestry |
| **VCM rebound to Article 6 prices** | +80% to $25 ARR / $40 mangrove | unchanged | unchanged | unchanged | Upside case; how much α can be reduced if integrity premium materialises |
| **EM macro stress** (USD +15%, rates +200bp) | −10% | −10% local-PPA only | −15% | unchanged | DFI lens — currency and rate shocks on infrastructure |
| **Climate physical risk** (drought/heat) | unchanged | −5% (curtailment) | unchanged | −25% on ag, −15% on ARR | Tests yield risk dominance in agroforestry; integrates Q3 climate signal |

These align with the four sensitivity tests already wired into `app.py` (CLAUDE.md says "Sensitivity" page has A/B/C/D stress tests), so implementation is largely renaming + new parameter overrides rather than new code.

**For the white paper:** the stress matrix becomes Figure 2 (vehicle α and portfolio leverage under each scenario) — much more compelling than a single base-case result.

---

## Anonymisation protocol for your NDA project sheets

You can't share the Drive folder directly with the tool (NDA + Claude infrastructure caches), and I can't pull from Drive. Here's a clean workflow for getting your real-deal data in without leaking specifics.

### The eight-step anonymisation checklist

Apply *before* the data ever touches the repo or a Claude session:

1. **Drop identifying labels.** Replace project names with `ARR-LATAM-1`, `SOLAR-SEA-1`, etc. Strip methodology codes ("VCS VM0047 v2.0 + AFOLU buffer 18%" → "ARR Verra-registered").
2. **Bucket geography to region, not country.** "Pakistan" → "South/SE Asia". "Peru + Colombia" → "Andean LatAm". *Exception:* keep the region granular enough to defend the parameters (a Pakistan project's ops cost reflects regional labour rates).
3. **Round to one or two significant figures.** $4,237,415 capex → $4M. 18,453 tCO₂e → 18,000. 24-year lifetime → 25.
4. **Multiplicative scale per project** by a random factor in [0.7, 1.4]. Apply *the same factor* to capex / opex / revenue within one project (preserves IRR and ratios); use *different factors* across projects (breaks any "I know which deal this is" pattern based on relative sizing).
5. **Shuffle developer/order.** Don't list the 3 same-developer LatAm projects adjacent. Distribute across the doc. Apply different scaling factors to each so they don't read as a single cohort.
6. **Drop the Rubber project** (or include only as a stress case). You flagged it as opportunistic — its parameters would skew the archetype upward. Better to anchor the archetype on the conservative projects and use Rubber as the "what does the LP do under an unrealistically rosy project" sensitivity.
7. **Adjust Pakistan ARR ops costs upward** by ~30–40% before averaging into the "wider Asia" archetype. Pakistan-specific labour rates are below the regional median (Indonesia, Philippines, Cambodia all higher).
8. **Perturb time-series.** Don't reuse the exact year-by-year yield ramp. Add ±5% Gaussian noise per year. Production curves with identical shapes are a fingerprint.

### What I'd actually need from you (in priority order)

Listed in increasing leak risk. You can stop at any level:

| Level | What you share | Where it goes | Anonymisation needed |
|---|---|---|---|
| **L1 — distributions only** | "ARR capex ranged $1.5–$3.2M, lifetime 25–30y, year-5 yield 8–18k tCO₂e" | Chat message | Light (already aggregated) |
| **L2 — per-archetype summary table** | One row per anonymised project: capex, lifetime, yield curve shape, price assumption | Chat or a comment block in the doc | Moderate (apply steps 1–4) |
| **L3 — full anonymised CSVs** | Format-2 CSVs in `examples/ui_sample/private/` (gitignored) | Local working tree only; *never committed* | Full (all 8 steps) |
| **L4 — committed anonymised CSVs** | Format-2 CSVs in `examples/ui_sample/` replacing the synthetic ones | Public branch | Full + a second-pair-of-eyes review before push |

For the white paper, **L2 is probably the sweet spot** — enough to anchor the archetype parameters and triangulate against public benchmarks, low enough leak risk to share via Claude.

### Pro tip on "anonymised vs synthetic" framing for the white paper

Don't present the archetypes as "based on real deals" *or* as "synthetic." The most defensible framing is:

> "Archetype parameters are drawn from public benchmarks (Verra, IRENA, Xpansiv, ICO) and cross-validated against an anonymised sample of N (N=8) real project financial models under NDA. Reported values fall within ±X% of the anonymised sample median."

This gives you the credibility of real data without disclosing it, and it gives reviewers a concrete claim to push on. Anyone who tries to back out specific projects from the parameters will find ranges, not point estimates.

---

## Implementation plan (after your data extraction)

When you've extracted anonymised numbers at L2 (or higher) and shared them:

1. I'll merge your numbers with the public-source ranges already in this doc and produce a final parameter table (one commit, doc-only)
2. Generate ~20 Format-2 CSV files under `examples/ui_sample/{forestry_arr,solar_ipp,agroforestry}/` — three vehicles, 2–3 projects each
3. Update `run_e2e.py` to load these as the built-in fixture (replaces the hardcoded East/West Africa data)
4. Add `tests/test_archetypes.py` — asserts calibrated α per vehicle falls within the expected band; asserts portfolio LP allocates differently under each stress scenario
5. Update README "Built-in sample data" section + `app.py` Setup page samples to point at the new fixtures
6. Re-run `/calibrate-smoke` — confirm α / leverage matches the predicted ranges
7. Write a one-page "method note" in `docs/` describing how parameters were calibrated (suitable as a white-paper appendix)

Time: 3–4 hours once the data is in. The white-paper method note is an additional 2–3 hours but should happen in the same workstream so the citations stay synchronised.

---

## What I need from you next

A single decision: **which leak level do you want to share at (L1 / L2 / L3)?**

- L1 — paste a quick text table of distributions in chat ("ARR capex $1.5–3.2M, etc."). I'll use this to refine the public-source ranges and proceed.
- L2 — paste per-project anonymised rows (apply anonymisation steps 1–4) in chat or as a markdown comment block. Higher fidelity, still safe to share via chat.
- L3 — anonymise CSVs locally, drop into `examples/ui_sample/private/` (gitignored), tell me to read from there. Highest fidelity, never leaves your machine into a public surface.

If you'd rather skip the real-data anchoring for now and ship a pure-public-source version (lower fidelity but zero NDA risk), say so and I'll proceed with the parameters already in this doc.
