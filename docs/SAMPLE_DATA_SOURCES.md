# Sample data — sources and assumptions

The demo portfolios are meant to be *representative*, not authoritative. This
document records where each figure comes from so you can challenge any of them.

Regenerate the UI CSVs after editing the parameters:

```bash
python scripts/build_sample_data.py
```

The built-in `run_e2e.py` portfolio is written inline in `_built_in_inputs()` and
uses the same economics in parametric form.

> **Read this first.** Most of the sources below could not be fetched directly —
> this environment's egress proxy blocks the domains — so the figures were read
> from search-result summaries of those pages rather than from the pages
> themselves. They are attributed honestly, but they have **not** been verified
> against the primary documents. Before these numbers inform anything that
> matters, check them against the sources directly. Figures marked *derived* are
> our own arithmetic on top of a cited number; figures marked *assumption* are
> judgement calls with no direct source.

---

## 1. Carbon credit prices ($/tCO2e)

| Used | Value | Basis |
|---|---|---|
| ARR | **26** | ARR credits rated BBB or higher averaged ~$26 in 2025 offtakes; lower-rated ARR averaged ~$14. A fund financing new, rated supply prices off the rated tier. |
| Agroforestry | **18–20** | Priced just below rated ARR (*assumption*): same buyer pool, weaker ratings coverage for smallholder aggregation. |
| REDD+ | **6** | REDD+ average as of 2026. Historically as low as $2.70. |
| Biochar | **130** | Low end of the $130–200/tCO2e range; US biochar credits were ~$150/t in Oct 2025 and averaged $164 across 2025. |

For scale, the **market-wide** average across all credit types was $4.04 (2021),
$7.37 (2022), $6.97 (2023) and $6.53 (2024). That average pools old, low-quality
vintages and is not the right price for new rated supply — hence the higher ARR
figure above.

- [Ecosystem Marketplace, State of the Voluntary Carbon Market 2023](https://www.ecosystemmarketplace.com/publications/state-of-the-voluntary-carbon-market-report-2023/) and [2025](https://www.ecosystemmarketplace.com/publications/2025-state-of-the-voluntary-carbon-market-sovcm/)
- [Sylvera — carbon offset pricing](https://www.sylvera.com/blog/carbon-offset-price)
- [Regreener — carbon credit prices by project type](https://www.regreener.earth/blog/carbon-credit-prices-today-trends-and-forecasts-for-2026)
- [CarbonCredits.com — biochar credit prices 2025](https://carboncredits.com/biochar-carbon-credits-in-2025-stable-prices-amid-weakening-demand/)

## 2. Price volatility

Spot volatility is high and well evidenced:

- The market-wide average price series above gives log returns of +0.600,
  −0.056 and −0.065. Sample standard deviation: **0.38** (*derived*).
- REDD+ credits **lost 62% of their value in 2024** on integrity concerns — a
  single-year log return of −0.97.
- CBL N-GEO nature-based futures fell from double digits at their 2022 peak to
  well under $1.
- Biochar prices, by contrast, are repeatedly described as stable.

Spot volatility used: ARR 0.28–0.45, agroforestry 0.40–0.50, REDD+ 0.55,
biochar 0.20.

- [Ecosystem Marketplace 2024 SOVCM](https://www.ecosystemmarketplace.com/publications/2024-state-of-the-voluntary-carbon-markets-sovcm/)
- [The collapse of N-GEO carbon prices](https://carboncredits.com/the-collapse-of-ngeo-carbon-prices-an-in-depth-analysis/)
- [CME Group — CBL N-GEO](https://www.cmegroup.com/markets/energy/emissions/cbl-nature-based-global-emissions-offset.html)

### The offtake adjustment — the most consequential assumption here

`price_vol` in this model is the volatility of project **revenue**, not of the
spot carbon price. Project-financed carbon projects forward-sell a share of
production, and lenders generally require it: **$12.25bn** of offtake deals were
signed in 2025, up from roughly $4bn in 2024.

With a fraction `f` of volume sold forward at a fixed price:

```
revenue volatility = (1 - f) x spot price volatility
```

Coverage assumed (*assumption*): **30%** for nature-based, 25% for REDD+, 70%
for biochar. 30% is deliberately conservative — well below what a project-finance
lender would normally require — reflecting how immature forward contracting still
is for new ARR and agroforestry supply.

| Type | Spot vol | Offtake | Revenue vol |
|---|---|---|---|
| ARR (core / low / high) | 0.35 / 0.28 / 0.45 | 30% | 0.245 / 0.196 / 0.315 |
| Agroforestry (std / volatile) | 0.40 / 0.50 | 30% | 0.280 / 0.350 |
| REDD+ | 0.55 | 25% | 0.413 |
| Biochar | 0.20 | 70% | 0.060 |

**α is highly sensitive to this.** Raising nature-based coverage from 30% to 50%
roughly halves the calibrated catalytic fraction. Re-run with your own coverage
assumption before drawing conclusions.

- [Sylvera — carbon credit demand and offtake volumes](https://www.sylvera.com/blog/carbon-credit-demand)

## 3. Sequestration rates (tCO2e/ha/yr)

| Used | Value | Basis |
|---|---|---|
| ARR | **12–14** | Restoration under recognised carbon standards averages a conservative ~12 tCO2e/ha/yr, ~490 tCO2e/ha cumulative over a 41-year project. Planted forests and woodlots remove **4.5–40.7** tCO2e/ha/yr over their first 20 years, so 12–14 sits at the conservative end of the planted range. |
| Agroforestry | **5–9** | African agroforestry: parklands, live fences and homegardens accumulate 0.2–0.8 MgC/ha/yr (0.7–2.9 tCO2e); rotational woodlots 2.2–5.8 MgC/ha/yr (8–21 tCO2e); shaded coffee in Togo and cacao in Cameroon reach ~6 MgC/ha/yr (22 tCO2e). Conversion ×3.667 (*derived*). |
| REDD+ | **1.3** | Avoided-deforestation yields vary enormously with baseline; a low figure is used deliberately (*assumption*). |
| Biochar | **2.8 tCO2e per t biochar** | One tonne of biochar removes ~2.5–3.3 tCO2e. |

Planted projects use a growth **ramp** rather than a flat yield — young trees fix
less carbon — rising from 15% of the plateau rate in year 1 to full rate by
year 8 (*assumption*, shape only; the plateau rate is sourced).

- [Global CO2 removal rates from forest landscape restoration](https://link.springer.com/article/10.1186/s13021-018-0110-8) (*Carbon Balance and Management*)
- [Carbon sequestration potential of agroforestry systems in Africa](https://link.springer.com/chapter/10.1007/978-94-007-1630-8_4)
- [green.earth — translating carbon into hectares](https://www.green.earth/blog/the-real-cost-of-1-tonne-of-co2-translating-carbon-into-hectares)

## 4. Capital and operating costs

| Used | Value | Basis |
|---|---|---|
| ARR capex | **$1,500–2,300/ha** | African restoration spans ~$14–153/ha (farmer-managed natural regeneration), $185/ha (median forest management), $680/ha (Madagascar), $87–1,445/ha (Ethiopia), up to $4,000–6,000/ha (commercial afforestation since 2000). These sit in the managed-planting middle, not plantation. |
| Agroforestry capex | **$500–900/ha** | Seedlings, extension and aggregation; no land acquisition. Below restoration planting (*assumption*, anchored to the low end of the range above). |
| ARR/agro opex | **$28–75/ha/yr** | Management, monitoring and verification. REDD+ administration alone runs $4–15/ha/yr, and MRV $0.21–1.46/tCO2. |
| REDD+ capex | **$700k** | Baseline study, validation and registration. One 200,000 ha Kenyan project cost ~$600k from inception to first issuance against a $150k estimate. |
| Biochar capex | **$2.6M** | A 12 t/day facility. 5–10 TPD plants run $800k–$2M; 20–50 TPD run $1M–$8M. |
| Biochar opex | **$806k/yr** | ~$210/t biochar against a commonly cited ~$230/t production cost. |

Cross-check: afforestation abates carbon at roughly **$16–25/tCO2e** and forest
conservation at **$4–9/tCO2e** in the most cost-efficient countries. The ARR
projects here imply costs inside that band.

- [Trillion Trees — Defining the real cost of restoring forests](https://trilliontrees.org/wp-content/uploads/2022/08/Trillion-Trees_Defining-the-real-cost-of-restoring-forests.pdf)
- [One Earth — the true cost of global land restoration commitments](https://www.oneearth.org/true-cost-of-global-land-restoration-commitments/)
- [OECD — A global analysis of the cost-efficiency of forest carbon](https://one.oecd.org/document/ENV/WKP(2021)17/En/pdf)
- [Transaction costs of six Peruvian REDD+ projects](https://www.ecologyandsociety.org/vol18/iss1/art17/)
- [Cost elements of REDD+ pilot projects in Tanzania](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3441278/)
- [World Economic Forum — cost of carbon removal technologies](https://www.weforum.org/stories/2025/01/cost-of-different-carbon-removal-technologies/)

## 5. Structure and horizon

| Used | Value | Basis |
|---|---|---|
| Operating horizon | **15 years** | VCS ARR projects carry a **20-year** crediting period; a fund vehicle exits before the tail rather than holding to year 20 (*assumption*). |
| Senior coupon | **7.5–8%** | Blended structures use 10–20% subordination to lift the senior tranche to investment grade (BBB), which is consistent with a high-single-digit EM coupon (*derived*). |
| Mezzanine | **10%** of capital | Within the 10–20% subordination band. |
| Guarantee coverage | **25–30%** of senior | Partial credit guarantee, in the range DFIs write. |
| Correlation | **0.35–0.55** rank | Projects share a carbon price and country risk, partly offset by different species, sites and buyers (*assumption*). |
| Total capital | capex **+ ~20%** | Fees, working capital and reserves (*assumption*). |

- [Amundi — demystifying credit enhancements in blended finance](https://research-center.amundi.com/article/how-can-investors-lean-blended-finance-structures-demystifying-credit-enhancements)
- [Verra VM0047 ARR methodology](https://verra.org/methodologies/vm0047-afforestation-reforestation-and-revegetation-v1-1/)

---

## Does the output look right?

Convergence reports an average blended-finance leverage ratio of **4.1x**, with
observed values ranging from **0.30 to 22**. Investment-grade senior tranches
typically sit behind **10–20%** subordination.

What the sample data produces (800 paths, seed 42, 7% hurdle):

| | UI sample (3 vehicles) | Built-in `run_e2e` (2 vehicles) |
|---|---|---|
| α by vehicle | 0.14 / 0.43 / 0.38 | 0.27 / 0.50 |
| Portfolio leverage | ~2.4x | ~1.8x |
| Catalytic share | ~30% | ~36% |
| Portfolio CVaR (95%) | ~8.5% | ~0% |

That sits inside Convergence's observed range but below its 4.1x average, and
the catalytic share is above the 10–20% investment-grade band. Both are
consequences of the conservative 30% offtake assumption and of these being
deliberately marginal projects. Reforestation calibrates far more cheaply than
smallholder agroforestry, which is the qualitative result you would expect.

α moves by a few percentage points across random seeds and simulation counts —
inherent Monte Carlo noise, not instability.

- [Convergence — leverage of concessional capital](https://www.convergence.finance/resource/leverage-of-concessional-capital/view)

## What would make this materially better

1. **Real project data.** One anonymised financial model from an actual ARR or
   agroforestry fund would beat every published average here.
2. **A real carbon price series.** The tool already supports Format 3
   (`price_file`), which estimates drift and volatility from an actual series
   instead of the point estimates above. Feeding it a real ARR or N-GEO price
   history would remove the largest guess in section 2.
3. **Actual offtake terms.** Coverage, tenor and strike from real contracts,
   replacing the flat 30% assumption.
