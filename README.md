# hydrogen-TEA-power_eng

An open-source Streamlit application for the techno-economic analysis (TEA) of
hydrogen production cost, implementing the after-tax discounted-cash-flow (DCF) 
methodology. This tool is the reference implementation accompanying the review
article *"Comparative Levelized Cost of Hydrogen Production Pathways: A Critical
Review"* (submitted to *Renewable and Sustainable Energy Reviews*), and computes 
the Levelized Cost of Hydrogen (LCOH) under a harmonized, transparent framework.

It is intended to lower the barrier to consistent LCOH estimation for
non-specialist users performing pre-feasibility analyses, and to support the
eventual adoption of a consensus LCOH standard analogous to the EU Clean Hydrogen
Observatory calculator.

---

## Features

The application accepts technical and economic inputs and renders results in real
time using Plotly:

**Inputs**
- *Technical:* rated hydrogen capacity (kW), project lifetime, stack replacement
  interval, annual operating hours, specific energy demand (kWh/kg-H₂), stack
  degradation rate
- *Economic:* construction CAPEX, equipment CAPEX, annual OPEX, electricity tariff,
  hydrogen sales price, oxygen and waste-heat sales prices, discount rate, inflation,
  corporate tax rate, stack replacement cost (% of CAPEX), depreciation method
  (MACRS-7 or straight-line)

**Outputs**
- LCOH (USD/kg-H₂) cost-component decomposition bar chart
- Cost-component pie chart
- ±40 % tornado sensitivity diagram over six variables (energy efficiency,
  electricity tariff, CAPEX, operating hours, OPEX, discount rate)
- Cumulative cash-flow / payback-period curve
- Financial summary: NPV, B/C ratio, project IRR

---

## Reference case 

The application opens pre-loaded with the fixed reference case used in the
accompanying review — a 1 MW PEM facility:

| Parameter | Default value |
|---|---|
| Rated capacity | 1,000 kW |
| Total CAPEX | 1,800 USD/kW (construction 180 + equipment 1,620) |
| Specific energy demand | 51 kWh/kg-H₂ |
| Electricity price | 50 USD/MWh |
| Annual operating hours | 4,380 h (capacity factor 50 %) |
| Project lifetime | 20 years |
| Fixed O&M | 54,000 USD/yr (3 % of CAPEX) |
| Stack replacement | 40 % of CAPEX, triggered in year 8 (35,040 h) |
| Stack degradation | 1.0 %/yr |
| Real discount rate | 8 % (nominal 8 %, inflation 0 %) |
| Corporate tax rate | 22 % |
| By-product credits | none (oxygen and heat prices default to 0) |

Annual hydrogen output for this case is **85,882 kg/yr**, identical to the
manuscript value. Under the nine institutional conventions surveyed in the
manuscript (Table 2), this same case yields LCOH values of 5.24–6.02 USD/kg
(a 1.15× purely methodological spread), with the after-tax DCF baseline (T1)
at **5.86 USD/kg**.

> **Convention note.** The manuscript T1 value capitalizes the year-8 stack
> replacement and depreciates it on the MACRS-7 schedule (Table 2 construction).
> This application, for transparency of the cash-flow trace, expenses the
> replacement outlay in the year it occurs; with all other inputs identical the
> displayed reference LCOH is therefore slightly higher than the manuscript T1
> value. The exact Table 2 cash-flow constructions, together with the scripts
> that regenerate the manuscript figures, are provided in `scripts/`.

---

## Methodology

The calculation follows the after-tax DCF convention (designated **T1** in
the accompanying review), which serves as the baseline against which alternative
institutional conventions are compared. Depreciation defaults to the MACRS-7
schedule, with straight-line available as an option; the depreciation tax shield
is subtracted from the revenue requirement in the standard manner.

---

## Installation

Requires Python 3.9 or later.

```bash
git clone https://github.com/DongukYang/hydrogen-TEA-power_eng.git
cd hydrogen-TEA-power_eng
pip install -r requirements.txt
```

## Usage

```bash
streamlit run TEA_power-based_eng_v1.py
```

The application will open in your default web browser. Adjust the technical and
economic inputs in the sidebar; all charts and the financial summary update in
real time.

---

## Recommended workflow

The tool is designed to couple with the review methodology in two stages:

1. **Feasibility stage** — use the ±40 % sensitivity output to identify critical
   threshold values for the principal variables and prioritize subsequent data
   collection.
2. **Pre-FEED stage** — repeat the analysis with the higher-fidelity NREL H2A
   spreadsheet model documented in the review for independent verification.

---

## Citation

If you use this tool in your research, please cite (see `CITATION.cff`):

> D. Yang, *hydrogen-TEA-power_eng: An Open-Source Techno-Economic Analysis Tool
> for Hydrogen Production Cost Estimation*, GitHub repository, 2026.
> https://github.com/DongukYang/hydrogen-TEA-power_eng

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for
details.

---

## Author

**Donguk Yang, Ph.D.**
Korea Electric Power Corporation (KEPCO)
