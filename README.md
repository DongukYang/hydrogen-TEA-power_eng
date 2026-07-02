# hydrogen-TEA-power_eng

An open-source Streamlit application for the techno-economic analysis (TEA) of
hydrogen production cost, implementing the NREL H2A after-tax discounted-cash-flow
(DCF) methodology. This tool is the reference implementation accompanying the review
article *"Comparative Levelized Cost of Hydrogen Production Pathways: A Review"*
(D. Yang and B. Oh), and computes the Levelized Cost of Hydrogen (LCOH) under a
harmonized, transparent framework.

It is intended to lower the barrier to consistent LCOH estimation for non-specialists
users performing pre-feasibility analyses, and to support the eventual adoption of a
consensus LCOH standard analogous to the EU Clean Hydrogen Observatory calculator.

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
  corporate tax rate

**Outputs**
- LCOH (USD/kg-H₂) cost-component decomposition bar chart
- Cost-component pie chart
- ±40 % tornado sensitivity diagram over six variables (energy efficiency,
  electricity tariff, CAPEX, operating hours, OPEX, discount rate)
- Cumulative cash-flow / payback-period curve
- Financial summary: NPV, B/C ratio, project IRR

---

## Methodology

The calculation follows the after-tax DCF convention (designated **T1** in
the accompanying review), which serves as the baseline against which alternative
institutional conventions are compared.

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

If you use this tool in your research, please cite:

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
