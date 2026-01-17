# app.py 

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import numpy_financial as nf

# ---------------- Basic Settings ----------------
st.set_page_config(page_title="Hydrogen Production TEA Results", layout="wide")
st.title("Hydrogen Production TEA Results [Ver.1]")

## ---------------- 1. Input Section (Sidebar) ----------------
st.sidebar.header("Input Parameters")


# 1-1. Technical Inputs
st.sidebar.subheader("Technical")

h2_capacity_kw = st.sidebar.number_input("Hydrogen Production Capacity (kW)", min_value=0.0, value=5000.0, step=100.0)
# construction_years = st.sidebar.number_input("Construction Period (years)", min_value=0.0, value=2.0, step=0.5)
operation_years = st.sidebar.number_input("Operation Period (years)", min_value=1.0, value=20.0, step=1.0)
stack_replacement_hours = st.sidebar.number_input("Stack Replacement Cycle (hours)", min_value=0.0, value=60000.0, step=1000.0)
annual_operating_hours = st.sidebar.number_input("Annual Operating Hours (hours/year)", min_value=0.0, max_value=8760.0, value=8000.0, step=100.0)
specific_energy = st.sidebar.number_input("Energy Efficiency (kWh/kgH₂)", min_value=1.0, value=55.5, step=0.1)
stack_degradation = st.sidebar.number_input("Stack Degradation Rate (%/year)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)


# 1-2. Economic Inputs
st.sidebar.subheader("Economic")

capex_construction = st.sidebar.number_input("CAPEX - Construction (USD)", min_value=0.0, value=77_000.0 * (h2_capacity_kw / 500.0), step=1_000.0)
capex_equipment = st.sidebar.number_input("CAPEX - Equipment (USD)", min_value=0.0, value=710_000.0 * (h2_capacity_kw / 500.0), step=1_000.0)
capex_total = capex_construction + capex_equipment


opex_annual = st.sidebar.number_input("OPEX (annual, USD/year)", min_value=0.0, value=2_300_000.0, step=1_000.0)
elec_price = st.sidebar.number_input("Electricity Fee (USD/kWh)", min_value=0.0, value=0.04, step=0.01)
h2_price = st.sidebar.number_input("Hydrogen Selling Price (USD/kgH₂)", min_value=0.0, value=7.7, step=0.1)
o2_price = st.sidebar.number_input("Oxygen Selling Price (USD/kgO₂) : TBD", min_value=0.0, value=0.04, step=0.01)
heat_price = st.sidebar.number_input("Heat Selling Price (USD/MWh) : TBD", min_value=0.0, value=0.0, step=1.0)
discount_rate = st.sidebar.number_input("Discount Rate (%/year)", min_value=0.0, max_value=20.0, value=7.0, step=0.1)
inflation_rate = st.sidebar.number_input("Inflation Rate (%/year)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
# cost_of_capital = st.sidebar.number_input("Cost of Capital (%/year)", min_value=0.0, max_value=20.0, value=8.0, step=0.1)
corp_tax_rate = st.sidebar.number_input("Corporate Tax Rate (%)", min_value=0.0, max_value=50.0, value=20.0, step=0.1)
tax_rate = corp_tax_rate / 100.0

# ---------------- 2. LCOH · Financial Model ----------------

# Annual hydrogen production (kgH2/year): P(kW) * hours(h) / (kWh/kgH2)
annual_h2_kg = 0.0
if specific_energy > 0:
    annual_h2_kg = (h2_capacity_kw * annual_operating_hours) / specific_energy

# ===== Reflect stack degradation =====
degradation_rate = stack_degradation / 100.0  
stack_factor_year = [(1 - degradation_rate) ** t for t in range(int(operation_years))]

# Annual electricity consumption and cost
annual_elec_kwh = h2_capacity_kw * annual_operating_hours
annual_elec_cost = annual_elec_kwh * elec_price

# Byproduct oxygen and heat revenue
annual_o2_kg = annual_h2_kg * 8.0  
annual_o2_revenue = annual_o2_kg * o2_price
heat_mwh_per_kg = 0.0  # If needed, revised the real-data
annual_heat_mwh = annual_h2_kg * heat_mwh_per_kg
annual_heat_revenue = annual_heat_mwh * heat_price

# Real discount rate using Fisher formula
r_nom = discount_rate / 100.0
inf = inflation_rate / 100.0
if (1 + inf) > 0:
    r_real = (1 + r_nom) / (1 + inf) - 1
else:
    r_real = r_nom

# Cash flow generation for NPV/BEP
cash_flows = [-capex_total]  
pv_in = 0.0
pv_out = capex_total  

# Initialize degradation tracking
stack_deg_rate = stack_degradation / 100.0
hours_since_last_replacement = 0.0
years_since_last_replacement = 0

for t in range(1, int(operation_years) + 1):
    opex_stack_t = 0.0
    if stack_replacement_hours > 0 and hours_since_last_replacement >= stack_replacement_hours:
        opex_stack_t = opex_annual * 0.3  
        hours_since_last_replacement = 0.0
        years_since_last_replacement = 0

    degradation_multiplier = (1 - stack_deg_rate) ** years_since_last_replacement if stack_deg_rate > 0 else 1.0
    h2_t = annual_h2_kg * degradation_multiplier
    elec_kwh_t = annual_elec_kwh * degradation_multiplier

    # Annual Cost & revenue
    yearly_cost = elec_kwh_t * elec_price + opex_annual + opex_stack_t  
    yearly_rev = (h2_t * h2_price) + (h2_t * 8.0 * o2_price) + (h2_t * heat_mwh_per_kg * heat_price)

    # net present value
    net_cf = yearly_rev - yearly_cost
    cash_flows.append(net_cf)
    if r_real > -0.9999:
        pv_in += yearly_rev / ((1 + r_real) ** t)
        pv_out += yearly_cost / ((1 + r_real) ** t)

    hours_since_last_replacement += annual_operating_hours
    years_since_last_replacement += 1

npv = pv_in - pv_out
bc_ratio = pv_in / pv_out if pv_out > 0 else np.nan

try:
    irr = nf.irr(cash_flows)
    p_irr_pct = irr * 100 if irr is not None else np.nan
except Exception:
    p_irr_pct = np.nan

# LCOH calcuation
depreciation_amount = capex_total / operation_years if operation_years > 0 else 0.0
numerator_sum_after_tax = capex_total
denominator_sum = 0.0
# Total NPV during the period
hours_since_last_replacement = 0.0
years_since_last_replacement = 0
for t in range(1, int(operation_years) + 1):
    opex_stack_t = 0.0
    if stack_replacement_hours > 0 and hours_since_last_replacement >= stack_replacement_hours:
        opex_stack_t = opex_annual * 0.3
        hours_since_last_replacement = 0.0
        years_since_last_replacement = 0
    degradation_multiplier = (1 - stack_deg_rate) ** years_since_last_replacement if stack_deg_rate > 0 else 1.0
    h2_t = annual_h2_kg * degradation_multiplier
    elec_kwh_t = annual_elec_kwh * degradation_multiplier
    cost_t = elec_kwh_t * elec_price + opex_annual + opex_stack_t
    o2_rev_t = h2_t * 8.0 * o2_price
    heat_rev_t = h2_t * heat_mwh_per_kg * heat_price
    numerator_sum_after_tax += ((cost_t - o2_rev_t - heat_rev_t) * (1 - tax_rate) + tax_rate * depreciation_amount) / ((1 + r_real) ** t)
    denominator_sum += h2_t / ((1 + r_real) ** t)
    hours_since_last_replacement += annual_operating_hours
    years_since_last_replacement += 1

if denominator_sum > 0:
    lcoh_krw_per_kg = numerator_sum_after_tax / denominator_sum / (1 - tax_rate)
else:
    lcoh_krw_per_kg = 0.0

# LCOH calculation function for sensitivity analysis
def compute_lcoh_given_params(capex_total_val, specific_energy_val, elec_price_val, annual_operating_hours_val, opex_annual_val, discount_rate_val):
    # Actual discoiunt rate calculation (Inflation rate, inf fixed)
    r_nominal = discount_rate_val / 100.0
    r_real_val = (1 + r_nominal) / (1 + inf) - 1 if (1 + inf) > 0 else r_nominal
    # Scenario of annual hydrogen production
    annual_h2_val = 0.0
    if specific_energy_val > 0:
        annual_h2_val = (h2_capacity_kw * annual_operating_hours_val) / specific_energy_val
    annual_elec_kwh_val = h2_capacity_kw * annual_operating_hours_val
    # Initialize parameters
    hours_since_last_replacement_val = 0.0
    years_since_last_replacement_val = 0
    numerator_val = capex_total_val
    denominator_val = 0.0
    # Depreciation (Straight-line Method)
    depreciation_val = capex_total_val / operation_years if operation_years > 0 else 0.0
    for t in range(1, int(operation_years) + 1):
        opex_stack_repl = 0.0
        if stack_replacement_hours > 0 and hours_since_last_replacement_val >= stack_replacement_hours:
            opex_stack_repl = opex_annual_val * 0.3
            hours_since_last_replacement_val = 0.0
            years_since_last_replacement_val = 0
        deg_multiplier = (1 - stack_degradation / 100.0) ** years_since_last_replacement_val if stack_degradation > 0 else 1.0
        h2_t_val = annual_h2_val * deg_multiplier
        elec_kwh_t_val = annual_elec_kwh_val * deg_multiplier
        cost_t_val = elec_kwh_t_val * elec_price_val + opex_annual_val + opex_stack_repl
        o2_rev_t_val = h2_t_val * 8.0 * o2_price
        heat_rev_t_val = h2_t_val * heat_mwh_per_kg * heat_price
        numerator_val += ((cost_t_val - o2_rev_t_val - heat_rev_t_val) * (1 - tax_rate) + tax_rate * depreciation_val) / ((1 + r_real_val) ** t)
        denominator_val += h2_t_val / ((1 + r_real_val) ** t)
        hours_since_last_replacement_val += annual_operating_hours_val
        years_since_last_replacement_val += 1
    if denominator_val > 0:
        return numerator_val / denominator_val / (1 - tax_rate)
    else:
        return 0.0

# --- LCOH component data (USD/kgH2 unit) ---
labels = ["CAPEX-Construction", "CAPEX-Facility", "OPEX-O&M", "OPEX-Electricity", "Total LCOH"]

# Initial values
numerator_construction = capex_construction   
numerator_equipment   = capex_equipment       
numerator_opex        = 0.0                  
numerator_elec        = 0.0                  
numerator_byprod      = 0.0                  

# Discounted revenue/cost
hours_since_last_replacement_val = 0.0
years_since_last_replacement_val = 0
for t in range(1, int(operation_years) + 1):
    # Stack replacement
    opex_stack_t = 0.0
    if stack_replacement_hours > 0 and hours_since_last_replacement_val >= stack_replacement_hours:
        opex_stack_t = opex_annual * 0.3 
        hours_since_last_replacement_val = 0.0
        years_since_last_replacement_val = 0

    # Stack efficiency
    deg_multiplier = (1 - stack_deg_rate) ** years_since_last_replacement_val if stack_deg_rate > 0 else 1.0

    # Hydrogen production, electricity usage
    h2_t = annual_h2_kg * deg_multiplier
    elec_kwh_t = annual_elec_kwh * deg_multiplier

    # Annual cost revenue
    elec_cost_t = elec_kwh_t * elec_price               # Electricity cost
    opex_cost_t = opex_annual + opex_stack_t            # O&M cost (fixed OPEX + replacement cost)
    o2_rev_t   = h2_t * 8.0 * o2_price                  # Oxygen revenue (8 kg O₂ per 1 kg H₂)
    heat_rev_t = h2_t * heat_mwh_per_kg * heat_price    # Heat revenue

    # Discount rate
    discount_factor = 1 / ((1 + r_real) ** t) if r_real != -1 else 1.0 

    # Accumulate present value of each component (after-tax)
    numerator_elec += elec_cost_t * (1 - tax_rate) * discount_factor        # After-tax electricity cost
    numerator_opex += opex_cost_t * (1 - tax_rate) * discount_factor        # AFter-tax O&M
    numerator_byprod += - (o2_rev_t + heat_rev_t) * (1 - tax_rate) * discount_factor  # Revenue

    numerator_construction += tax_rate * (depreciation_amount * (capex_construction / capex_total)) * discount_factor
    numerator_equipment   += tax_rate * (depreciation_amount * (capex_equipment   / capex_total)) * discount_factor

    hours_since_last_replacement_val += annual_operating_hours
    years_since_last_replacement_val += 1

capex_construction_per_kg = numerator_construction / denominator_sum / (1 - tax_rate) if denominator_sum > 0 else 0.0
capex_equipment_per_kg   = numerator_equipment   / denominator_sum / (1 - tax_rate) if denominator_sum > 0 else 0.0
opex_per_kg              = (numerator_opex + numerator_byprod) / denominator_sum / (1 - tax_rate) if denominator_sum > 0 else 0.0
elec_per_kg              = numerator_elec        / denominator_sum / (1 - tax_rate) if denominator_sum > 0 else 0.0

# total LCOH
total_lcoh_calc = capex_construction_per_kg + capex_equipment_per_kg + opex_per_kg + elec_per_kg

# Piechart/Cost (Excluding Total LCOH)
drivers = ["CAPEX-Construction", "CAPEX-Facility", "OPEX-O&M", "OPEX-Electricity"]
values  = [capex_construction_per_kg, capex_equipment_per_kg, opex_per_kg, elec_per_kg]

# Colors
colors = {
    "CAPEX-Construction": "#1f77b4",
    "CAPEX-Facility": "#aec7e8",
    "OPEX-O&M":  "#d62728",
    "OPEX-Electricity": "#ff9896",
}

# Bar Chart
y_capex_constr = [capex_construction_per_kg, 0, 0, 0, capex_construction_per_kg]
y_capex_equip  = [0, capex_equipment_per_kg, 0, 0, capex_equipment_per_kg]
y_opex         = [0, 0, opex_per_kg, 0, opex_per_kg]
y_elec         = [0, 0, 0, elec_per_kg, elec_per_kg]

fig_bar = go.Figure()
fig_bar.add_bar(name="CAPEX-Construction", x=labels, y=y_capex_constr,
               text=[f"{v:,.0f}" if v > 0 else "" for v in y_capex_constr], textposition="auto", marker_color=colors["CAPEX-Construction"])
fig_bar.add_bar(name="CAPEX-Facility", x=labels, y=y_capex_equip,
               text=[f"{v:,.0f}" if v > 0 else "" for v in y_capex_equip], textposition="auto", marker_color=colors["CAPEX-Facility"])
fig_bar.add_bar(name="OPEX-O&M", x=labels, y=y_opex,
               text=[f"{v:,.0f}" if v > 0 else "" for v in y_opex], textposition="auto", marker_color=colors["OPEX-O&M"])
fig_bar.add_bar(name="OPEX-Electricity", x=labels, y=y_elec,
               text=[f"{v:,.0f}" if v > 0 else "" for v in y_elec], textposition="auto", marker_color=colors["OPEX-Electricity"])

fig_bar.update_layout(barmode="stack", yaxis_title="USD/kgH₂", xaxis_title="Cost Item")

# ---------------- 3. Layout ----------------
col_lcoh, col_fin = st.columns([1.4, 1])

# 3-1. Left: LCOH analysis
with col_lcoh:
    st.subheader("LCOH Analysis")

    annual_elec_mwh = annual_elec_kwh / 1000.0
    annual_h2_ton = annual_h2_kg / 1000.0

    table_md = f"""
    <table style='font-size:18px; font-weight:500;'>
    <tr>
        <td style='padding: 6px 15px;'>LCOH (USD/kgH₂)</td>
        <td style='padding: 6px 15px; font-weight:bold;'>{lcoh_krw_per_kg:,.2f}</td>
    </tr>
    <tr>
        <td style='padding: 6px 15px;'>Annual Electricity Consumption  (MWh/year)</td>
        <td style='padding: 6px 15px; font-weight:bold;'>{annual_elec_mwh:,.0f}</td>
    </tr>
    <tr>
        <td style='padding: 6px 15px;'>Annual Hydrogen Production (Ton H₂/year)</td>
        <td style='padding: 6px 15px; font-weight:bold;'>{annual_h2_ton:,.0f}</td>
    </tr>
    </table>
    """
    st.markdown(table_md, unsafe_allow_html=True)

    # Upper: LCOH cost piechart + sensitivity analysis
    col_pie, col_sensi = st.columns(2)

    with col_pie:
        st.markdown("### LCOH Component")
        total_for_share = sum(values)
        if total_for_share > 0:
            fig_pie = px.pie(names=drivers, values=values, color=drivers, color_discrete_map=colors)
            fig_pie.update_layout(font=dict(family="Arial Black", size=18), legend=dict(font=dict(size=16)))
            fig_pie.update_traces(textfont=dict(size=16))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Please enter valid input values to calculate the LCOH.")

    with col_sensi:
        st.markdown("### Sensitivity Analysis (Variation ±40%)")
        sensi_vars = ["Energy Efficiency", "Electricity Fee", "CAPEX", "Annual Operating Hours", "OPEX", "Discount Rate"]
        base_params = {
            "specific_energy": specific_energy,
            "elec_price": elec_price,
            "capex_total": capex_total,
            "annual_operating_hours": annual_operating_hours,
            "opex_annual": opex_annual,
            "discount_rate": discount_rate,
        }
        delta_minus, delta_plus = [], []

        for name in sensi_vars:
            # --- -40% Case ---
            p = base_params.copy()
            if name == "Energy Efficiency":
                p["specific_energy"] *= 0.6
            elif name == "Electricity Fee":
                p["elec_price"] *= 0.6
            elif name == "CAPEX":
                p["capex_total"] *= 0.6
            elif name == "Annual Operating Hours":
                p["annual_operating_hours"] *= 0.6
            elif name == "OPEX":
                p["opex_annual"] *= 0.6
            elif name == "Discount Rate":
                p["discount_rate"] *= 0.6

            lcoh_m = compute_lcoh_given_params(p["capex_total"], p["specific_energy"], p["elec_price"], p["annual_operating_hours"], p["opex_annual"], p["discount_rate"])
            delta_minus.append(lcoh_m - lcoh_krw_per_kg)

            # --- +40% Case ---
            p = base_params.copy()
            if name == "Energy Efficiency":
                p["specific_energy"] *= 1.4
            elif name == "Electricity Fee":
                p["elec_price"] *= 1.4
            elif name == "CAPEX":
                p["capex_total"] *= 1.4
            elif name == "Annual Operating Hours":
                p["annual_operating_hours"] *= 1.4
            elif name == "OPEX":
                p["opex_annual"] *= 1.4
            elif name == "Discount Rate":
                p["discount_rate"] *= 1.4

            lcoh_p = compute_lcoh_given_params(p["capex_total"], p["specific_energy"], p["elec_price"], p["annual_operating_hours"], p["opex_annual"], p["discount_rate"])
            delta_plus.append(lcoh_p - lcoh_krw_per_kg)

        max_change = max(max(abs(np.array(delta_minus))), max(abs(np.array(delta_plus))))

        fig_sensi = go.Figure()
        fig_sensi.add_bar(y=sensi_vars, x=delta_minus, orientation="h", name="-40%")
        fig_sensi.add_bar(y=sensi_vars, x=delta_plus, orientation="h", name="+40%")
        fig_sensi.update_layout(xaxis_title="ΔLCOH (USD/kgH₂)", barmode="relative",
                                 font=dict(family="Arial Black", size=18), legend=dict(font=dict(size=16)))
        fig_sensi.update_xaxes(range=[-max_change * 1.1, max_change * 1.1], zeroline=True, tickfont=dict(size=16))
        fig_sensi.update_yaxes(tickfont=dict(size=16))
        st.plotly_chart(fig_sensi, use_container_width=True)
    
    st.markdown("### LCOH Component")
    capex_construction_per_kg = capex_construction_per_kg
    capex_equipment_per_kg = capex_equipment_per_kg
    opex_per_kg = opex_per_kg
    elec_per_kg = elec_per_kg

    # Cumulative calculation
    c1 = capex_construction_per_kg
    c2 = c1 + capex_equipment_per_kg
    c3 = c2 + opex_per_kg
    c4 = c3 + elec_per_kg

    x_labels = ["Total LCOH", "CAPEX-Construction", "CAPEX-Facility", "OPEX-O&M", "OPEX-Electricity"]
    fig_major = go.Figure()
    fig_major.add_bar(name="CAPEX-Construction", x=["Total LCOH"], y=[capex_construction_per_kg], base=[0], marker_color=colors["CAPEX-Construction"])
    fig_major.add_bar(name="CAPEX-Facility", x=["Total LCOH"], y=[capex_equipment_per_kg], base=[c1], marker_color=colors["CAPEX-Facility"])
    fig_major.add_bar(name="OPEX-O&M", x=["Total LCOH"], y=[opex_per_kg], base=[c2], marker_color=colors["OPEX-O&M"])
    fig_major.add_bar(name="OPEX-Electricity", x=["Total LCOH"], y=[elec_per_kg], base=[c3], marker_color=colors["OPEX-Electricity"])
    # (Each Bar)
    fig_major.add_bar(name="CAPEX-Construction (Each)", x=["CAPEX-Construction"], y=[capex_construction_per_kg], base=[0], marker_color=colors["CAPEX-Construction"], showlegend=False)
    fig_major.add_bar(name="CAPEX-Facility (Each)", x=["CAPEX-Facility"], y=[capex_equipment_per_kg], base=[c1], marker_color=colors["CAPEX-Facility"], showlegend=False)
    fig_major.add_bar(name="OPEX-O&M (Each)", x=["OPEX-O&M"], y=[opex_per_kg], base=[c2], marker_color=colors["OPEX-O&M"], showlegend=False)
    fig_major.add_bar(name="OPEX-Electricity (Each)", x=["OPEX-Electricity"], y=[elec_per_kg], base=[c3], marker_color=colors["OPEX-Electricity"], showlegend=False)

    fig_major.update_layout(barmode="overlay", yaxis_title="USD/kgH₂", xaxis_title="", 
                             font=dict(family="Arial Black", size=18), legend=dict(font=dict(size=18)))
    fig_major.update_xaxes(categoryorder="array", categoryarray=x_labels, tickfont=dict(size=18))
    fig_major.update_yaxes(tickfont=dict(size=18))
    st.plotly_chart(fig_major, use_container_width=True)

# 3-2. Economical Analysis
with col_fin:
    st.subheader("Economical Analysis")
    npv_million = npv / 1_000_000
    bc_str = f"{bc_ratio:,.2f}" if not np.isnan(bc_ratio) else "N/A"
    pirr_str = f"{p_irr_pct:,.2f}" if not np.isnan(p_irr_pct) else "N/A"
    fin_table = f"""
    <table style='font-size:18px; font-weight:500;'>
        <tr><td style='padding: 6px 15px;'>NPV (Million USD)</td><td style='padding: 6px 15px; font-weight:bold;'>{npv_million:,.2f}</td></tr>
        <tr><td style='padding: 6px 15px;'>B/C Ratio (-)</td><td style='padding: 6px 15px; font-weight:bold;'>{bc_str}</td></tr>
        <tr><td style='padding: 6px 15px;'>P-IRR (%)</td><td style='padding: 6px 15px; font-weight:bold;'>{pirr_str}</td></tr>
    </table>
    """
    st.markdown(fin_table, unsafe_allow_html=True)
    st.markdown("### Present Value of Inflows/Outflows and NPV")
    fin_labels = ["Present Value of Inflows", "Present Value of Outflows", "Net Present Value (NPV)"]
    # npv_display = abs(npv)
    fin_values = [pv_in, pv_out, npv]
    fig_fin = go.Figure(data=[go.Bar(x=fin_labels, y=fin_values, text=[f"{v:,.0f}" for v in fin_values], textposition="auto")])
    fig_fin.update_layout(yaxis_title="USD", font=dict(family="Arial Black", size=18), legend=dict(font=dict(size=16)))
    fig_fin.update_yaxes(range=[0, max(pv_in, pv_out) * 1.2])
    fig_fin.update_xaxes(tickfont=dict(size=16))
    fig_fin.update_yaxes(tickfont=dict(size=16))
    st.plotly_chart(fig_fin, use_container_width=True)

    # Payback period
    years = list(range(len(cash_flows)))
    cum_cf = np.cumsum(cash_flows)
    payback_year = None
    for i in range(1, len(years)):
        if cum_cf[i] >= 0:
            prev_cf = cum_cf[i-1]
            this_cf = cum_cf[i]
            if this_cf == prev_cf:
                payback_year = years[i]
            else:
                frac = -prev_cf / (this_cf - prev_cf)
                payback_year = years[i-1] + frac
            break

    st.markdown("### Payback Period")
    fig_pay = go.Figure()
    fig_pay.add_trace(go.Scatter(x=years, y=cum_cf, mode="lines+markers", name="Cumulative Cash Flow"))
    fig_pay.add_shape(type="line", x0=years[0], x1=years[-1], y0=0, y1=0, line=dict(color="gray", dash="dash"))
    if payback_year is not None:
        fig_pay.add_shape(type="line", x0=payback_year, x1=payback_year, y0=min(cum_cf), y1=max(cum_cf), line=dict(color="red", dash="dot"))
        fig_pay.add_trace(go.Scatter(x=[payback_year], y=[0], mode="markers", marker=dict(size=10, color="red"), name="Payback point"))
    fig_pay.update_layout(xaxis_title="Operation Year", yaxis_title="Cumulative Cash Flow (USD)", font=dict(family="Arial Black", size=18), legend=dict(font=dict(size=16)))
    fig_pay.update_xaxes(tickfont=dict(size=16))
    fig_pay.update_yaxes(tickfont=dict(size=16))
    st.plotly_chart(fig_pay, use_container_width=True)

