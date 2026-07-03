import numpy as np

# ============================================
# Reference case: 1 MW PEM
# 1,800 USD/kW, 51 kWh/kg, 50 USD/MWh, CF 50%, 20-yr life
# ============================================
P_kW = 1000.0
capex_per_kW = 1800.0
I0 = P_kW * capex_per_kW          # 1.8 M USD
SEC = 51.0                         # kWh/kg
elec_price = 50.0 / 1000.0         # USD/kWh = 0.05
CF = 0.50
N = 20
r = 0.08
tau = 0.22

hours = 8760 * CF                  # 4380 h/yr
E_annual = P_kW * hours            # kWh/yr = 4.38 GWh
H_annual = E_annual / SEC          # kg/yr
print(f"Annual H2 (no degradation): {H_annual:,.0f} kg/yr")
# Manuscript Table 5 PEM: 85,882 kg/yr

FOM = 0.03 * I0                    # 3%/yr of CAPEX
rep_frac = 0.40                    # 40% of CAPEX in year 8
elec_cost = E_annual * elec_price  # USD/yr

disc = np.array([(1+r)**-t for t in range(0, N+1)])

def npv_H(H_t):
    return sum(H_t[t]*(1+r)**-t for t in range(1, N+1))

# --- T1: After-tax DCF, MACRS, output decay ---
# MACRS-7 schedule
macrs7 = [0.1429,0.2449,0.1749,0.1249,0.0893,0.0892,0.0893,0.0446]
deg = 0.010  # 1.0 %/yr degradation (T1 includes output decay)
H_t = [0]+[H_annual*(1-deg)**(t-1) for t in range(1,N+1)]
# Costs: I0 at t=0; FOM+elec each year; replacement at t=8; tax shield -tau*D_t
costs = np.zeros(N+1)
costs[0] = I0
for t in range(1, N+1):
    costs[t] = FOM + elec_cost
costs[8] += rep_frac * I0
# Depreciation shield on I0 (MACRS-7, years 1-8)
for i, d in enumerate(macrs7):
    costs[i+1] -= tau * d * I0
# Depreciation shield on replacement CAPEX (years 9-16)
for i, d in enumerate(macrs7):
    costs[8+i+1] -= tau * d * rep_frac * I0
npvC = sum(costs[t]*(1+r)**-t for t in range(0,N+1))
# After-tax DCF: LCOH must be grossed up so post-tax revenue covers costs
# Standard after-tax LCOH: LCOH = NPV(costs incl tax effects)/NPV(H) / (1-tau) applied to margin...
# Simple version (shield-credit form, as in Eq.2-3 with -tau*D_t toggle):
lcoh_T1_simple = npvC / npv_H(H_t)
print(f"T1 (shield-credit form, w/ decay): {lcoh_T1_simple:.2f}")

# Full after-tax revenue-requirement form:
# sum (LCOH*H_t*(1-tau) - (FOM+elec)*(1-tau) + tau*D_t) /(1+r)^t = I0 + rep
npvH = npv_H(H_t)
npv_opex = sum((FOM+elec_cost)*(1+r)**-t for t in range(1,N+1))
npv_shield = sum(tau*macrs7[i]*I0*(1+r)**-(i+1) for i in range(8)) + \
             sum(tau*macrs7[i]*rep_frac*I0*(1+r)**-(8+i+1) for i in range(8))
npv_capex = I0 + rep_frac*I0*(1+r)**-8
lcoh_T1_full = (npv_capex + npv_opex*(1-tau) - npv_shield) / (npvH*(1-tau))
print(f"T1 (full after-tax rev-req, w/ decay): {lcoh_T1_full:.2f}")
print("Manuscript T1 = 5.86")

# --- T2: Pre-tax CRF annuity, no tau*D ---
CRF = r*(1+r)**N/((1+r)**N-1)
H_flat = H_annual
rep_annualized = rep_frac*I0*(1+r)**-8 * CRF
lcoh_T2 = (CRF*I0 + FOM + elec_cost + rep_annualized) / H_flat
print(f"\nT2 (pre-tax CRF): {lcoh_T2:.2f}  (manuscript 5.42)")

# --- T6: like T2 minus O2 credit ---
# O2: 8 kg O2 per kg H2, price ~0.02-0.03 USD/kg O2 gives ~0.16-0.24 USD/kg credit
o2_credit = lcoh_T2 - 5.24
print(f"T6 implied by-product+subsidy credit vs T2: {o2_credit:.2f} USD/kg")

# --- T7: levered IRR 12%, 60/40 ---
# r_lev = 0.6*r_d*(1-tau)+0.4*r_e; if equity IRR 12%, debt ~5-6%
for rd in [0.05, 0.06]:
    r_lev = 0.6*rd*(1-tau) + 0.4*0.12
    CRF7 = r_lev*(1+r_lev)**N/((1+r_lev)**N-1)
    rep7 = rep_frac*I0*(1+r_lev)**-8*CRF7
    lcoh_T7 = (CRF7*I0 + FOM + elec_cost + rep7)/H_flat
    print(f"T7 (rd={rd}): r_lev={r_lev:.4f}, LCOH={lcoh_T7:.2f}  (manuscript 5.92)")

# --- Spread check ---
print(f"\nSpread check: 6.02/5.24 = {6.02/5.24:.3f}  -> 1.15x claim")
print(f"6.3x check: 11.67/1.84 = {11.67/1.84:.2f}")

# --- Sanity: Table 5 PEM annual H2 with degradation avg? ---
# 85,882 vs 85,882? Simple: 4,380,000/51 = 85,882.35
print(f"\nTable 5 PEM Annual H2: {4380000/51:,.1f} (manuscript 85,882) OK")
print(f"Table 5 ALK Annual H2: {4380000/53:,.1f} (manuscript 82,642)")
print(f"Table 5 AEM Annual H2: {4380000/50:,.1f} (manuscript 87,600)")
print(f"Table 5 SOEC Annual H2: {4380000/38:,.1f} (manuscript 115,263)")
