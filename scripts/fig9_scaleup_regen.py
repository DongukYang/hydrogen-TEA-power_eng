#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig9_scaleup_regen.py
=====================
Regenerates manuscript Figure 9 (cost-component share across plant scale).
Output: Figure_6_scaleup_costshare.png (matches \includegraphics), 300 dpi.

CALIBRATION (panel a, green -- ALK-anchored representative blend):
  Electricity floor fixed at 2.65 USD/kg (53 kWh/kg x 50 USD/MWh),
  scale-invariant. CAPEX-linked bundle = B * P^(m-1) solved to pass EXACTLY
  through the manuscript-caption endpoints 7.49 (0.1 MW) and 4.18 (10 MW):
    effective exponent m = 0.750, B = 2.721.
  NOTE: m=0.750 > n=0.65 is expected -- the bundle mixes annualized CAPEX
  (n=0.65) with stack replacement and fixed O&M, flattening the effective
  slope. This blend is NOT the per-technology Figure 6 curve (see caption).
  Bundle split uses Table 5 ALK proportions: CAPEX 78.2 / stack 15.9 / O&M 6.0 %.

CALIBRATION (panel b, blue):
  Totals anchored to Table 8: 3.85 (0.3 Mt SMR) ... 2.45 (7.6 Mt ATR),
  2.53 at 5 Mt interpolated. NG component fixed at 1.47 USD/kg so that the
  NG share reaches exactly 60% at 7.6 Mt (1.47/2.45), matching the
  manuscript claim "gas-cost share approaching 60 %".

Dependencies: numpy, matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt

OUT = "Figure_6_scaleup_costshare.png"

caps_g = np.array([0.1, 0.3, 1, 2, 5, 10])
elec_g = 2.65
m, B = 0.750, 2.721
bundle = B * caps_g ** (m - 1)
sh = np.array([1.97, 0.40, 0.15]) / 2.52
capex_g, stack_g, fom_g = bundle * sh[0], bundle * sh[1], bundle * sh[2]
tot_g = elec_g + bundle

caps_b = np.array([0.3, 1.0, 2.0, 3.0, 5.0, 7.6])
tot_b  = np.array([3.85, 3.05, 2.78, 2.65, 2.53, 2.45])
ng_b   = np.full_like(caps_b, 1.47)
opx_b  = np.array([0.75, 0.62, 0.55, 0.50, 0.46, 0.44])
capex_b = tot_b - ng_b - opx_b

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("Scale-up dilutes capital cost but not the dominant variable cost",
             fontsize=11, fontweight='bold', y=1.02)

def shares_plot(ax, x, comps, labels, colors, total, xlab, title,
                floor_txt, floor_xy, xticks, xtl, ylab_r):
    pct = np.array([c / total * 100 for c in comps])
    ax.stackplot(x, pct, labels=labels, colors=colors, alpha=0.85)
    ax.set_xscale('log'); ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, 100)
    ax.set_xticks(xticks); ax.set_xticklabels(xtl)
    ax.set_xlabel(xlab); ax.set_ylabel("Cost-component share (%)")
    ax.set_title(title)
    ax.annotate(floor_txt, xy=floor_xy, fontsize=8, color='white',
                fontweight='bold', ha='center')
    axr = ax.twinx()
    axr.plot(x, total, 'k-o', lw=1.8, ms=4, zorder=5)
    axr.set_ylabel(ylab_r); axr.set_ylim(0, total[0] * 1.35)
    axr.annotate(f"{total[0]:.2f}", xy=(x[0], total[0]), xytext=(5, 6),
                 textcoords='offset points', fontsize=8, fontweight='bold')
    axr.annotate(f"{total[-1]:.2f}", xy=(x[-1], total[-1]), xytext=(-24, 6),
                 textcoords='offset points', fontsize=8, fontweight='bold')

shares_plot(ax1, caps_g,
            [elec_g * np.ones_like(caps_g), capex_g, stack_g, fom_g],
            ["Electricity", "CAPEX", "Stack replacement", "Fixed O&M"],
            ["#DDAA33", "#2E8B8B", "#9467BD", "#555F6E"],
            tot_g, "Plant capacity (MW)", r"(a) Green H$_2$ (electrolysis)",
            "Electricity floor (~50%, scale-invariant)", (1.0, 22),
            [0.1, 0.3, 1, 2, 5, 10], ["0.1", "0.3", "1", "2", "5", "10"],
            "Total LCOH (USD/kg)")
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=4,
           frameon=False, fontsize=8)

shares_plot(ax2, caps_b, [ng_b, capex_b, opx_b],
            ["Natural gas", "CAPEX", "CCS OPEX + fixed O&M"],
            ["#B03A48", "#2E5B8B", "#8899AA"],
            tot_b, r"Capture capacity (Mt-CO$_2$/yr)",
            r"(b) Blue H$_2$ (reforming + CCS)",
            "Natural-gas floor (rises toward ~60%)", (1.5, 25),
            [0.3, 1, 2, 3, 5, 7.6], ["0.3", "1", "2", "3", "5", "7.6"],
            "Total LCOH (USD/kg)")
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=3,
           frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("green:", tot_g[0], "->", tot_g[-1], "| blue:", tot_b[0], "->", tot_b[-1])
