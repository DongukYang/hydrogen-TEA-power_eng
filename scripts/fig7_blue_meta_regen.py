#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig7_blue_meta_regen.py
=======================
Regenerates manuscript Figure 7 (blue hydrogen meta-analysis).
Output: Figure_2_blue_meta.png (matches \includegraphics reference), 300 dpi.

REVISION HISTORY :
  - Panel (b): ALL project-name annotations removed (editorial decision:
    de-identify projects; references remain in Table 6).
  - Panel (b): Linde/Woodside Beaumont coordinate corrected 4.2 -> 2.2 Mt-CO2/yr.
    Verified directly against IEA Hydrogen Production Projects Database
    (June 2026), row "Woodside Energy Beaumont New Ammonia (Texas)",
    Technology details = ATR+CCUS, Capacity (t CO2 capt/y) = 2,200,000.
  - Panel (b): Ascension coordinate = 7.6 Mt (IEA DB "ACE complex (LA),
    phase 2" = full build-out; the old ~12 Mt coordinate is obsolete).
  - Panel (a): pools follow Table 6 Note verbatim -- Table 6 TEA values +
    Table 1 reforming-route estimates, dual "a/b" values split:
    SMR-CCS n=11 (median 2.35), ATR-CCS n=8 (median 1.63).
    Manuscript text updated accordingly (2.36->2.35, 1.66->1.63).

DATA PROVENANCE (panel a):
  SMR pool: Salkuyeh2022 (1.69, 2.36) | Wu2024/NETL2023 (1.64) | Zang2024 (1.56)
            ACS2025 (3.22, 2.59) | IEAGHG2022 range endpoints (2.35, 2.83)
            Cownden2023 (3.33) | Table1: salkuyeh2017 (2.16), IEAGHG2017 (1.65)
  ATR pool: Salkuyeh2022/2017 (1.66, 1.58) | NETL2023 (1.60, 1.33)
            Cownden2023 (3.28) | Table1: NETL2022 (1.51), Equinor/lak2019 (2.72),
            RMI2025 midpoint (2.5)
DATA PROVENANCE (panel b):  (Mt-CO2/yr, billion USD)
  Quest (1.08, 1.05: CAD1.35B @ ~0.78 FX) | Port Arthur (0.90 per IEA DB, 0.431:
  CCS-retrofit capital only) | BP Teesside (2.0, 2.5: GBP2B+) |
  Beaumont (2.2, 1.8) | Ascension full build-out (7.6, 7.5) | AP Louisiana POX (5.0, 8.5)

Dependencies: numpy, matplotlib (>=3.8; tick-label kw auto-selected).
"""
import numpy as np, inspect
import matplotlib.pyplot as plt

OUT = "Figure_2_blue_meta.png"   # run from the figure directory
smr = [1.69, 2.36, 1.64, 1.56, 3.22, 2.59, 2.35, 2.83, 3.33, 2.16, 1.65]  # n=11
atr = [1.66, 1.58, 1.60, 1.33, 3.28, 1.51, 2.5, 2.72]                      # n=8
pts = [(1.08, 1.05, 'SMR'), (0.90, 0.431, 'SMR'), (2.0, 2.5, 'ATR'),
       (2.2, 1.8, 'ATR'), (7.6, 7.5, 'ATR'), (5.0, 8.5, 'POX')]
colors  = {'SMR': '#4C72B0', 'ATR': '#C44E52', 'POX': '#DD8452'}
markers = {'SMR': 'o', 'ATR': 's', 'POX': '^'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
tk = ("tick_labels" if "tick_labels" in inspect.signature(ax1.boxplot).parameters
      else "labels")
bp = ax1.boxplot([smr, atr], widths=0.5, patch_artist=True, showfliers=True,
                 **{tk: [f"SMR-CCS\nn={len(smr)}", f"ATR-CCS\nn={len(atr)}"]})
for p, c in zip(bp['boxes'], ["#4C72B0", "#C44E52"]):
    p.set_facecolor(c); p.set_alpha(0.6)
for m in bp['medians']:
    m.set_color("black"); m.set_linewidth(1.6)
ax1.set_ylabel(r"LCOH (USD/kg-H$_2$)")
ax1.set_title("(a) TEA LCOH distribution by technology")
ax1.set_ylim(0, 4.0); ax1.grid(axis='y', ls=':', alpha=0.4)

seen = set()
for mt, cx, t in pts:
    lbl = {'SMR': 'SMR-CCS', 'ATR': 'ATR-CCS', 'POX': 'POX-CCS'}[t]
    ax2.scatter(mt, cx, s=90, c=colors[t], marker=markers[t],
                edgecolors='black', linewidths=0.7, zorder=3,
                label=lbl if t not in seen else None)
    seen.add(t)
x = np.linspace(0, 8.2, 50)
for s, ls in [(1.0, '--'), (0.5, ':')]:
    ax2.plot(x, s * x, ls=ls, lw=0.9, color='grey', alpha=0.7, zorder=1)
    ax2.annotate(f"{s:.1f} B\\$/Mt", xy=(7.0, s * 7.0), fontsize=8,
                 color='grey', xytext=(3, 4), textcoords='offset points')
ax2.set_xlabel(r"CO$_2$ capture (Mt-CO$_2$/yr)")
ax2.set_ylabel("Disclosed CAPEX (billion USD)")
ax2.set_title(r"(b) Announced-project CAPEX vs CO$_2$ capture")
ax2.set_xlim(0, 8.2); ax2.set_ylim(0, 10)
ax2.legend(frameon=True, loc="upper left", fontsize=9)
ax2.grid(ls=':', alpha=0.35)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("medians (a):", np.median(smr), np.median(atr), "| wrote", OUT)
