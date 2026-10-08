"""The three figures in the manuscript.

fig1_nc_choropleth.png     distance to nearest store by tract, with stores
fig2_quartile_gradient.png access by quartile of tract percent-Black
fig3_metros.png            share of tracts with no store within 5 km, five metros
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED = "#1b1b1b", "#6b6b6b"
BLUE, ACCENT = "#2a6f97", "#7a1f3d"   # neutral bars, highlighted bar and store points

plt.rcParams.update({
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.edgecolor": "#999999", "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200,
})


def fig1_choropleth(tracts, mapped_stores, results, out_path):
    state = results["statewide"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    gp = tracts.to_crs(3857).copy()
    cap = np.nanpercentile(tracts["nearest_store_km"], 95)   # cap so one far tract cannot flatten the ramp
    gp["d"] = gp["nearest_store_km"].clip(upper=cap)
    gp.plot(column="d", cmap="viridis_r", linewidth=0.05, edgecolor="#ffffff", ax=ax,
            legend=True, legend_kwds={"label": "Distance to nearest store (km)", "shrink": 0.6})
    mapped_stores.to_crs(3857).plot(ax=ax, color=ACCENT, markersize=6, marker="o",
                                    alpha=0.9, edgecolor="white", linewidth=0.2)
    ax.set_title("Beauty-supply retail access across North Carolina",
                 fontsize=12, color=INK, loc="left", weight="bold")
    ax.text(0, -0.04,
            "Dark = far from a store (worse access).  Maroon dots = stores "
            f"(n={state['stores_mapped_within_8km_of_state']} within or near NC).",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="top")
    ax.text(0, -0.095,
            f"{state['pct_no_store_5km']}% of NC tracts have no store within 5 km; "
            "access deserts are predominantly rural.",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="top")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig2_quartiles(results, out_path):
    qs = results["quartiles"]
    labels = ["Q1\nlowest\n%Black", "Q2", "Q3", "Q4\nhighest\n%Black"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6))
    panels = [
        (a1, [q["median_nearest_km"] for q in qs], "Median distance to\nnearest store (km)",
         "Closer where more residents are Black", "{:.1f}", 0.3),
        (a2, [q["pct_no_store_5km"] for q in qs], "% of tracts with no\nstore within 5 km",
         "Fewer access deserts, too", "{:.0f}%", 1.2),
    ]
    for ax, vals, ylabel, title, fmt, pad in panels:
        bars = ax.bar(range(4), vals, color=BLUE, width=0.68, zorder=3)
        bars[3].set_color(ACCENT)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        for i, v in enumerate(vals):
            ax.text(i, v + pad, fmt.format(v), ha="center", fontsize=8, color=INK)
        ax.grid(axis="y", color="#eeeeee", zorder=0)
    fig.suptitle(f"Beauty-supply access improves with tract percent-Black "
                 f"({results['statewide']['tracts']:,} NC tracts)",
                 fontsize=11, x=0.01, ha="left", weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig3_metros(results, out_path):
    metros = sorted(results["metros"], key=lambda m: m["median_nearest_km"])
    state_pct = results["statewide"]["pct_no_store_5km"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    vals = [m["pct_no_store_5km"] for m in metros]
    ax.bar(range(len(metros)), vals, color=BLUE, width=0.6, zorder=3)
    ax.axhline(state_pct, color=ACCENT, linewidth=1.2, linestyle="--", zorder=4)
    ax.text(len(metros) - 0.5, state_pct + 1.2, f"Statewide {state_pct}%", ha="right",
            fontsize=8.5, color=ACCENT)
    ax.set_xticks(range(len(metros)))
    ax.set_xticklabels([m["metro"] for m in metros], fontsize=9)
    ax.set_ylabel("% of tracts with no\nstore within 5 km", fontsize=9)
    ax.set_ylim(0, max(vals + [state_pct]) + 10)
    ax.set_title("Retail access in five NC metros", fontsize=11, loc="left", weight="bold", color=INK)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    ax.grid(axis="y", color="#eeeeee", zorder=0)
    ax.text(0, -0.17, "Metros are county subsets of the statewide layer; "
            "bars sorted by median distance to the nearest store.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_all(tracts, mapped_stores, results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig1_choropleth(tracts, mapped_stores, results, out_dir / "fig1_nc_choropleth.png")
    fig2_quartiles(results, out_dir / "fig2_quartile_gradient.png")
    fig3_metros(results, out_dir / "fig3_metros.png")
    return sorted(p.name for p in out_dir.glob("fig*.png"))
