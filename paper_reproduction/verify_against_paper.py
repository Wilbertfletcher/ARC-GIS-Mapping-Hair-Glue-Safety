#!/usr/bin/env python3
"""Check that the numbers printed in the manuscript match the computed results.

    python reproduce.py
    python verify_against_paper.py            # exit code 0 = everything matches

Each check below pairs a value typed from the manuscript (the `paper` column)
with the value `reproduce.py` computed (the `computed` column), and also confirms
that the manuscript text really contains that number as written. A final group of
checks confirms that Tables 1 and 2 in the .tex are identical to the rows the code
generates, and that the manuscript contains no en or em dashes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TEX = HERE.parent / "manuscript" / "fletcher_mulrooney_2026.tex"

rows = []   # (ok, where, paper, computed)


def squash(text):
    return re.sub(r"\s+", " ", text)


def strip_comments(tex):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in tex.split("\n"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "results" / "results.json")
    ap.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    args = ap.parse_args(argv)
    if not args.results.exists():
        sys.exit("results/results.json not found. Run `python reproduce.py` first.")
    R = json.loads(args.results.read_text())
    raw_tex = args.tex.read_text(encoding="utf-8")
    tex = squash(strip_comments(raw_tex))

    def check(where, paper, computed, nd=None, tex_strings=(), op="eq"):
        got = round(computed, nd) if nd is not None and computed is not None else computed
        ok = (got == paper) if op == "eq" else (got < paper)
        missing = [t for t in tex_strings if squash(t) not in tex]
        if missing:
            ok = False
            where += f"  [text missing: {missing}]"
        rows.append((ok, where, paper if op == "eq" else f"< {paper}", got))

    st, mo, sp = R["statewide"], R["moran"], R["spearman"]
    nb, de = R["negative_binomial"], R["density_quartiles"]
    metro = {m["metro"]: m for m in R["metros"]}

    # ---- statewide counts (Abstract, Methods 2.1, Results 3.1, Limitations)
    check("tracts analysed", 2649, st["tracts"], tex_strings=["2{,}649"])
    check("tracts in file", 2672, st["tracts_in_file"], tex_strings=["2{,}672"])
    check("zero-population tracts excluded", 23, st["zero_population_tracts_excluded"], tex_strings=["23 zero-population"])
    check("population, millions", 10.5, st["population"] / 1e6, 1, ["10.5~million"])
    check("stores in extract", 349, st["stores_in_extract"], tex_strings=["349"])
    check("stores inside the state", 183, st["stores_inside_state"], tex_strings=["183 inside the state"])
    check("stores mapped in Fig. 1 (state + 8 km)", 188, st["stores_mapped_within_8km_of_state"], tex_strings=["188 extracted stores"])
    check("stores more than 8 km outside", 161, st["stores_outside_state"], tex_strings=["161 of its 349"])
    check("...of which inside NC bounding box", 156, st["stores_outside_but_inside_state_bbox"], tex_strings=["156 of those"])
    check("% tracts with no store within 5 km", 67.5, st["pct_no_store_5km"], 1, ["67.5"])
    check("median nearest store, km", 9.3, st["nearest_km_median"], 1, ["9.3~km"])
    check("IQR lower, km", 3.9, st["nearest_km_q1"], 1, ["3.9 to 20.4"])
    check("IQR upper, km", 20.4, st["nearest_km_q3"], 1)
    check("median tract percent Black", 14.7, st["median_pct_black"], 1, ["14.7"])

    # ---- Moran's I (Results 3.2)
    check("Moran I, nearest-store distance", 0.92, mo["nearest_store_km"]["I"], 2, ["0.92"])
    check("Moran I, store count within 5 km", 0.83, mo["store_count_5km"]["I"], 2, ["0.83"])
    check("Moran I, percent Black", 0.66, mo["pct_black"]["I"], 2, ["0.66"])
    for k in mo:
        check(f"Moran pseudo-p, {k}", 0.001, mo[k]["p"], 3)

    # ---- Table 1 and Spearman (Results 3.3)
    t1 = {"Q1": (663, 14.2, 0.68, 81.7), "Q2": (662, 9.2, 0.97, 70.1),
          "Q3": (662, 8.7, 0.98, 64.0), "Q4": (662, 5.5, 1.23, 53.9)}
    for q in R["quartiles"]:
        n, med, mean, none = t1[q["quartile"]]
        check(f"Table 1 {q['quartile']} tracts", n, q["tracts"])
        check(f"Table 1 {q['quartile']} median nearest km", med, q["median_nearest_km"], 1)
        check(f"Table 1 {q['quartile']} mean stores within 5 km", mean, q["mean_stores_5km"], 2)
        check(f"Table 1 {q['quartile']} % with no store", none, q["pct_no_store_5km"], 1)
    check("Abstract: Q4 vs Q1 nearest km", "5.5~km versus 14.2~km", f"{R['quartiles'][3]['median_nearest_km']}~km versus {R['quartiles'][0]['median_nearest_km']}~km",
          tex_strings=["5.5~km versus 14.2~km"])
    check("Abstract: Q4 vs Q1 % no store", "53.9\\% versus 81.7\\%", f"{R['quartiles'][3]['pct_no_store_5km']}\\% versus {R['quartiles'][0]['pct_no_store_5km']}\\%",
          tex_strings=["53.9\\% versus 81.7\\%"])
    check("Spearman, percent Black vs distance", -0.21, sp["pct_black_vs_nearest_km"]["rho"], 2, ["-0.21"])
    check("Spearman, percent Black vs store count", 0.22, sp["pct_black_vs_store_count_5km"]["rho"], 2, ["0.22"])

    # ---- negative binomial and sensitivity checks
    check("NB sample size", 2624, nb["n"], tex_strings=["2{,}624"])
    check("NB IRR percent Black", 1.02, nb["terms"]["pct_black"]["irr"], 2)
    check("NB IRR poverty", 1.02, nb["terms"]["poverty_rate"]["irr"], 2)
    check("NB IRR income per $10k", 1.21, nb["terms"]["inc10k"]["irr"], 2, ["1.21"])
    for k in ("pct_black", "poverty_rate", "inc10k"):
        check(f"NB p < 0.001, {k}", 0.001, nb["terms"][k]["p"], op="lt")
    est = nb["sensitivity_estimated_alpha"]["terms"]
    check("NB (estimated dispersion) IRR percent Black", 1.02, est["pct_black"]["irr"], 2)
    check("NB (estimated dispersion) IRR poverty", 1.02, est["poverty_rate"]["irr"], 2)
    check("NB (estimated dispersion) IRR income", 1.24, est["inc10k"]["irr"], 2, ["1.24 for income"])
    for k in est:
        check(f"NB (estimated dispersion) p < 0.001, {k}", 0.001, est[k]["p"], op="lt")
    for k, v in nb["sensitivity_county_clustered_se"]["terms"].items():
        check(f"NB (county-clustered) p < 0.001, {k}", 0.001, v["p"], op="lt")

    # ---- density quartiles (Results 3.3)
    lo, hi = de[0], de[-1]
    check("lowest-density quartile, % no store", 98.5, lo["pct_no_store_5km"], 1, ["98.5\\%"])
    check("lowest-density quartile, median km", 22.6, lo["median_nearest_km"], 1, ["22.6~km"])
    check("highest-density quartile, % no store", 24.2, hi["pct_no_store_5km"], 1, ["24.2\\%"])
    check("highest-density quartile, median km", 3.1, hi["median_nearest_km"], 1, ["3.1~km"])

    # ---- metros (Table 2, Results 3.4)
    t2 = {  # tracts, median % Black, median nearest, % none, mean stores, Moran I
        "Charlotte": (302, 29.8, 3.2, 22.2, 3.11, 0.79), "Raleigh": (229, 14.3, 3.7, 31.9, 1.75, 0.78),
        "Greensboro": (125, 29.4, 3.0, 29.6, 4.18, 0.70), "Durham": (67, 31.2, 3.6, 29.9, 1.22, 0.61),
        "Fayetteville": (81, 35.9, 5.2, 54.3, 1.33, 0.76)}
    for name, (n, blk, med, none, mean, mi) in t2.items():
        m = metro[name]
        check(f"Table 2 {name} tracts", n, m["tracts"])
        check(f"Table 2 {name} median % Black", blk, m["median_pct_black"], 1)
        check(f"Table 2 {name} median nearest km", med, m["median_nearest_km"], 1)
        check(f"Table 2 {name} % no store", none, m["pct_no_store_5km"], 1)
        check(f"Table 2 {name} mean stores within 5 km", mean, m["mean_stores_5km"], 2)
        check(f"Table 2 {name} Moran I", mi, m["moran_I_distance"], 2)
        check(f"Table 2 {name} pseudo-p", 0.001, m["moran_p_distance"], 3)
    check("Table 2 statewide mean stores within 5 km", 0.97, st["mean_stores_5km"], 2)
    check("Moran range across metros, low", 0.61, R["metro_moran_range"][0], 2, ["0.61 to 0.79"])
    check("Moran range across metros, high", 0.79, R["metro_moran_range"][1], 2)

    # within-metro comparisons quoted in the text: (majority-Black km, other km)
    within = {"Greensboro": (2.4, 3.4), "Raleigh": (3.4, 3.8), "Fayetteville": (4.4, 5.7),
              "Charlotte": (3.5, 3.1), "Durham": (4.3, 3.5)}
    text_form = {"Greensboro": "Greensboro (median 2.4 vs 3.4~km)", "Raleigh": "Raleigh (3.4 vs 3.8~km)",
                 "Fayetteville": "Fayetteville (4.4 vs 5.7~km)", "Charlotte": "Charlotte (3.5 vs 3.1~km)",
                 "Durham": "Durham (4.3 vs 3.5~km)"}
    for name, (maj, oth) in within.items():
        m = metro[name]
        check(f"within-metro {name}: majority-Black vs other (km)", f"{maj} vs {oth}",
              f"{m['nearest_km_majority_black']} vs {m['nearest_km_other']}", tex_strings=[text_form[name]])
    for name in ("Greensboro", "Raleigh", "Fayetteville"):
        check(f"text says majority-Black tracts CLOSER in {name}", True,
              metro[name]["nearest_km_majority_black"] < metro[name]["nearest_km_other"])
    for name in ("Charlotte", "Durham"):
        check(f"text says majority-Black tracts FARTHER in {name}", True,
              metro[name]["nearest_km_majority_black"] > metro[name]["nearest_km_other"])

    # ---- tables in the .tex equal the generated rows
    for fname, label in (("table1_quartiles_rows.tex", "Table 1"), ("table2_metros_rows.tex", "Table 2")):
        for line in (args.results.parent / "tables" / fname).read_text().strip().split("\n"):
            rows.append((squash(line) in tex, f"{label} row in manuscript matches generated row", squash(line)[:60], "present" if squash(line) in tex else "ABSENT"))

    # ---- style rule: no en or em dashes in the manuscript text
    body = strip_comments(raw_tex)
    rows.append(("--" not in body, "no LaTeX en/em dashes (-- or ---) in manuscript", "none", "none" if "--" not in body else "FOUND"))
    rows.append((not re.search("[–—]", body), "no unicode en/em dashes in manuscript", "none", "none" if not re.search("[–—]", body) else "FOUND"))

    # ---- report
    width = max(len(r[1]) for r in rows)
    failed = [r for r in rows if not r[0]]
    for ok, where, paper, got in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {where:<{width}}  paper={paper!s:<28} computed={got}")
    print(f"\n{len(rows) - len(failed)} of {len(rows)} checks passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
