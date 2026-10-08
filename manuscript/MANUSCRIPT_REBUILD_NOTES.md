# Manuscript notes

Paper: `fletcher_mulrooney_2026.tex` (PDF alongside it). Everything quantitative in
it is regenerated and checked by `../paper_reproduction/` (see its README).

## History

The first draft in this repository described a study that was never run: 8,347
incident records, a risk stratification index, nine placeholder references, and an
incorrect expansion of "ArcGIS". None of that data existed here, so the draft was
replaced with the analysis the repository actually performs: beauty-supply retail
access across North Carolina census tracts, framed as an exposure-opportunity proxy
for hair adhesives.

## Corrections made in the latest revision

Found while building `paper_reproduction/`, which recomputes every number:

- Table 1 and the abstract had three rounding slips from rounding twice
  (81.8 should be 81.7, 5.6 should be 5.5, 64.1 should be 64.0). Fixed.
- Table 2 had Fayetteville's Moran's I as 0.77; the value is 0.76. Fixed. The
  placeholder cell for the statewide mean store count is now 0.97.
- The text said majority-Black tracts in Fayetteville were farther from stores.
  They are closer (4.4 vs 5.7 km). The within-metro paragraph is rewritten with all
  five metros, and its conclusion is now worded as a suggestion, not a finding.
- "188 stores within the state" was wrong. 183 are inside the state outline and 188
  are inside it or within 8 km. The 161 points beyond that are mostly real stores in
  South Carolina, Georgia and Tennessee returned by a bounding-box query (156 of the
  161 lie inside North Carolina's bounding box), not coordinate noise as the draft
  said. The Limitations item is rewritten.
- The methods said two case studies; there are five.
- The regression is a negative binomial GLM with the dispersion fixed at 1. The
  paper now says so and reports two sensitivity checks (estimated dispersion,
  county-clustered standard errors). Neither changes any conclusion.
- Moran's I p-values are permutation pseudo-p values, now labelled as such.
- New sentence in Results 3.3: access by population-density quartile (98.5% of
  lowest-density tracts have no store within 5 km, against 24.2% of the highest).
  This gives the paper's repeated "urban to rural" statements a measured basis.
  Delete the sentence if you prefer not to include it.
- The affiliation lines ran off the right edge of page 1. Fixed.
- All en and em dashes were removed. Ranges read "3.9 to 20.4"; page ranges in the
  reference list use a hyphen. The only dash-like mark left in the PDF is the minus
  sign in "rho = -0.21".

## Still open before submission

1. `[CONFIRM]` markers: affiliations for Schultz and Mulrooney, co-author approval,
   acknowledgments.
2. Store data. OpenStreetMap undercounts, varied between runs, and the statewide
   extract mixes in out-of-state stores. A validated registry is the biggest
   improvement available.
3. Framing claim to source or soften: the abstract says hair adhesives "are
   marketed disproportionately to Black consumers", and the introduction says they
   are "used disproportionately by" Black women and girls. The cited papers support
   hair products in general, not adhesives specifically.
4. References: the reference list was written from memory and needs checking
   against the sources (titles, volumes, pages, DOIs), especially James-Todd et al.
   The proposal's `references.bib` has verified entries for several of these.
5. Spatial dependence: the regression p-values ignore it. A spatial lag or error
   model, or county-level random effects, would be the stronger analysis.

## Target journals

Health & Place; International Journal of Health Geographics; Applied Geography;
Journal of Exposure Science and Environmental Epidemiology.
