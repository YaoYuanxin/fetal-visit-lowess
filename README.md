# Fetal Weight Longitudinal LOWESS Analysis

Exploratory analysis of longitudinal fetal weight trajectories from an NICHD
study: 1120 subjects, each with exactly 4 estimated fetal weight (EFW)
ultrasound measurements at roughly 17, 25, 33, and 37 weeks gestation, plus
birthweight recorded at delivery (treated as a 5th measurement occasion).

The motivating question is whether the growth trajectory differs enough
across the four visit occasions to justify fitting four separate models
rather than one pooled model — LOWESS smooths are used as the visual
diagnostic for that decision.

## Scripts

- **`qc.py`** — Data quality report: record counts, missingness, visit
  timing, monotonicity of time/EFW within subject, birthweight consistency,
  and derived-column checks. Prints to stdout, produces no files.
- **`analysis.py`** — Tasks 1-3: LOWESS fits of EFW vs. gestational day, one
  panel per visit (j=1..4), both raw and visit-centered, for all subjects and
  for the top 15th percentile of birthweight. Writes per-visit and composite
  PNGs plus `visit_phase_summary.csv` to `figures/`. Also contains the shared
  `lowess()`/`trim_tails()` implementation used by `analysis5.py`.
- **`analysis5.py`** — Task 4: adds birthweight as a 5th measurement occasion
  (j=5, time = day of birth) and re-runs the same LOWESS treatment across all
  five occasions, plus an overlay figure showing all five on shared axes.
  Writes PNGs and `five_measurement_summary.csv` to `figures/`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running

```powershell
.\.venv\Scripts\python.exe qc.py
.\.venv\Scripts\python.exe analysis.py
.\.venv\Scripts\python.exe analysis5.py
```

Place `longitudinal_data_processed.csv` in `./data/` first. Figures and
summary CSVs are written to `./figures/`.

## Data quality notes

- **Non-monotonic EFW.** Subjects 647, 3346, and 6225 have at least one visit
  where EFW decreases from the prior visit, despite gestational age always
  increasing.
- **Final EFW exceeds birthweight.** 121 subjects (10.8% of the cohort) have
  their last ultrasound EFW estimate higher than their recorded birthweight —
  plausible near term given measurement error and the scan-to-birth gap
  (median 20 days), but worth flagging before treating EFW and birthweight
  as directly comparable.
- **Visit windows overlap.** Visit 1 timing is disjoint from visit 2, but
  visits 2/3 and 3/4 overlap in gestational day range. `j` indexes measurement
  occasion (the within-subject visit rank), not a partition of gestational
  time — a given calendar day can fall in different `j` for different
  subjects.
