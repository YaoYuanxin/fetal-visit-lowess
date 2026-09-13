# Project context

Statistics research collaboration analyzing longitudinal fetal weight data.
The collaborator (Dr. Sinha) sends numbered task lists; the expected reply
is figures and summary statistics, not narrative prose.

## Data

- `data/longitudinal_data_processed.csv` and `data/five_measurement_long.csv`
  are NICHD study data. The `data/` directory is gitignored — never commit
  these files, and never paste data values (subject IDs aside, which are
  arbitrary) into commit messages or docs.
- `five_measurement_long.csv` is a convenience export only — no script reads
  it. `analysis5.py` derives the same 5-occasion frame itself from
  `longitudinal_data_processed.csv`.

## Scripts

- `qc.py` — data quality checks (structure, missingness, timing, monotonicity,
  internal consistency). Read-only, no figures.
- `analysis.py` — Tasks 1-3: per-visit LOWESS. Also holds the shared
  `lowess()` and `trim_tails()` implementation that `analysis5.py` imports.
- `analysis5.py` — Task 4: treats birthweight as a 5th measurement occasion
  (j=5, t=birthday). Imports from `analysis.py` via a normal relative import
  (`from analysis import lowess, trim_tails, FRAC, TRIM`) — no sys.path hacks.

Keep scripts flat in the repo root; do not reorganize into `src/`.
Figures and summary CSVs go to `./figures/` (created via `os.makedirs`).

## LOWESS conventions

`frac=0.5`, `it=3` (bisquare robustness iterations), fit restricted to the
1st-99th percentile of `t` within each visit — the tails are 1-2 isolated
points and the local fit degenerates there. Do not change these parameters
without asking; they were chosen deliberately, not defaults.

## Environment

Windows PowerShell. `.venv\Scripts\Activate.ps1` does not persist across
separate tool calls — always invoke the interpreter by path:
`.\.venv\Scripts\python.exe script.py`
