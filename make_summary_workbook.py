"""Emit a self-contained summary workbook for the collaborator.

Reads the source data, recomputes the per-occasion summary statistics, and
writes figures/summary_tables.xlsx with three sheets:

  Data dictionary  - every abbreviation and column defined
  All subjects     - j = 1..5, full cohort
  Top 15 percent   - j = 1..4, birthweight at or above the 85th percentile

Run after analysis.py / analysis5.py:
    .\\.venv\\Scripts\\python.exe make_summary_workbook.py
"""

import os
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

DATA = "data/longitudinal_data_processed.csv"
OUT_DIR = "figures"
OUT = os.path.join(OUT_DIR, "summary_tables.xlsx")

FONT = "Arial"
HDR_FILL = PatternFill("solid", fgColor="D9D9D9")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(bottom=THIN)

OCCASION = {1: "Visit 1 (~17 weeks)", 2: "Visit 2 (~25 weeks)",
            3: "Visit 3 (~33 weeks)", 4: "Visit 4 (~37 weeks)",
            5: "Birth"}

DICTIONARY = [
    ("Term", "Definition"),
    ("i", "Subject index. The cohort has 1,120 subjects."),
    ("j", "Measurement occasion index, being the within-subject rank of the "
          "measurement date. j = 1 to 4 are ultrasound scans; j = 5 is birth. "
          "Note that j indexes the occasion, not an interval of gestational "
          "time: the occasions overlap, so the same gestational day can fall "
          "under different j for different subjects."),
    ("t_ij", "Gestational age in days at which measurement j was taken for "
             "subject i. For j = 5 this is the day of birth."),
    ("Y_ij(t_ij)", "Weight in grams at occasion j for subject i. For j = 1 to 4 "
                   "this is estimated fetal weight from ultrasound; for j = 5 "
                   "it is measured birthweight."),
    ("EFW", "Estimated fetal weight. Derived from ultrasound biometry, not a "
            "direct measurement."),
    ("ybar_j", "Sample mean of Y_ij across all subjects at occasion j. Used to "
               "centre the plots in item 2."),
    ("n", "Number of observations contributing to the row."),
    ("Mean (g)", "Arithmetic mean of Y_ij at that occasion, in grams."),
    ("SD (g)", "Sample standard deviation of Y_ij at that occasion, in grams. "
               "Denominator n - 1."),
    ("CV", "Coefficient of variation, SD divided by mean. Unitless. Included "
           "because absolute SD rises across occasions while CV falls, which "
           "bears on whether a multiplicative model fits better than separate "
           "additive ones."),
    ("Slope (g/day)", "Ordinary least squares slope of Y_ij regressed on t_ij "
                      "within that occasion only. A local rate of weight gain, "
                      "not a growth curve across occasions."),
    ("t range", "Minimum and maximum gestational day observed at that occasion."),
    ("Top 15 percent", "Subjects whose birthweight is at or above the 85th "
                       "percentile of the cohort. The percentile is taken over "
                       "subjects, not over records, because birthweight is "
                       "constant within a subject."),
    ("LOWESS", "Locally weighted scatterplot smoothing, Cleveland (1979). "
               "Fitted with bandwidth frac = 0.5 and 3 robustness iterations."),
    ("Trimming", "Each LOWESS fit is restricted to the 1st to 99th percentile "
                 "of t within its occasion. At the extremes the local fit rests "
                 "on one or two isolated points and swings by several hundred "
                 "grams. Excluded points are still plotted, marked with a grey "
                 "cross."),
]

NOTES = [
    "Source: data/longitudinal_data_processed.csv. Every figure in this row "
    "is recomputed from that file by make_summary_workbook.py; nothing is "
    "typed in by hand.",
    "Three subjects (647, 3346, 6225) have a j = 3 weight exceeding their "
    "j = 4 weight. They are retained in these figures.",
    "121 subjects, 10.8 percent of the cohort, have a j = 4 EFW above their "
    "recorded birthweight. Median excess 99 g, maximum 626 g.",
]


def summarise(d, occasions):
    rows = []
    for j in occasions:
        g = d[d.j == j]
        t, y = g.time.values.astype(float), g.weight.values.astype(float)
        rows.append([j, OCCASION[j], len(g),
                     round(float(y.mean()), 1),
                     round(float(y.std(ddof=1)), 1),
                     round(float(y.std(ddof=1) / y.mean()), 3),
                     round(float(np.polyfit(t, y, 1)[0]), 2),
                     int(t.min()), int(t.max())])
    return rows


def load():
    df = pd.read_csv(DATA).sort_values(["id", "time"]).reset_index(drop=True)
    df["j"] = df.groupby("id").cumcount() + 1
    scans = df[["id", "j", "time", "efw"]].rename(columns={"efw": "weight"})
    born = (df.groupby("id")
              .agg(time=("birthday", "first"), weight=("bw", "first"))
              .reset_index())
    born["j"] = 5
    long5 = pd.concat([scans, born[["id", "j", "time", "weight"]]],
                      ignore_index=True)

    bw = df.groupby("id")["bw"].first()
    thresh = bw.quantile(0.85)
    top_ids = bw[bw >= thresh].index
    return long5, long5[long5.id.isin(top_ids)], thresh, len(top_ids)


def style_header(ws, row, ncol):
    for c in range(1, ncol + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(name=FONT, bold=True)
        cell.fill = HDR_FILL
        cell.border = BORDER
        cell.alignment = Alignment(vertical="center", wrap_text=True)


def write_table(ws, rows, title, subtitle):
    ws["A1"] = title
    ws["A1"].font = Font(name=FONT, bold=True, size=13)
    ws["A2"] = subtitle
    ws["A2"].font = Font(name=FONT, italic=True, size=10)
    ws["A2"].alignment = Alignment(wrap_text=True)

    hdr = ["j", "Occasion", "n", "Mean (g)", "SD (g)", "CV",
           "Slope (g/day)", "t min (days)", "t max (days)"]
    for c, h in enumerate(hdr, 1):
        ws.cell(row=4, column=c, value=h)
    style_header(ws, 4, len(hdr))

    for r, row in enumerate(rows, 5):
        for c, v in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.font = Font(name=FONT)
            if c in (4, 5):
                cell.number_format = "#,##0.0"
            elif c == 6:
                cell.number_format = "0.000"
            elif c == 7:
                cell.number_format = "#,##0.00"

    last = 4 + len(rows)
    ws.cell(row=last + 2, column=1, value="Notes").font = Font(name=FONT, bold=True)
    for i, n in enumerate(NOTES, last + 3):
        c = ws.cell(row=i, column=1, value=n)
        c.font = Font(name=FONT, size=9)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=9)
        ws.row_dimensions[i].height = 28

    widths = [5, 22, 8, 11, 11, 9, 14, 13, 13]
    for c, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[4].height = 30
    ws.freeze_panes = "A5"


def write_dictionary(ws):
    ws["A1"] = "Data dictionary"
    ws["A1"].font = Font(name=FONT, bold=True, size=13)
    ws["A2"] = ("Definitions for every abbreviation used in this workbook and "
                "in the accompanying figures.")
    ws["A2"].font = Font(name=FONT, italic=True, size=10)

    for c, h in enumerate(DICTIONARY[0], 1):
        ws.cell(row=4, column=c, value=h)
    style_header(ws, 4, 2)

    for r, (term, defn) in enumerate(DICTIONARY[1:], 5):
        a = ws.cell(row=r, column=1, value=term)
        a.font = Font(name=FONT, bold=True)
        a.alignment = Alignment(vertical="top")
        b = ws.cell(row=r, column=2, value=defn)
        b.font = Font(name=FONT)
        b.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = max(15, 13 * (len(defn) // 95 + 1))

    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 95
    ws.freeze_panes = "A5"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    long5, top, thresh, ntop = load()

    wb = Workbook()
    write_dictionary(wb.active)
    wb.active.title = "Data dictionary"

    write_table(wb.create_sheet("All subjects"),
                summarise(long5, range(1, 6)),
                "Summary by measurement occasion, all subjects",
                "1,120 subjects, each with four ultrasound measurements and "
                "one birthweight.")

    write_table(wb.create_sheet("Top 15 percent"),
                summarise(top, range(1, 5)),
                "Summary by measurement occasion, top 15 percent of birthweight",
                f"{ntop} subjects with birthweight at or above the 85th "
                f"percentile ({thresh:.0f} g). Ultrasound occasions only.")

    wb.save(OUT)
    print(f"wrote {OUT}")
    for ws in wb.worksheets:
        print(f"  sheet: {ws.title}")


if __name__ == "__main__":
    main()
