"""Task 4: treat birthweight as a fifth measurement (j=5) at t = birthday.

Produces 5 figures: EFW at visits j=1..4 against scan day, and birthweight
against day of birth, each with a LOWESS fit. Also an all-visits overlay.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import lowess, trim_tails, FRAC, TRIM

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
PHASE5 = {1: "~17 wk", 2: "~25 wk", 3: "~33 wk", 4: "~37 wk", 5: "birth"}


def build():
    df = pd.read_csv("data/longitudinal_data_processed.csv")
    df = df.sort_values(["id", "time"]).reset_index(drop=True)
    df["j"] = df.groupby("id").cumcount() + 1

    # j=5: one row per subject, weight = birthweight, time = day of birth
    born = (df.groupby("id")
              .agg(time=("birthday", "first"), efw=("bw", "first"))
              .reset_index())
    born["j"] = 5
    long5 = pd.concat([df[["id", "j", "time", "efw"]], born[["id", "j", "time", "efw"]]],
                      ignore_index=True)
    return df, long5


def panel5(ax, t, y, j, centered=False):
    measured = (j == 5)
    col = "#55A868" if measured else "#4878A8"
    ax.scatter(t, y, s=9, alpha=.30, color=col, edgecolors="none",
               label=("birthweight" if measured else "estimated fetal weight")
                     + f" (n={len(t)})")

    tf, yf, (lo, hi), ndrop = trim_tails(t, y)
    s = lowess(tf, yf)
    ax.plot(s[0], s[1], color="#C44E52", lw=2.2,
            label=f"LOWESS (frac={FRAC}, {TRIM}-{100-TRIM} pctile of t)")
    if ndrop:
        ax.scatter(t[(t < lo) | (t > hi)], y[(t < lo) | (t > hi)],
                   s=18, color="#999", marker="x", lw=.9,
                   label=f"excluded from fit ({ndrop})")

    if centered:
        ax.axhline(0, color="#555", lw=.8, ls="--", zorder=0)
        ax.set_ylabel(r"$Y_{ij}(t_{ij}) - \bar{y}_j$  (g)")
    else:
        ax.set_ylabel(r"$Y_{ij}(t_{ij})$  (g)")
    ax.set_xlabel(r"$t_{ij}$  (gestational days)")
    ax.set_title(f"Measurement j = {j}  ({PHASE5[j]})", fontsize=11)
    ax.legend(fontsize=8, framealpha=.9)
    ax.grid(alpha=.25, lw=.5)


def make5(long5, centered, tag, suptitle):
    for j in range(1, 6):
        d = long5[long5.j == j]
        t = d.time.values.astype(float)
        y = d.efw.values - (d.efw.values.mean() if centered else 0)
        fig, ax = plt.subplots(figsize=(7, 5))
        panel5(ax, t, y, j, centered)
        fig.tight_layout()
        fig.savefig(f"{OUT}/{tag}_meas{j}.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    for j in range(1, 6):
        d = long5[long5.j == j]
        t = d.time.values.astype(float)
        y = d.efw.values - (d.efw.values.mean() if centered else 0)
        panel5(axes.flat[j - 1], t, y, j, centered)
    axes.flat[5].axis("off")
    fig.suptitle(suptitle, fontsize=13, y=.995)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{tag}_composite.png", dpi=150)
    plt.close(fig)


def overlay(long5):
    """All five measurements on shared axes - shows the phase structure directly."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    cols = {1: "#4878A8", 2: "#6A9AC4", 3: "#8CB4D2", 4: "#B0CBE0", 5: "#55A868"}
    for j in range(1, 6):
        d = long5[long5.j == j]
        t = d.time.values.astype(float)
        y = d.efw.values.astype(float)
        ax.scatter(t, y, s=7, alpha=.22, color=cols[j], edgecolors="none")
        tf, yf, _, _ = trim_tails(t, y)
        s = lowess(tf, yf)
        ax.plot(s[0], s[1], lw=2.4,
                color="#C44E52" if j == 5 else "#2B4C6F",
                label=f"j={j} ({PHASE5[j]})")
    ax.set_xlabel(r"$t_{ij}$  (gestational days)")
    ax.set_ylabel("weight (g)")
    ax.set_title("All five measurements, shared axes "
                 "(green = birthweight at day of birth)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=.25, lw=.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/task4_overlay_all_measurements.png", dpi=160)
    plt.close(fig)


def summary(long5):
    rows = []
    for j in range(1, 6):
        d = long5[long5.j == j]
        t, y = d.time.values.astype(float), d.efw.values.astype(float)
        rows.append(dict(j=j, label=PHASE5[j], n=len(d),
                         t_min=int(t.min()), t_max=int(t.max()),
                         t_sd=round(t.std(ddof=1), 1),
                         ybar=round(y.mean(), 1), y_sd=round(y.std(ddof=1), 1),
                         cv=round(y.std(ddof=1) / y.mean(), 3),
                         ols_slope=round(float(np.polyfit(t, y, 1)[0]), 2)))
    s = pd.DataFrame(rows)
    s.to_csv(f"{OUT}/five_measurement_summary.csv", index=False)
    return s


if __name__ == "__main__":
    df, long5 = build()
    make5(long5, False, "task4_raw5",
          "Task 4 - Four fetal weights plus birthweight, by measurement occasion")
    make5(long5, True, "task4_centered5",
          r"Task 4 - Centered $Y_{ij}-\bar{y}_j$, five measurement occasions")
    overlay(long5)
    print(summary(long5).to_string(index=False))
