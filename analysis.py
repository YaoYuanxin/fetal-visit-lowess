"""
Fetal weight longitudinal data: per-visit LOWESS exploration.

Tasks (per Dr. Sinha, 2026-09):
  1. Y_ij(t_ij) vs t_ij with LOWESS, one figure per visit j=1..4
  2. Y_ij(t_ij) - ybar_j vs t_ij with LOWESS, one figure per visit j
  3. Both of the above restricted to subjects in the top 15th percentile of birthweight
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
FRAC = 0.5      # LOWESS bandwidth
NIT = 3         # robustness iterations
TRIM = 1.0      # drop t outside [TRIM, 100-TRIM] pctile within each visit


# ---------------------------------------------------------------- LOWESS
def lowess(x, y, frac=FRAC, it=NIT):
    """Cleveland (1979) locally-weighted linear regression with tricube
    weights and bisquare robustness iterations. Returns fitted y at sorted x.
    Mirrors statsmodels.nonparametric.smoothers_lowess.lowess."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    r = int(np.ceil(frac * n))
    r = max(r, 2)

    # distance to the r-th nearest neighbour for each point
    h = np.array([np.sort(np.abs(x - x[i]))[r - 1] for i in range(n)])
    h[h == 0] = np.finfo(float).eps

    w = np.clip(np.abs((x[:, None] - x[None, :]) / h[:, None]), 0.0, 1.0)
    w = (1 - w ** 3) ** 3                      # tricube

    yfit = np.zeros(n)
    delta = np.ones(n)                          # robustness weights

    for _ in range(it + 1):
        for i in range(n):
            wi = delta * w[i]
            sw = wi.sum()
            if sw <= 0:
                yfit[i] = y[i]
                continue
            mx = (wi * x).sum() / sw
            my = (wi * y).sum() / sw
            sxx = (wi * (x - mx) ** 2).sum()
            sxy = (wi * (x - mx) * (y - my)).sum()
            b = sxy / sxx if sxx > np.finfo(float).eps else 0.0
            yfit[i] = my + b * (x[i] - mx)
        res = y - yfit
        s = np.median(np.abs(res))
        if s <= np.finfo(float).eps:
            break
        delta = np.clip(res / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2           # bisquare

    return x, yfit


def _selftest():
    """Local linear LOWESS must reproduce an exact linear relationship."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 200)
    y = 3.0 * x - 7.0
    xs, ys = lowess(x, y)
    assert np.max(np.abs(ys - (3 * xs - 7))) < 1e-8, "lowess failed linear recovery"
    # monotone signal recovery under noise
    y2 = np.sin(x) + rng.normal(0, .05, 200)
    xs2, ys2 = lowess(x, y2, frac=.3)
    assert np.corrcoef(ys2, np.sin(xs2))[0, 1] > .99, "lowess failed curve recovery"
    print("lowess self-test passed")


# ---------------------------------------------------------------- data
def load():
    df = pd.read_csv("data/longitudinal_data_processed.csv")
    df = df.sort_values(["id", "time"]).reset_index(drop=True)
    df["j"] = df.groupby("id").cumcount() + 1
    assert set(df.groupby("id").size().unique()) == {4}
    return df


# ---------------------------------------------------------------- plotting
PHASE = {1: "~17 wk", 2: "~25 wk", 3: "~33 wk", 4: "~37 wk"}


def trim_tails(t, y, pct=TRIM):
    """LOWESS at the extremes is fit on 1-2 isolated points and swings wildly.
    Restrict the fit to the central mass of t within the visit."""
    lo, hi = np.percentile(t, [pct, 100 - pct])
    m = (t >= lo) & (t <= hi)
    return t[m], y[m], (lo, hi), (~m).sum()


def panel(ax, t, y, j, centered, n_subj, trim=True):
    ax.scatter(t, y, s=9, alpha=.30, color="#4878A8", edgecolors="none",
               label=f"observed (n={len(t)})")
    if trim:
        tf, yf, (lo, hi), ndrop = trim_tails(t, y)
        lab = f"LOWESS (frac={FRAC}, {TRIM}–{100-TRIM} pctile of t)"
    else:
        tf, yf, ndrop = t, y, 0
        lab = f"LOWESS (frac={FRAC}, untrimmed)"
    xs, ys = lowess(tf, yf)
    ax.plot(xs, ys, color="#C44E52", lw=2.2, label=lab)
    if ndrop:
        ax.scatter(t[(t < lo) | (t > hi)], y[(t < lo) | (t > hi)],
                   s=18, color="#999", marker="x", lw=.9,
                   label=f"excluded from fit ({ndrop})")
    if centered:
        ax.axhline(0, color="#555", lw=.8, ls="--", zorder=0)
        ylab = r"$Y_{ij}(t_{ij}) - \bar{y}_j$  (g)"
    else:
        ylab = r"$Y_{ij}(t_{ij})$  (g)"
    ax.set_xlabel(r"$t_{ij}$  (gestational days)")
    ax.set_ylabel(ylab)
    ax.set_title(f"Visit j = {j}  ({PHASE[j]})", fontsize=11)
    ax.legend(fontsize=8, framealpha=.9)
    ax.grid(alpha=.25, lw=.5)


def make_set(df, centered, tag, suptitle, trim=True, singles=True):
    """One standalone figure per visit, plus a 2x2 composite."""
    paths = []
    fig_all, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    for j in range(1, 5):
        d = df[df.j == j]
        t = d["time"].values
        y = d["efw"].values
        if centered:
            y = y - y.mean()

        if singles:
            fig, ax = plt.subplots(figsize=(7, 5))
            panel(ax, t, y, j, centered, d.id.nunique(), trim)
            fig.tight_layout()
            p = f"{OUT}/{tag}_visit{j}.png"
            fig.savefig(p, dpi=160)
            plt.close(fig)
            paths.append(p)

        panel(axes.flat[j - 1], t, y, j, centered, d.id.nunique(), trim)

    fig_all.suptitle(suptitle, fontsize=13, y=.995)
    fig_all.tight_layout()
    pc = f"{OUT}/{tag}_composite.png"
    fig_all.savefig(pc, dpi=150)
    plt.close(fig_all)
    return paths, pc


def main():
    _selftest()
    df = load()

    # top 15th percentile of birthweight, one bw per subject
    bw_subj = df.groupby("id")["bw"].first()
    thresh = bw_subj.quantile(0.85)
    top_ids = bw_subj[bw_subj >= thresh].index
    top = df[df.id.isin(top_ids)]

    print(f"n subjects total : {df.id.nunique()}")
    print(f"bw 85th pctile   : {thresh:.0f} g")
    print(f"n subjects top15 : {top.id.nunique()} "
          f"({100*top.id.nunique()/df.id.nunique():.1f}%)")

    make_set(df,  False, "task1_raw",
             "Task 1 — Fetal weight vs gestational day, by visit (all subjects)")
    make_set(df,  True,  "task2_centered",
             r"Task 2 — Visit-centered fetal weight $Y_{ij}-\bar{y}_j$ (all subjects)")
    make_set(top, False, "task3a_raw_top15",
             f"Task 3a — Fetal weight vs gestational day, top 15% birthweight "
             f"(bw ≥ {thresh:.0f} g, n={top.id.nunique()})")
    make_set(top, True,  "task3b_centered_top15",
             r"Task 3b — Visit-centered $Y_{ij}-\bar{y}_j$, top 15% birthweight "
             f"(bw ≥ {thresh:.0f} g, n={top.id.nunique()})")

    # untrimmed comparison, composites only — shows why the trim is needed
    make_set(df, False, "appendix_untrimmed_all",
             "Appendix — untrimmed LOWESS, all subjects (tail artifacts visible)",
             trim=False, singles=False)
    make_set(top, False, "appendix_untrimmed_top15",
             "Appendix — untrimmed LOWESS, top 15% birthweight",
             trim=False, singles=False)

    # summary table of within-visit slope, useful for the 4-model argument
    rows = []
    for lbl, d in (("all", df), ("top15", top)):
        for j in range(1, 5):
            g = d[d.j == j]
            t, y = g["time"].values, g["efw"].values
            slope = np.polyfit(t, y, 1)[0]
            rows.append(dict(cohort=lbl, j=j, n=len(g),
                             t_min=t.min(), t_max=t.max(),
                             t_sd=round(t.std(ddof=1), 1),
                             ybar=round(y.mean(), 1),
                             y_sd=round(y.std(ddof=1), 1),
                             ols_slope_g_per_day=round(slope, 2)))
    summ = pd.DataFrame(rows)
    summ.to_csv(f"{OUT}/visit_phase_summary.csv", index=False)
    print()
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
