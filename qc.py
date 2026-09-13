import numpy as np
import pandas as pd

df = pd.read_csv("data/longitudinal_data_processed.csv")
df = df.sort_values(["id", "time"]).reset_index(drop=True)
df["j"] = df.groupby("id").cumcount() + 1

print("=" * 68)
print("STRUCTURE")
print("=" * 68)
print(f"records                    : {len(df)}")
print(f"subjects                   : {df.id.nunique()}")
sz = df.groupby("id").size()
print(f"records per subject        : min {sz.min()}, max {sz.max()}, "
      f"unique {sorted(sz.unique())}")
print(f"subjects with != 4 records : {(sz != 4).sum()}")
print(f"1120 x 4                   : {1120*4} vs {len(df)} -> "
      f"{'exact' if 1120*4 == len(df) else 'MISMATCH'}")
print(f"duplicate (id, time) rows  : {df.duplicated(['id','time']).sum()}")
print(f"duplicate full rows        : {df.duplicated().sum()}")

print()
print("=" * 68)
print("MISSINGNESS")
print("=" * 68)
na = df.isna().sum()
print(f"total nulls across all {df.shape[1]} columns: {int(na.sum())}")
if na.sum():
    print(na[na > 0].to_string())

print()
print("=" * 68)
print("VISIT TIMING  (t_ij, gestational days)")
print("=" * 68)
tt = df.groupby("j")["time"].agg(["count", "min", "max", "mean", "std"]).round(1)
tt["p1"] = df.groupby("j")["time"].quantile(.01).round(0)
tt["p99"] = df.groupby("j")["time"].quantile(.99).round(0)
print(tt.to_string())

# do visit windows overlap between adjacent j?
print()
for j in range(1, 4):
    a = df[df.j == j]["time"]
    b = df[df.j == j + 1]["time"]
    ov = (a.max() >= b.min())
    print(f"  j={j} max {a.max():>3.0f}  vs  j={j+1} min {b.min():>3.0f}   "
          f"-> {'OVERLAP' if ov else 'disjoint'}")

# within-subject time strictly increasing?
bad_t = df.groupby("id")["time"].apply(lambda s: (s.diff().dropna() <= 0).any()).sum()
print(f"\nsubjects with non-increasing t : {bad_t}")

print()
print("=" * 68)
print("FETAL WEIGHT  (efw, g)")
print("=" * 68)
ee = df.groupby("j")["efw"].agg(["count", "min", "max", "mean", "std"]).round(1)
ee["cv"] = (ee["std"] / ee["mean"]).round(3)
print(ee.to_string())
print(f"\nnon-positive efw               : {(df.efw <= 0).sum()}")
bad_e = df.groupby("id")["efw"].apply(lambda s: (s.diff().dropna() <= 0).any()).sum()
print(f"subjects with non-increasing efw: {bad_e}")

print()
print("=" * 68)
print("BIRTHWEIGHT  (bw, g) -- subject level")
print("=" * 68)
nun = df.groupby("id")["bw"].nunique()
print(f"subjects with >1 distinct bw   : {(nun != 1).sum()}  "
      f"(constant within subject: {'yes' if (nun == 1).all() else 'NO'})")
bw = df.groupby("id")["bw"].first()
print(f"n                              : {len(bw)}")
print(f"min / max                      : {bw.min():.0f} / {bw.max():.0f} g")
print(f"mean / sd                      : {bw.mean():.0f} / {bw.std():.0f} g")
for q in [.10, .25, .50, .75, .85, .90, .95]:
    print(f"  p{int(q*100):<3}                        : {bw.quantile(q):.0f} g")
print(f"macrosomia (bw > 4000 g)       : {(bw > 4000).sum()} "
      f"({100*(bw > 4000).mean():.1f}%)")
print(f"bw < 2500 g                    : {(bw < 2500).sum()} "
      f"({100*(bw < 2500).mean():.1f}%)")

print()
print("=" * 68)
print("INTERNAL CONSISTENCY")
print("=" * 68)
bd = df.groupby("id")["birthday"].nunique()
print(f"subjects with >1 birthday      : {(bd != 1).sum()}")
print(f"rows with t_ij > birthday      : {(df.time > df.birthday).sum()}")
print(f"rows with efw > bw             : {(df.efw > df.bw).sum()}")
last = df[df.j == 4]
print(f"  of which at j=4              : {(last.efw > last.bw).sum()} "
      f"({100*(last.efw > last.bw).mean():.1f}% of final visits)")
gap = (last.birthday.values - last.time.values)
print(f"days from last scan to birth   : min {gap.min()}, median "
      f"{np.median(gap):.0f}, max {gap.max()}")
print(f"  negative gaps                : {(gap < 0).sum()}")

print()
print("=" * 68)
print("DERIVED COLUMNS")
print("=" * 68)
print(f"time_2 == time^2               : "
      f"{'yes' if (df.time_2 == df.time**2).all() else 'NO'}")
for c in ["wt_before_preg_scaled", "height_scaled"]:
    print(f"{c:<24} mean {df[c].mean():+.4f}  sd {df[c].std():.4f}")

bin_cols = ["NoPrevPreg", "hpb", "cardiac", "baseline_diabetes", "renal", "reg_smoke"]
print()
for c in bin_cols:
    vals = sorted(df[c].unique())
    rate = df.groupby("id")[c].first().mean()
    print(f"{c:<20} values {str(vals):<12} subject-level mean {rate:.3f}")
