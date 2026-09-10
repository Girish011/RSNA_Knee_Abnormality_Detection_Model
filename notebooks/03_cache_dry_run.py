# Goal: for every study, print/save the planned output path and confirm 3 series are attached. Still CSV-only on your Mac

from pathlib import Path

import pandas as pd

ROOT = Path("/Users/girish11/RSNA_Knee_Abnormality_Detection_Model")
picks = pd.read_csv(ROOT / "outputs/eda/gf_v0_series_picks.csv")

# Pretend Kaggle cache folder (we only plan paths locally)
cache_dir = Path("/kaggle/working/cache_gf_v0")

print("picked rows:", len(picks))
print("studies:", picks["StudyInstanceUID"].nunique())

# Build one planned row per study
plans = []
for study_uid, group in picks.groupby("StudyInstanceUID"):
    planes = list(group["Anatomical_Plane"])
    series_uids = list(group["SeriesInstanceUID"])
    out_path = cache_dir / f"{study_uid}.npz"
    plans.append(
        {
            "StudyInstanceUID": study_uid,
            "n_series": len(group),
            "planes": ",".join(planes),
            "out_path": str(out_path),
        }
    )

plan_df = pd.DataFrame(plans)
print("planned cache files:", len(plan_df))
print("all have 3 series?", bool((plan_df["n_series"] == 3).all()))
print("\nexample plans:")
print(plan_df.head(3).to_string(index=False))

out = ROOT / "outputs/eda/gf_v0_cache_plan.csv"
out.parent.mkdir(parents=True, exist_ok=True)
plan_df.to_csv(out, index=False)
print("\nwrote:", out)

# Words to know
# dry-run: practice the steps without doing the expensive work
# .npz: compressed NumPy archive (common for ML caches)
# groupby: split the table by study, then loop each group