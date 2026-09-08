# Goal: pick one study ID and list every series it has (plane + flags). No selecting yet.
from pathlib import Path

import pandas as pd

DOWNLOADS = Path("/Users/girish11/Downloads")
train_series = pd.read_csv(DOWNLOADS / "train_series.csv")

# Pick the first study ID in the file (any one is fine for learning)
study_uid = train_series["StudyInstanceUID"].iloc[0]
print("example study:", study_uid)

# All series rows that belong to this one exam
one = train_series[train_series["StudyInstanceUID"] == study_uid].copy()
print("number of series for this study:", len(one))
print()
print(one[["SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]])

# Words to know
# filter with ==: keep only rows for that study
# .iloc[0]: first row
# .copy(): safe copy so later edits do not warn

# Goal: write a small function that returns one Sagittal + one Coronal + one Axial, preferring fluid/fat = 1.
def pick_one_per_plane(series_df: pd.DataFrame, study_uid: str) -> pd.DataFrame:
    """Choose 1 Sagittal, 1 Coronal, 1 Axial for one study.

    Preference: Fluid_Sensitive=1 (same as Fat_Suppression in this CSV).
    """
    one = series_df[series_df["StudyInstanceUID"] == study_uid].copy()
    chosen_rows = []

    for plane in ["Sagittal", "Coronal", "Axial"]:
        plane_rows = one[one["Anatomical_Plane"] == plane]
        if plane_rows.empty:
            print(f"WARNING: no {plane} series for this study")
            continue

        # Sort so preferred series rise to the top:
        # Fluid_Sensitive 1 before 0, then Fat_Suppression 1 before 0
        ranked = plane_rows.sort_values(
            by=["Fluid_Sensitive", "Fat_Suppression"],
            ascending=[False, False],
        )
        best = ranked.iloc[0]
        chosen_rows.append(best)

    return pd.DataFrame(chosen_rows)


study_uid = train_series["StudyInstanceUID"].iloc[0]
picked = pick_one_per_plane(train_series, study_uid)

print("picked series:")
print(
    picked[
        [
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Anatomical_Plane",
            "Fluid_Sensitive",
            "Fat_Suppression",
        ]
    ]
)

# Words to know
# function (def): reusable block of code with a name
# sort_values(..., ascending=False): bigger numbers first (1 before 0)
# loop for plane in ...: do the same idea three times

# Goal: check that all 4,407 studies get exactly 3 series (one per plane), and save a lookup table for later caching.
# Unique study IDs in train_series
all_studies = train_series["StudyInstanceUID"].unique()
print("\nrunning picker on", len(all_studies), "studies...")

rows = []
n_ok = 0
n_short = 0

for i, uid in enumerate(all_studies):
    picked = pick_one_per_plane(train_series, uid)
    n = len(picked)

    if n == 3:
        n_ok += 1
    else:
        n_short += 1

    for _, r in picked.iterrows():
        rows.append(
            {
                "StudyInstanceUID": uid,
                "SeriesInstanceUID": r["SeriesInstanceUID"],
                "Anatomical_Plane": r["Anatomical_Plane"],
                "Fluid_Sensitive": int(r["Fluid_Sensitive"]),
                "Fat_Suppression": int(r["Fat_Suppression"]),
            }
        )

    # Progress every 1000 studies so you know it is working
    if (i + 1) % 1000 == 0:
        print("  processed", i + 1)

picks = pd.DataFrame(rows)
print("studies with exactly 3 planes:", n_ok)
print("studies with fewer than 3 planes:", n_short)
print("total picked series rows:", len(picks))

out_dir = Path("outputs/eda")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "gf_v0_series_picks.csv"
picks.to_csv(out_csv, index=False)
print("wrote:", out_csv.resolve())

# Words to know
# unique(): distinct study IDs
# enumerate: loop with a counter i
# to_csv: save a table to a CSV file
# index=False: do not write the 0,1,2,... row numbers as a column

# Goal: prove the saved file is valid before we write a training config.
# Reload from disk (proves the file is readable)
picks_check = pd.read_csv(out_csv)

print("\n--- B.2 sanity checks ---")
print("rows:", len(picks_check))
print("unique studies:", picks_check["StudyInstanceUID"].nunique())
print("planes:")
print(picks_check["Anatomical_Plane"].value_counts())

# Each study should appear exactly 3 times
counts = picks_check.groupby("StudyInstanceUID").size()
print("series per study min/max:", counts.min(), counts.max())

# Within each study, planes should be unique (one sag, one cor, one ax)
dup_planes = (
    picks_check.groupby(["StudyInstanceUID", "Anatomical_Plane"]).size() > 1
).sum()
print("duplicate plane rows inside a study:", int(dup_planes))

# How often we preferred fluid-sensitive series
print(
    "Fluid_Sensitive=1 share:",
    round(picks_check["Fluid_Sensitive"].mean(), 3),
)