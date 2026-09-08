from pathlib import Path
import pandas as pd

DOWNLOADS = Path("/Users/girish11/Downloads")

train = pd.read_csv(DOWNLOADS / "train.csv")
train_series = pd.read_csv(DOWNLOADS / "train_series.csv")

# .shape means (number of rows, number of columns)
print("train shape:", train.shape)
print("train_series shape:", train_series.shape)

print("train columns:", list(train.columns))
print("train_series columns:", list(train_series.columns))

# StudyInstanceUID = unique ID for one knee exam
n_studies = train["StudyInstanceUID"].nunique()
n_series = train_series["SeriesInstanceUID"].nunique()

print("unique train studies:", n_studies)
print("unique train series:", n_series)

# How many series belong to each study?
series_per_study = train_series.groupby("StudyInstanceUID").size()

print("series per study — min:", series_per_study.min())
print("series per study — mean:", round(series_per_study.mean(), 2))
print("series per study — max:", series_per_study.max())


# Goal: count how many series are sagittal / coronal / axial.

# Plane = which way the scanner looks through the knee:
# Sagittal = side view
# Coronal = front view
# Axial = top-down view
# value_counts = how often each plane name appears
plane_counts = train_series["Anatomical_Plane"].value_counts(dropna=False)

print("\nseries by Anatomical_Plane:")
print(plane_counts)

# Optional: share of all series (as percent)
plane_pct = (plane_counts / plane_counts.sum() * 100).round(1)
print("\npercent of series by plane:")
print(plane_pct)

# Words to know
# value_counts: tally of each unique value in a column
# dropna=False: also show missing/blank plane values if any

# Goal: see how often series are marked fluid-sensitive and/or fat-suppressed.

# In MRI:
# Fluid-sensitive = sequence that makes fluid (like joint fluid) stand out
# Fat_Suppression = technique that darkens fat so other tissues are clearer
# These are True/False (or 1/0) flags on each series row.
print("\nFluid_Sensitive value counts:")
print(train_series["Fluid_Sensitive"].value_counts(dropna=False))

print("\nFat_Suppression value counts:")
print(train_series["Fat_Suppression"].value_counts(dropna=False))

# Both flags at once (cross-tab)
# crosstab = count rows for every combination of two columns
both = pd.crosstab(
    train_series["Fluid_Sensitive"],
    train_series["Fat_Suppression"],
    dropna=False,
)
print("\nFluid_Sensitive vs Fat_Suppression:")
print(both)

# Words to know
# crosstab: a small table that counts every combo of two columns (for example True+True, True+False)

# Goal: count how many training exams have full expert labels (the “gold” 58).

# Most rows in train.csv have empty label cells. Only a small set has all 12 labels filled.
# The 12 findings we must predict
LABEL_COLS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# notna() = cell is not empty (has a real value)
# .all(axis=1) = True only when ALL 12 labels are filled on that row
is_gold = train[LABEL_COLS].notna().all(axis=1)
gold = train.loc[is_gold]

print("\nexpert-gold studies:", len(gold))
print("percent of train:", round(100 * len(gold) / len(train), 2))

# How common is each finding in the gold set? (mean of 0/1 = positive rate)
print("\ngold positive rate per label:")
print(gold[LABEL_COLS].mean().round(3))

# Words to know
# notna: “this cell has a value”
# gold / expert set: the 58 exams checked by radiologists
# positive rate: share of exams where the label is 1 (present)

# Goal: check whether any exam is missing a side-view (sagittal) sequence. That matters for ACL and meniscus.

# For each study, collect the set of planes it has
planes_by_study = (
    train_series.groupby("StudyInstanceUID")["Anatomical_Plane"]
    .apply(lambda planes: set(planes))
)

has_sagittal = planes_by_study.apply(lambda planes: "Sagittal" in planes)
has_coronal = planes_by_study.apply(lambda planes: "Coronal" in planes)
has_axial = planes_by_study.apply(lambda planes: "Axial" in planes)

print("\nstudies with at least one Sagittal:", int(has_sagittal.sum()), "/", len(has_sagittal))
print("studies with at least one Coronal:", int(has_coronal.sum()), "/", len(has_coronal))
print("studies with at least one Axial:", int(has_axial.sum()), "/", len(has_axial))

missing_sag = (~has_sagittal).sum()
print("studies missing Sagittal:", int(missing_sag))

# Words to know
# set: a bag of unique items (here: plane names for one exam)
# in: checks membership ("Sagittal" in planes)
# ~: means “not” for True/False Series

# Create an output folder inside the project (safe if it already exists)
out_dir = Path("outputs/eda")
out_dir.mkdir(parents=True, exist_ok=True)

summary_path = out_dir / "step_a_summary.txt"

summary = f"""RSNA Knee — Step A data audit (metadata only)
train studies: {n_studies}
train series: {n_series}
series per study: min={series_per_study.min()}, mean={series_per_study.mean():.2f}, max={series_per_study.max()}
planes (series counts): {plane_counts.to_dict()}
expert-gold studies: {len(gold)} ({100 * len(gold) / len(train):.2f}% of train)
studies with Sagittal/Coronal/Axial: {int(has_sagittal.sum())}/{int(has_coronal.sum())}/{int(has_axial.sum())} of {len(has_sagittal)}
Fluid_Sensitive==Fat_Suppression always paired: True in this CSV
"""

summary_path.write_text(summary)
print("\nwrote:", summary_path.resolve())
print(summary)

import matplotlib.pyplot as plt

# Folder for blog images
assets_dir = Path("site/assets")
assets_dir.mkdir(parents=True, exist_ok=True)

# ----- Plot 1: series by plane -----
# Sort planes in a fixed order so the chart is easy to read
plane_order = ["Sagittal", "Coronal", "Axial"]
plane_for_plot = plane_counts.reindex(plane_order)

fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.bar(plane_for_plot.index, plane_for_plot.values)
ax1.set_title("Train series by anatomical plane")
ax1.set_xlabel("Plane (view direction)")
ax1.set_ylabel("Number of series")
fig1.tight_layout()

plane_png = assets_dir / "eda_planes.png"
fig1.savefig(plane_png, dpi=150)
plt.close(fig1)
print("wrote:", plane_png.resolve())

# ----- Plot 2: gold positive rates -----
gold_rates = gold[LABEL_COLS].mean().sort_values(ascending=True)

fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.barh(gold_rates.index, gold_rates.values)
ax2.set_title("Expert-gold positive rate (n=58)")
ax2.set_xlabel("Share of gold exams where label = 1")
ax2.set_xlim(0, 1)
fig2.tight_layout()

gold_png = assets_dir / "eda_gold_rates.png"
fig2.savefig(gold_png, dpi=150)
plt.close(fig2)
print("wrote:", gold_png.resolve())

# Words to know
# bar / barh: vertical / horizontal bar chart
# savefig: write the chart to a .png image file
# dpi: image sharpness (150 is fine for a blog)
# plt.close: free memory after saving (avoids leftover windows)