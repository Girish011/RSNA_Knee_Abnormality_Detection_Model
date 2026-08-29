"""Clinical series ranking and selection."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rsna_knee.constants import PLANE_TO_ID


@dataclass(frozen=True)
class SeriesChoice:
    study_uid: str
    series_uid: str
    plane: str
    plane_id: int
    fluid_sensitive: int
    fat_suppression: int
    score: float


def _plane_name(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "Unknown"
    name = str(raw).strip()
    if name in PLANE_TO_ID:
        return name
    lower = name.lower()
    if "sag" in lower:
        return "Sagittal"
    if "cor" in lower:
        return "Coronal"
    if "ax" in lower:
        return "Axial"
    return "Unknown"


def score_series_row(row: pd.Series) -> float:
    """Higher is better. Prefer fluid-sensitive + fat-sup + known plane."""
    fluid = int(row.get("Fluid_Sensitive", 0) or 0)
    fat = int(row.get("Fat_Suppression", 0) or 0)
    plane = _plane_name(row.get("Anatomical_Plane"))
    score = 0.0
    score += 3.0 * fluid
    score += 2.0 * fat
    if plane != "Unknown":
        score += 1.0
    # Slight preference for sagittal (many key structures).
    if plane == "Sagittal":
        score += 0.3
    elif plane == "Coronal":
        score += 0.2
    elif plane == "Axial":
        score += 0.1
    return score


def rank_and_select_series(
    series_df: pd.DataFrame,
    study_uid: str,
    *,
    max_series: int = 4,
    require_plane_coverage: bool = True,
) -> list[SeriesChoice]:
    """Select top series for a study with optional plane diversity.

    Strategy:
    1. Rank by clinical utility score.
    2. If require_plane_coverage, greedily keep best series per Sag/Cor/Ax first.
    3. Fill remaining slots by global rank.
    """
    sub = series_df[series_df["StudyInstanceUID"] == study_uid].copy()
    if sub.empty:
        return []

    sub["_score"] = sub.apply(score_series_row, axis=1)
    sub["_plane"] = sub["Anatomical_Plane"].map(_plane_name)
    sub = sub.sort_values("_score", ascending=False)

    chosen: list[SeriesChoice] = []
    seen_series: set[str] = set()

    def _append(row: pd.Series) -> None:
        sid = str(row["SeriesInstanceUID"])
        if sid in seen_series:
            return
        if len(chosen) >= max_series:
            return
        plane = str(row["_plane"])
        chosen.append(
            SeriesChoice(
                study_uid=study_uid,
                series_uid=sid,
                plane=plane,
                plane_id=PLANE_TO_ID.get(plane, PLANE_TO_ID["Unknown"]),
                fluid_sensitive=int(row.get("Fluid_Sensitive", 0) or 0),
                fat_suppression=int(row.get("Fat_Suppression", 0) or 0),
                score=float(row["_score"]),
            )
        )
        seen_series.add(sid)

    if require_plane_coverage:
        for plane in ("Sagittal", "Coronal", "Axial"):
            plane_rows = sub[sub["_plane"] == plane]
            if not plane_rows.empty:
                _append(plane_rows.iloc[0])

    for _, row in sub.iterrows():
        if len(chosen) >= max_series:
            break
        _append(row)

    return chosen
