"""Competition constants and shared schema."""

from __future__ import annotations

# Exact submission column order (excluding StudyInstanceUID).
LABEL_COLS: list[str] = [
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

NUM_LABELS: int = len(LABEL_COLS)

PLANE_TO_ID: dict[str, int] = {
    "Sagittal": 0,
    "Coronal": 1,
    "Axial": 2,
    "Unknown": 3,
}

ID_TO_PLANE: dict[int, str] = {v: k for k, v in PLANE_TO_ID.items()}

# Preferred planes per label (soft routing prior).
LABEL_PLANE_PRIOR: dict[str, list[str]] = {
    "ACL": ["Sagittal", "Coronal"],
    "MCL": ["Coronal", "Axial"],
    "Medial Meniscus": ["Sagittal", "Coronal"],
    "Lateral Meniscus": ["Sagittal", "Coronal"],
    "Medial OA": ["Coronal", "Sagittal"],
    "Lateral OA": ["Coronal", "Sagittal"],
    "PF OA": ["Axial", "Sagittal"],
    "Effusion": ["Sagittal", "Axial"],
    "Synovitis": ["Sagittal", "Axial"],
    "Baker's": ["Axial", "Sagittal"],
    "Contusion": ["Sagittal", "Coronal"],
    "Fracture": ["Sagittal", "Coronal", "Axial"],
}

SUBMISSION_ID_COL = "StudyInstanceUID"
