"""Tests for greenfield ligament gap-fill teacher."""

from __future__ import annotations

import pandas as pd

from rsna_knee.text.gf_ligament_teacher import (
    FOCUS_LABELS_LIG2,
    FOCUS_LABELS_LIG3,
    gap_fill_focus_labels,
)
from rsna_knee.text.weak_labels_v7 import extract_label_v7


def test_turkish_anterior_capraz_acl_positive():
    text = "Anterior çapraz bağda komplet rüptüre sekonder bütünlük seçilememiştir."
    r = extract_label_v7(text, "ACL")
    assert r.value == 1 and r.confidence >= 0.5, r


def test_english_mcl_grade_injury_positive():
    text = "There is a grade 2 injury of the MCL with adjacent soft tissue edema."
    r = extract_label_v7(text, "MCL")
    assert r.value == 1 and r.confidence >= 0.5, r


def test_gap_fill_only_nans():
    base = pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b"],
            "ACL": [1.0, pd.NA],
            "MCL": [pd.NA, pd.NA],
            "Medial Meniscus": [pd.NA, 0.0],
            "ACL__conf": [1.0, 0.0],
            "MCL__conf": [0.0, 0.0],
            "Medial Meniscus__conf": [0.0, 1.0],
        }
    )
    reports = pd.Series(
        [
            "Normal study.",
            "Complete ACL tear with discontinuity of fibers.",
        ],
        index=base.index,
    )
    out, stats = gap_fill_focus_labels(base, reports, min_confidence=0.5)
    assert out.loc[0, "ACL"] == 1.0  # unchanged
    assert out.loc[1, "ACL"] == 1.0  # filled
    assert stats["ACL_filled"] >= 1


def test_gap_fill_lig2_med_men_only_skips_acl_mcl():
    base = pd.DataFrame(
        {
            "StudyInstanceUID": ["a"],
            "ACL": [pd.NA],
            "MCL": [pd.NA],
            "Medial Meniscus": [pd.NA],
            "ACL__conf": [0.0],
            "MCL__conf": [0.0],
            "Medial Meniscus__conf": [0.0],
        }
    )
    reports = pd.Series(
        [
            "Complete ACL tear with discontinuity of fibers. "
            "There is a grade 2 injury of the MCL. "
            "Tear of the medial meniscus.",
        ],
        index=base.index,
    )
    out, stats = gap_fill_focus_labels(
        base, reports, min_confidence=0.5, focus_labels=FOCUS_LABELS_LIG2
    )
    assert pd.isna(out.loc[0, "ACL"])
    assert pd.isna(out.loc[0, "MCL"])
    assert out.loc[0, "Medial Meniscus"] == 1.0
    assert stats["Medial Meniscus_filled"] == 1
    assert "ACL_filled" not in stats
    assert "MCL_filled" not in stats


def test_gap_fill_lig3_skips_studies_with_no_existing_labels():
    """lig3 must not admit new studies into the training pool."""
    base = pd.DataFrame(
        {
            "StudyInstanceUID": ["already", "new"],
            "ACL": [1.0, pd.NA],
            "MCL": [pd.NA, pd.NA],
            "Medial Meniscus": [pd.NA, pd.NA],
            "Effusion": [pd.NA, pd.NA],
            "ACL__conf": [1.0, 0.0],
            "MCL__conf": [0.0, 0.0],
            "Medial Meniscus__conf": [0.0, 0.0],
            "Effusion__conf": [0.0, 0.0],
        }
    )
    # Eligible = already has any label (row 0 only).
    eligible = base[["ACL", "MCL", "Medial Meniscus", "Effusion"]].notna().any(axis=1)
    reports = pd.Series(
        ["Tear of the medial meniscus.", "Tear of the medial meniscus."],
        index=base.index,
    )
    out, stats = gap_fill_focus_labels(
        base,
        reports,
        min_confidence=0.5,
        focus_labels=FOCUS_LABELS_LIG3,
        eligible_mask=eligible,
    )
    assert out.loc[0, "Medial Meniscus"] == 1.0
    assert pd.isna(out.loc[1, "Medial Meniscus"])
    assert stats["Medial Meniscus_filled"] == 1
    assert stats["n_eligible"] == 1
    assert stats["n_skipped_ineligible"] == 1

