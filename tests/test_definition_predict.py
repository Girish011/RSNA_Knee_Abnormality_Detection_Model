"""Deployment contract checks; no GPU or private images required."""
import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runpod"))
import definition_cache as cache
import definition_predict as runner


def original_cache():
    source = (ROOT / "outputs/kernels/dense512_cache/dense512_cache_template.py").read_text()
    source = source.replace("__SHARD__", "0").replace("__N_SHARDS__", "8")
    ns = {"__name__": "cache_parity_reference"}
    exec(compile(source, "cache_parity_reference", "exec"), ns)
    return ns


def test_pixel_parity_and_sequence_fallback():
    ref = original_cache()
    rng = np.random.default_rng(9)
    for shape, spacing in [((256, 320), .7), ((512, 512), .3), ((640, 480), .42)]:
        a = rng.uniform(0, 1, shape).astype(np.float32)
        actual, ag = cache.mm_crop_resize(a, spacing)
        expected, eg = ref["mm_crop_resize"](a, spacing)
        np.testing.assert_array_equal(actual, expected)
        assert ag == eg
    rows = [{"SeriesInstanceUID": "a", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 0},
            {"SeriesInstanceUID": "b", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1}]
    for used in [set(), {"b"}, {"a", "b"}]:
        assert cache.pick_series(rows, "Sagittal", 1, used) == ref["pick_series"](rows, "Sagittal", 1, used)


def test_fill_matches_training_normalization_and_geometry():
    ref = original_cache()
    rng = np.random.default_rng(1)
    pixels = {0: rng.normal(300, 100, (300, 350)), 1: None, 2: rng.normal(700, 200, (300, 350))}
    files = [("unused", .5, np.arange(20, dtype=np.float32)) for _ in range(3)]
    a = np.zeros((80,512,512), np.uint8); b = a.copy()
    ag = np.zeros((80,20), np.float32); bg = ag.copy()
    cache.fill(a, ag, 22, [0,1,2], pixels, files, .5)
    ref["fill"](b, bg, 22, [0,1,2], pixels, files, .5)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(ag,bg)
    assert not a[23].any()


def test_empty_chunk_cannot_silently_submit(tmp_path):
    built = [(np.zeros((80,32,32), np.uint8), np.zeros(80,np.uint8), np.zeros((80,20)), [], "")]
    with pytest.raises(RuntimeError, match="no usable"):
        runner.write_chunk(tmp_path, ["one"], built)


def test_submission_alignment_and_probabilities():
    frame = pd.DataFrame(np.full((2,12),.5), columns=runner.reader.LAB)
    frame.insert(0,"StudyInstanceUID",["a","b"])
    runner.validate_submission(frame,["a","b"])
    with pytest.raises(ValueError, match="order"):
        runner.validate_submission(frame,["b","a"])
    frame.iloc[0,1] = np.nan
    with pytest.raises(ValueError, match="probabilities"):
        runner.validate_submission(frame,["a","b"])
