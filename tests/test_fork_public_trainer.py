import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fork_public_trainer as F  # noqa: E402

UPSTREAM = ('ARM_MODE = "e11"\nimport numpy as np\n'
            'def make_targets(train):\n    """three public teachers"""\n    return None\n\n\n'
            'def report_groups(train):\n    pass\n')
CELL32 = "if ARM_MODE == 'e11':\n    print('x')\n"


def _fake_pull(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "REPO", tmp_path)

    def run(cmd, check):
        d = Path(cmd[cmd.index("-p") + 1])
        d.mkdir(parents=True, exist_ok=True)
        (d / "u.ipynb").write_text(json.dumps({"cells": [{"cell_type": "code", "source": UPSTREAM},
                                                         {"cell_type": "code", "source": CELL32}]}))
        (d / "kernel-metadata.json").write_text(json.dumps(
            {"id": "a/b", "id_no": 1, "code_file": "u.ipynb", "enable_gpu": True,
             "dataset_sources": ["x/keep", "antoinegg1/rsna-knee-e9-radimagenet-heads-v15"]}))
    monkeypatch.setattr(F.subprocess, "run", run)


@pytest.mark.parametrize("kind", ["e13", "e9main"])
def test_fork_swaps_targets_and_metadata(tmp_path, monkeypatch, kind):
    _fake_pull(tmp_path, monkeypatch)
    d = F.fork(kind, "labels_bestof_v1.csv")
    nb = json.loads(next(d.glob("fork-*.ipynb")).read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert "OURS: image-informed" in src and "three public teachers" not in src
    assert "def report_groups(train)" in src
    meta = json.loads((d / "kernel-metadata.json").read_text())
    assert meta["is_private"] and meta["id"] == f"{F.USER}/{d.name}" and F.LABEL_DS in meta["dataset_sources"]
    if kind == "e9main":
        assert 'ARM_MODE = "e9main"' in src and "ARM_MODE in ('e11', 'e9main')" in src
        assert "antoinegg1/rsna-knee-e9-radimagenet-heads-v15" not in meta["dataset_sources"]
    else:
        assert "antoinegg1/rsna-knee-e9-radimagenet-heads-v15" in meta["dataset_sources"]


def test_our_make_targets_runs():
    import numpy as np
    import pandas as pd
    T = ["A", "B"]
    ours = pd.DataFrame({"StudyInstanceUID": ["1", "2"], "A": [0.05, 0.95], "B": [0.5, 0.2]})
    train = pd.DataFrame({"StudyInstanceUID": ["1", "2", "3"], "A": [None, None, 1.0], "B": [None, None, 0.0]})
    ns = {"pd": pd, "np": np, "TARGETS": T, "find_input_file": lambda n: n}
    ns["pd"] = type("P", (), {"read_csv": staticmethod(lambda *a, **k: ours)})
    exec(F.OUR_TARGETS.replace("__LABELS__", "x.csv"), ns)
    y, w, gold = ns["make_targets"](train)
    assert gold.tolist() == [False, False, True]
    assert np.allclose(y[:2], [[0.05, 0.5], [0.95, 0.2]]) and y[2].tolist() == [1.0, 0.0]
    assert (w[2] == 3.0).all() and np.isclose(w[0, 1], 0.15)
