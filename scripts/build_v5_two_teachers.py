#!/usr/bin/env python3
"""Base notebook = the forked 0.950 notebook (`outputs/kernels/pub/rsna-versia-5`) + the SECOND expert-trained
checkpoint of the same attached dataset (224 px "efficiency" model, 0.945 alone per its README) as another arm of the
notebook's own weighted rank-mean. Nothing else changes (ConvNeXt reader fusion, K 94, mirror pass of the 384 arm).

    python scripts/build_v5_two_teachers.py --mode B --w224 0.35            # writes outputs/kernels/_base_v5t2-B-w35/
    python scripts/build_ensemble_kernel.py --name v5t2-b-w35 --base outputs/kernels/_base_v5t2-B-w35   # plain candidate
    (add --members / --tail / --negate as for any other base)

--mode is the 224 arm's input, from the 384 px / 140 mm corpus the notebook builds: A = full field resized to 224,
B = central 0.8 resized to 224, C = central 224 px. Pick it from `scripts/teacher_dump_read.py`. The 224 arm is scored
on one view (no mirror pass), as its README states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "outputs" / "kernels" / "pub" / "rsna-versia-5"
F384 = "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
F224 = "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt"
SHA224 = "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6"

EDITS = [
    # second arm
    ('    {"file": "' + F384 + '", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},\n',
     '    {"file": "' + F384 + '", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": __W384__},\n'
     '    {"file": "' + F224 + '", "arch": "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", "res": 224, "w": __W224__},\n'),
    # backbone at the arm's own input size
    ("def build_backbone(arch, pretrained=False):\n", "def build_backbone(arch, pretrained=False, img_size=None):\n"),
    ("    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3, img_size=INPUT_RES)\n",
     "    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3, img_size=img_size or INPUT_RES)\n"),
    ("    assert observed_sha == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', \"Bound checkpoint bytes differ\"\n",
     "    assert observed_sha in ('7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', '" + SHA224 + "'), \"Bound checkpoint bytes differ\"\n"),
    ("    assert int(ck.get(\"res\", res_default)) == 384, \"Expected original 384 checkpoint\"\n    ck_res = INPUT_RES\n"
     "    bb = build_backbone(arch, pretrained=False)\n",
     "    ck_res = INPUT_RES if int(ck.get(\"res\", res_default)) == 384 else 224   # 224 = the second (efficiency) arm\n"
     "    bb = build_backbone(arch, pretrained=False, img_size=ck_res)\n"),
    # the 224 arm: one view, its own input
    ("                for a, (model, res) in enumerate(models[d]):\n",
     "                for a, (model, res) in enumerate(models[d]):\n"
     "                    if res == 224:\n"
     "                        arm_probs[a][i] = infer_probs(model, win224(vol, mask, K_EVAL, d), d)\n"
     "                        continue\n"),
    ("def rankpct(x):                                   # per-column percentile rank in [0,1]\n",
     "MODE224 = __MODE__\n"
     "def win224(vol, mask, k, dev):\n"
     "    \"\"\"Windows of the 224 px arm from the 384 px corpus: A = full field, B = central 0.8, C = central 224 px.\"\"\"\n"
     "    D = vol.shape[0]\n"
     "    cs = np.array([max(1, min(c, D - 2)) for c in _eval_centers(mask, D, k)])\n"
     "    x = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev).float() / 255.0\n"
     "    n = {'A': IMG, 'B': 307, 'C': 224}[MODE224]\n"
     "    lo = (IMG - n) // 2\n"
     "    x = x[..., lo:lo + n, lo:lo + n]\n"
     "    if n != 224:\n"
     "        x = F.interpolate(x, size=(224, 224), mode='bilinear', antialias=True, align_corners=False)\n"
     "    return (x - _MEAN.to(dev)) / _STD.to(dev)\n\n\n"
     "def rankpct(x):                                   # per-column percentile rank in [0,1]\n"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["A", "B", "C"])
    ap.add_argument("--w224", type=float, required=True, help="rank weight of the 224 arm inside the CoAtNet rank-mean")
    a = ap.parse_args()
    assert 0.0 < a.w224 < 1.0
    nb_path = next(SRC.glob("*.ipynb"))
    nb = json.loads(nb_path.read_text())
    idx = max((i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"), key=lambda i: len("".join(nb["cells"][i]["source"])))
    s = "".join(nb["cells"][idx]["source"])
    for old, new in EDITS:
        assert s.count(old) == 1, f"expected exactly one match for: {old[:70]!r} (found {s.count(old)})"
        s = s.replace(old, new)
    s = s.replace("__W384__", repr(round(1 - a.w224, 4))).replace("__W224__", repr(a.w224)).replace("__MODE__", repr(a.mode))
    compile(s, "cell", "exec")
    nb["cells"][idx]["source"] = s.splitlines(keepends=True)
    d = REPO / "outputs" / "kernels" / f"_base_v5t2-{a.mode}-w{int(round(a.w224 * 100)):02d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / nb_path.name).write_text(json.dumps(nb, indent=1))
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.pop("id_no", None)
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {d} | 224 arm mode {a.mode} weight {a.w224}")


if __name__ == "__main__":
    main()
