
import glob, json, os
import cv2, numpy as np, pandas as pd, pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
ROWS = [{'uid': '1.2.826.0.1.3680043.8.498.11599387556469808488160280438305325860', 'side': 'L', 'tag': 'R', 'cx': 87.3, 'group': 'discordant'}, {'uid': '1.2.826.0.1.3680043.8.498.82579118531566493365068788437906656162', 'side': 'R', 'tag': 'L', 'cx': -108.4, 'group': 'discordant'}, {'uid': '1.2.826.0.1.3680043.8.498.46172451572836863687073454997158931662', 'side': 'R', 'tag': 'L', 'cx': -77.3, 'group': 'discordant'}, {'uid': '1.2.826.0.1.3680043.8.498.26072622476167889001255150030772906081', 'side': 'L', 'tag': 'R', 'cx': 99.3, 'group': 'discordant'}, {'uid': '1.2.826.0.1.3680043.8.498.81327951282145199298572980264557740321', 'side': 'L', 'tag': 'R', 'cx': 110.3, 'group': 'discordant'}, {'uid': '1.2.826.0.1.3680043.8.498.74864145321761783409935288797327597738', 'side': 'R', 'tag': 'L', 'cx': -106.8, 'group': 'discordant'}, {'uid': '1.2.826.0.1.3680043.8.498.82053853036886620816821481139413594407', 'side': 'R', 'tag': 'L', 'cx': -88.4, 'group': 'discordant'}, {'uid': '1.2.826.0.1.3680043.8.498.33880013318702285028756482543453984598', 'side': 'L', 'tag': 'R', 'cx': 111.4, 'group': 'discordant'}, {'uid': '1.2.826.0.1.3680043.8.498.59475691595504590121311599301170361869', 'side': 'R', 'tag': 'L', 'cx': -68.2, 'group': 'discordant'}, {'uid': '1.2.826.0.1.3680043.8.498.72697172162233919540021967137047646179', 'side': 'L', 'tag': 'R', 'cx': 79.5, 'group': 'discordant'}, {'uid': '1.2.826.0.1.3680043.8.498.13103513835731355705707295729643672786', 'side': 'R', 'tag': 'L', 'cx': -78.8, 'group': 'discordant'}, {'uid': '1.2.826.0.1.3680043.8.498.58811528407321581206563412079643094538', 'side': 'L', 'tag': 'R', 'cx': 112.6, 'group': 'discordant'}, {'uid': '1.2.826.0.1.3680043.8.498.50342726517205151681842826322659275894', 'side': 'L', 'tag': 'L', 'cx': 82.7, 'group': 'concordant'}, {'uid': '1.2.826.0.1.3680043.8.498.75547515925579172661099437338637082554', 'side': 'L', 'tag': 'L', 'cx': 85.9, 'group': 'concordant'}, {'uid': '1.2.826.0.1.3680043.8.498.11180457573346204459956051956213528997', 'side': 'L', 'tag': 'L', 'cx': 105.0, 'group': 'concordant'}, {'uid': '1.2.826.0.1.3680043.8.498.10896245499713818848442117831482755673', 'side': 'L', 'tag': 'L', 'cx': 67.2, 'group': 'concordant'}, {'uid': '1.2.826.0.1.3680043.8.498.12907457183137517669505067507754394039', 'side': 'L', 'tag': 'L', 'cx': 107.7, 'group': 'concordant'}, {'uid': '1.2.826.0.1.3680043.8.498.22394208178597637441149549895512290013', 'side': 'L', 'tag': 'L', 'cx': 71.2, 'group': 'concordant'}, {'uid': '1.2.826.0.1.3680043.8.498.65013994414976103275014874284064145461', 'side': 'R', 'tag': 'R', 'cx': -90.2, 'group': 'concordant'}, {'uid': '1.2.826.0.1.3680043.8.498.63195088465290711580668024587233364997', 'side': 'R', 'tag': 'R', 'cx': -65.4, 'group': 'concordant'}, {'uid': '1.2.826.0.1.3680043.8.498.83271280040777227673434567660749430744', 'side': 'R', 'tag': 'R', 'cx': -84.8, 'group': 'concordant'}, {'uid': '1.2.826.0.1.3680043.8.498.85137704768728261642637391657270006448', 'side': 'R', 'tag': 'R', 'cx': -111.0, 'group': 'concordant'}, {'uid': '1.2.826.0.1.3680043.8.498.38528511947818361078957998546327084024', 'side': 'R', 'tag': 'R', 'cx': -90.1, 'group': 'concordant'}, {'uid': '1.2.826.0.1.3680043.8.498.10321556388278642027434955207617064548', 'side': 'R', 'tag': 'R', 'cx': -75.2, 'group': 'concordant'}, {'uid': '1.2.826.0.1.3680043.8.498.93096918345463865518663434763610016449', 'side': 'R', 'tag': '', 'cx': -3.8, 'group': 'ambiguous'}, {'uid': '1.2.826.0.1.3680043.8.498.30947630416867960170317316514422933066', 'side': 'R', 'tag': 'R', 'cx': 3.3, 'group': 'ambiguous'}, {'uid': '1.2.826.0.1.3680043.8.498.77290477043075828139630734968573461136', 'side': 'R', 'tag': 'R', 'cx': 2.7, 'group': 'ambiguous'}, {'uid': '1.2.826.0.1.3680043.8.498.94117540970936035429846288243624086693', 'side': 'R', 'tag': '', 'cx': -3.8, 'group': 'ambiguous'}, {'uid': '1.2.826.0.1.3680043.8.498.59938475193950583662125713924156951751', 'side': 'L', 'tag': 'L', 'cx': 2.6, 'group': 'ambiguous'}, {'uid': '1.2.826.0.1.3680043.8.498.46036209147177723970966724006242055397', 'side': 'R', 'tag': '', 'cx': -0.9, 'group': 'ambiguous'}]
root = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
if not os.path.exists(root + "/train.csv"):
    root = [d for d, _, f in os.walk("/kaggle/input") if "train_series.csv" in f][0]
ser = pd.read_csv(root + "/train_series.csv", dtype=str)
tiles, meta = [], []
for r in ROWS:
    s = ser[(ser.StudyInstanceUID == r["uid"]) & (ser.Anatomical_Plane == "Coronal")]
    if not len(s):
        continue
    sdir = f"{root}/train_series/{r['uid']}/{s.iloc[0].SeriesInstanceUID}"
    recs = []
    for f in glob.glob(sdir + "/*.dcm"):
        h = pydicom.dcmread(f, stop_before_pixels=True)
        recs.append((float(h.ImagePositionPatient[1]), f))     # coronal: y = anterior -> posterior
    recs.sort()
    row = []
    for q in (0.45, 0.6, 0.75):
        d = pydicom.dcmread(recs[int(q * (len(recs) - 1))][1])
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        lo, hi = np.percentile(a, [1, 99]); a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
        row.append(cv2.resize((a * 255).astype(np.uint8), (200, 200), interpolation=cv2.INTER_AREA))
    im = np.concatenate(row, 1)
    cv2.putText(im, f"{len(tiles)} {r['group'][:4]} rule={r['side']} tag={r['tag'] or '-'} cx={r['cx']:.0f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
    tiles.append(im); meta.append(dict(r, n=len(tiles) - 1))
for k in range(0, len(tiles), 6):
    cv2.imwrite(f"/kaggle/working/side_check_{k // 6}.png", np.concatenate(tiles[k:k + 6], 0))
json.dump(meta, open("/kaggle/working/side_check.json", "w"), indent=1)
print("wrote", len(tiles), "studies")
