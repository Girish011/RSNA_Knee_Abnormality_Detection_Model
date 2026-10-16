import pandas as pd, numpy as np, glob, os, json, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
LAB=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
s=pd.read_csv("data/external_labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"); s["StudyInstanceUID"]=s.StudyInstanceUID.astype(str); s=s.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
B=(s[LAB]>=0.5).astype(float)
pairs=[("Effusion","Synovitis"),("Contusion","Fracture"),("Medial OA","Lateral OA"),("Medial OA","PF OA"),("Lateral OA","PF OA"),("Medial Meniscus","Lateral Meniscus")]
def spec(ids,P):
    R=B.reindex(ids).values; ok=~np.isnan(R).any(1); R=R[ok]; P=P[ok]; out=[]
    for a,b in pairs:
        i,j=LAB.index(a),LAB.index(b); m=R[:,i]!=R[:,j]
        out.append(roc_auc_score(R[m,i],P[m,i]-P[m,j]))
    mac=np.mean([roc_auc_score(R[:,i],P[:,i]) for i in range(12)])
    return out,mac,len(R)
runs={}
for d in sorted(glob.glob("outputs/runpod_results/*/*/")):
    fs=[f for f in glob.glob(d+"raptor_oof_*_fold0.npz") if not any(k in f for k in ("_best_","_ema_","_ep","_last_","_swa_","epens"))]
    if not fs: continue
    try:
        z=np.load(fs[0],allow_pickle=True)
        if "pred" not in z.files or z["pred"].shape[1]!=12: continue
        lab="?"
        js=glob.glob(d+"raptor_ft_*.json")
        if js:
            j=json.load(open(js[0])); a=j.get("args",j)
            lab=os.path.basename(str(a.get("labels","?")))
        runs[d.split("/")[2]]=(spec([str(x) for x in z["ids"]],z["pred"]),lab)
    except Exception as e:
        print("skip",d,repr(e)[:80])
want=["r1-ksf-f0","r1-kadj-f0","r2-cnxs336-f0","r4-cnxs336-f0","r6-cnxs336-f0","r7-cnxs336-f0","r8-cnxs336-f0","r5-cnxs336-f0","r4-kcnxtmil-f0","r4-kcnxtseq-f0","r4-cnxsseq-f0","r4-cnxstok-f0","r2-effv2s336-f0","r2-c224hvy40-f0b","r2-swint224-f0","r2-dinos336-f0","r3-cnxsx-f0"]
print("fold 0, discordant-pair AUC vs report verdicts (0.5 = cannot tell the two findings apart)")
print(f"{'run':20s} {'labels':26s} n    Eff/Syn Con/Fra MOA/LOA MOA/PF  LOA/PF  MM/LM | macro-vs-report")
for k in want+[r for r in runs if r not in want][:30]:
    if k in runs:
        (o,mac,n),lab=runs[k]; print(f"{k:20s} {lab[:26]:26s} {n:4d} "+" ".join(f"{x:7.3f}" for x in o)+f" | {mac:.3f}")
