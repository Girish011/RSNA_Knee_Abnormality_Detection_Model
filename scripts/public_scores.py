#!/usr/bin/env python3
"""Morning check: REAL best public score of every public notebook + tie blocks of the public leaderboard.

    python scripts/public_scores.py [out.json]

Read-only, no login: the same endpoints Kaggle's public pages load (ListKernels carries `bestPublicScore`,
GetLeaderboard the displayed scores). Titles overstate ("LB 0.954" in a title is not a score) and
`kaggle kernels list --sort-by scoreDescending` gives the order without the numbers. Method from the public notebook
varshithakolupuri/rsna-knee-real-scores-of-every-public-notebook (12 Oct). A notebook whose score needs a private
dataset of its author (velci-kcd25-20) scores lower as a fork: check the inputs in its kernel-metadata.json.
"""
import json, sys, time, collections
import requests
COMP = "rsna-knee-abnormality-detection"
s = requests.Session()
s.get("https://www.kaggle.com/competitions", timeout=60)
h = {"x-xsrf-token": s.cookies.get("XSRF-TOKEN", "")}
def api(service, method, body, tries=4):
    for a in range(tries):
        r = s.post(f"https://www.kaggle.com/api/i/{service}/{method}", json=body, headers=h, timeout=90)
        if r.status_code == 200:
            return r.json()
        time.sleep(3 * (a + 1))
    raise RuntimeError(f"{method}: HTTP {r.status_code}")
comp = api("competitions.CompetitionService", "GetCompetition", {"competitionName": COMP})
comp = comp.get("competition", comp)
print(comp["title"], "| teams", comp["totalTeams"], "| deadline", comp["deadline"][:16])
lb = api("competitions.LeaderboardService", "GetLeaderboard", {"competitionId": comp["id"]})["publicLeaderboard"]
n = int(comp["totalTeams"])
cuts = {"gold": 10 + n // 500, "silver": n // 20, "bronze": n // 10}
blocks = collections.OrderedDict()
for r in lb:
    blocks.setdefault(r["displayScore"], []).append(r["rank"])
print("cuts:", {m: (rk, next((r["displayScore"] for r in lb if r["rank"] == rk), None)) for m, rk in cuts.items()})
for sc, rk in list(blocks.items())[:16]:
    print(f"  {sc}: {len(rk):5d} teams, ranks {min(rk)}-{max(rk)}")
raw = {}
for sort_by in ("SCORE_DESCENDING", "DATE_RUN"):
    for page in range(1, 4):
        body = {"kernelFilterCriteria": {"search": "", "listRequest": {"competitionId": comp["id"], "sortBy": sort_by,
                "pageSize": 100, "group": "EVERYONE", "page": page}},
                "detailFilterCriteria": {"deletedAccessBehavior": "RETURN_NOTHING",
                "unauthorizedAccessBehavior": "RETURN_NOTHING", "excludeResultsFilesOutputs": True},
                "readMask": "pinnedKernels"}
        got = api("kernels.KernelsService", "ListKernels", body).get("kernels", [])
        if not got:
            break
        for k in got:
            raw[k["scriptUrl"]] = k
rows = [(k.get("bestPublicScore"), k.get("totalVotes") or 0, str(k.get("scriptVersionDateCreated", ""))[:16],
         k["scriptUrl"].removeprefix("/code/")) for k in raw.values() if k.get("bestPublicScore") is not None]
rows.sort(key=lambda r: (-r[0], r[2]))
print(len(raw), "notebooks read,", len(rows), "scored")
for sc, v, d, u in rows[:45]:
    print(f"  {sc:.3f}  votes {v:4d}  {d}  {u}")
if len(sys.argv) > 1:
    json.dump(rows, open(sys.argv[1], "w"))
