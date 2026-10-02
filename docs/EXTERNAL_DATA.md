# External data — what is allowed, what is worth it (checked 2026-10-02)

## Rules (Kaggle rules §6 + host rulings in Discussion)
- §6a: external data allowed if publicly available and equally accessible at no cost.
- Host (Po-Hao Chen, topic 733965): non-commercial licences are NOT excluded by themselves; prize money is not
  commercial use. "Straightforward registration or click-through data-use agreement" generally OK. Datasets needing
  institution-specific approval, negotiated agreements, IRB or lengthy credentialing may be NOT allowed.
- Host ruling (topic 734109): **KneeCoT not allowed** (formal agreement with the hospital).
- Commercial LLM APIs for report labels: explicitly allowed (topic 733965) → our DeepSeek labels are fine.
- Open question (topic 743416, 56 votes, no host answer yet): per-dataset yes/no for MRNet, SKM-TEA, fastMRI(+),
  OAI, KneeMRI. Winners must release code + weights → "no-derivatives" licences are a prize-eligibility risk.
- Teams are responsible for each dataset's own licence (incl. releasing trained weights).

## Candidates
| Dataset | Access | Labels vs our 12 | Risk for a prize-eligible finish | Verdict |
|---|---|---|---|---|
| **fastMRI knee + fastMRI+** (NYU; annotations MIT on GitHub) | online application + data-sharing agreement, free | 974 volumes, 16,168 boxes: Meniscus tear 5,658, Cartilage partial 2,985 / full 615, Effusion 1,311, Fracture/Contusion 1,060, Subchondral edema 986, Periarticular cysts 864, ACL low/high 1,443, MCL 296, … (coronal PD/PDFS) | low–medium (click-through; DSA forbids redistributing data, weights not addressed) | **register** |
| fastMRI knee DICOM (10,000 clinical exams, same application) | same | none (unlabelled) — self-supervised pretraining only | low–medium | same registration covers it |
| SKM-TEA (Stanford AIMI) | AIMI login + research-use agreement | 155 qDESS scans, boxes: meniscal/ligament tears, cartilage lesions, effusion | medium (licence terms on derivatives) | optional |
| MRNet (Stanford) | research-use agreement | 1,370 exams: abnormal / ACL / meniscus | **high**: agreement forbids derivative works | skip |
| KneeMRI (Rijeka) | open download | ACL only | **high**: CC BY-NC-ND | skip |
| OAI (NIH NDA) | Data Use Certification via an institution with FWA | MOAKS: compartment cartilage, menisci, BML, effusion-synovitis (ideal labels) | **high**: institutional approval ≈ KneeCoT ruling | skip |
| KneeCoT | gated + hospital agreement | — | **ruled out by host** | skip |

## How the data reaches the pipeline (no secrets in chat or repo)
- The user registers and receives NYU's download links by email. Links are personal: never paste them in chat,
  commit them, or share them.
- The user saves the links in `~/.rsna_ext/fastmri_links.txt` (chmod 600; outside the repo). Scripts read that
  file, download straight to a RunPod volume (~100 GB singlecoil knee train+val), and never print the links.
- Derived arrays stay in private storage only (DSA: no redistribution).
</content>
</invoke>
<invoke name="Bash">
<parameter name="command">cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model && git add docs/EXTERNAL_DATA.md && git commit -q -m "EXTERNAL_DATA: rules + host rulings + candidate datasets (fastMRI+ = register; MRNet/KneeMRI/OAI/KneeCoT = skip)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git log --oneline -1