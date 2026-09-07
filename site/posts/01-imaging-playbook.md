# An Imaging Playbook for RSNA Knee Abnormality Detection

This series takes ideas from top Kaggle competitors and applies them to [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

**RSNA** is the Radiological Society of North America. This contest is about finding knee problems in **MRI** scans (magnetic resonance imaging: detailed hospital scans of soft tissue). It is a **code competition**: you submit a program that runs on hidden data, not a spreadsheet of answers. It is **not** a classic table-of-numbers (tabular) contest.

We follow two NVIDIA writeups for how to work, not for copy-paste tricks meant for tables:

- [The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [Winning a Kaggle Competition with Generative AI-Assisted Coding](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/)

What we keep from them: try ideas quickly, trust a careful local score before believing the public scoreboard, use coding assistants to move faster, and always save each experiment’s predictions. What we do **not** keep: recipes built for rows and columns. A knee exam is a stack of scan sequences, not one spreadsheet row.

This series starts **from scratch**. Older trial runs on this contest are only a list of what already failed, not a finished system we keep using.

---

## What the problem actually is

**Job:** look at one knee MRI **exam** (one scanning visit for one knee) and output **12 numbers** between 0 and 1. Each number is “how likely is this finding present?”

**Score:** **macro AUC**. **AUC** means area under the **ROC** curve (receiver operating characteristic: a standard plot of true-positive rate vs false-positive rate; AUC summarizes how well the model ranks true cases above false ones). **Macro** means: compute AUC for each of the 12 labels, then average those 12 scores. If the model is great on fluid around the joint but useless on a torn **ACL** (anterior cruciate ligament) or on fracture, the average still suffers.

| Piece | In plain terms |
|---|---|
| Training exams | About 4,400 visits (`train.csv`) |
| Expert-checked exams | Only 58 visits have full expert labels (~1.3%) |
| Hidden test exams | About 1,300 (the public test file is almost empty) |
| Raw scans | Hundreds of gigabytes of **DICOM** files (the hospital image format) on Kaggle |
| Sequences per exam | Usually several (about 5.5 on average; from 3 to 14) |

One exam is **not** one photo. It is several **series** (scan sequences) from different **planes** (view directions: side/sagittal, front/coronal, top-down/axial), plus flags such as fluid-sensitive or fat-suppressed imaging.

**Training data** includes written radiology reports in many languages, but most of the 12 label columns are empty except for those 58 expert exams. **Test data** has exam IDs, series info, and DICOM images, and **no reports**. So you cannot use report text when the final program runs. Building a system that needs reports at test time would also leak information that will not exist later.

### Contest rules: paid online language models (training only)

An **LLM** is a large language model (a text AI such as a chat model). An **API** is a remote service you call over the internet.

The hosts say you **may** send training report text to a commercial LLM or similar service to help build labels, if that service follows the [Competition Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules), including section **2.6.b (External Data and Tools)**: it must be **reasonably available to everyone** and **not expensive**. Doing that is **not**, by itself, illegal “private sharing” of contest data outside your team. Private sharing still means: do not team up or trade contest work with people outside your registered team. The hosts can still ban a service that is unfair, locked away, or too costly.

For this series: using an affordable, public LLM service to help create training labels offline is allowed. Calling such a service **inside the final submission notebook** is not (no internet; no reports on the test set).

### Contest rules: KneeCoT is banned

[KneeCoT](https://huggingface.co/datasets/YiHui0124/KneeCoT) is an outside knee dataset on Hugging Face. Access is gated. The license is **CC BY-NC 4.0** (Creative Commons Attribution NonCommercial), and users must sign a formal ethics agreement with the hospital that owns the data.

The hosts decided KneeCoT is **not allowed**. Reason: that agreement can favor people who can get institutional approval over people who cannot, which breaks the “equally accessible” idea in the rules. Unless the hosts change their mind, we will **not** train or copy knowledge from KneeCoT.

This is still a **code competition**: your notebook must finish within **9 hours**, with the internet turned off. Uploading a hand-made answer file is rejected. There are **two finals**: one for highest accuracy (AUC), and one for **efficiency** (good enough accuracy plus how fast the run is).

The **CSV** files (comma-separated values: simple spreadsheet-like text tables) only describe the exams. The real data is the images.

---

## The two exams problem

Most failed ideas come from training and scoring on different “exams”:

- **What you train on:** imperfect labels taken from radiology reports written in many languages (plus those 58 expert exams).
- **What Kaggle scores:** expert radiologists judging the **images** on a hidden test set.

You never directly optimize the exact thing the scoreboard uses. So choosing *how* you measure progress is as important as choosing a model.

| Yardstick | Honest use |
|---|---|
| Weak-label validation AUC | Quick smoke check only; it usually looks better than the public scoreboard |
| Full-58 expert macro AUC | Main keep-or-kill score (single labels are noisy at this tiny size; use the average of all 12) |
| Public **LB** (leaderboard) | Reality check only; tiny score changes are often noise |
| Top of the leaderboard (~0.94) | The real gap we must close |

From older trials (not this series’ current standing): weak-label scores can sit about 0.07 above the public leaderboard; the 58-expert score can sit about 0.02 above it. Use those as rough offsets for the yardsticks, not as a claim that a new system is already there.

One more host detail: borderline findings (“maybe / on the fence”) are marked **negative**. The hosts prefer fewer false alarms. Also, the 58 expert exams are **not** a mini copy of the full report set (for example, ACL tears are much more common in the 58 than in the reports). Do not use those 58 exams to set probability cutoffs. Use them only as a ranking yardstick, with a firm margin before you call a change a win.

### Multilingual reports (short)

Reports appear in French (many), Turkish (about 600), Spanish, German, Greek (about 320), Dutch, and English. Saying “no” is not always English-shaped. In Turkish, the denial often comes **after** the noun (for example, wording like “efüzyon izlenmedi” means roughly “effusion not seen”). Tools that only search English keywords miss a lot.

How we build language recipes in detail comes in a later post. Here the point is simple: the text teacher speaks many languages and is imperfect.

---

## Two foundations, remapped

The Grandmasters Playbook rests on two habits. Here they mean:

**Try ideas fast.** The full image set does not fit a normal laptop. We keep small tables locally, and we decode images, train, and submit on Kaggle, under weekly **GPU** limits (graphics processing unit: the chip used to train deep models quickly) and session caps. Speed means: (1) a reusable image cache that is cheap to rebuild, (2) submission code that opens known folders instead of blindly searching the whole scan tree, and (3) coding assistants that write boilerplate so people spend time on scoring rules and kill decisions.

**Validate carefully.** Match the yardstick to what the contest grades. Policy for this series:

- Keep or kill using **full-58 expert macro AUC**, needing at least a **0.005** gain.
- Do not ship a change based only on weak-label scores.
- Do not flip a decision because one single label jumped on only 58 exams.
- The public leaderboard confirms; it does not choose the winner for us.

---

## Four-step assistant workflow

Chris Deotte’s post on AI-assisted coding describes a four-step loop where software assistants write code and a person steers. For knee MRI:

1. **EDA** (exploratory data analysis: looking carefully at the data before modeling): How many exams and series? Which view directions? What is missing at test time? Where do reports disagree with experts? Do hospitals and scanners differ?
2. **Baselines:** Train a first full pipeline (often with **k-fold** validation: split data into k parts, hold one part out each time). Save **OOF** predictions (out-of-fold: scores on data the model did not train on in that fold) and test predictions with clear file names. Print the metric every fold.
3. **Improve:** Change **one** thing per run (how much of the scan we load, how we combine slices, which text labels we trust, or the loss). Keep winners. Write down losers.
4. **Combine:** Blend or stack models only after one model is strong enough, and only if the blend still fits inside the 9-hour limit.

Assistants speed up coding. Yardsticks decide what we keep. Every run leaves prediction files behind, including failures.

---

## What already failed (kill list)

Older work on this contest produced a clear “do not repeat” list. This series treats it as design input, not as the system we are shipping.

| What was tried | What happened, and the rule we take |
|---|---|
| Unfreezing the whole image backbone aggressively | Expert score collapsed → keep the backbone frozen by default; unfreeze only in a careful, gated trial |
| Naively loading more series and more slices | Score got worse than a thinner cache → “more pixels” without a plan is not a plan |
| Fine-tuning only the last layer on the 58 experts | No lasting win → those 58 exams are too small to be the main training set |
| Using a public outside KneeMRI set as the score or as extra training | About chance-level on ACL (different hospitals and scan layouts) → outside MRI is not a stand-in yardstick |
| Gated KneeCoT (Hugging Face + hospital agreement) | Host banned it as unfair access → never use it for extra training |
| Clever per-plane routing plus class weights | Expert score fell; **MCL** (medial collateral ligament) got worse → routing must prove itself under the same yardstick |
| Endless tiny label edits without a hard quality gate | Random wrong fills hurt ligament labels → build at most one fair LLM/keyword teacher, check it on the 58 experts, then **lock** it |
| Thin fixed cache such as 3 series × 12 slices × 224 pixels | Stuck far below leaders → the remaining gap is how much of the MRI the model actually sees |
| Multi-model ensemble decoding on **CPU** (central processing unit: normal computer chip, slower for this) | Timed out on the full hidden test size → submission speed is a hard product requirement from day one |

Leaders sit near about 0.94 on the public leaderboard. Thin frozen-image setups with noisy text labels historically sat in the high 0.60s. Closing that gap is mostly an **image system** problem under a locked text teacher, not another spreadsheet of labels.

---

## Starting plan (from scratch)

Given that kill list, this series bets on a different design:

1. **Lock a fair text teacher.** During training only, we may use affordable, open LLM services (or public local models) under rule 2.6.b to help turn reports into labels. Check quality once on the 58 experts, write down which service we used so others could afford the same path, then lock it. Never call outside APIs from the submission notebook.
2. **Make the MRI the main product.** Stop using one thin, uniform crop of the exam. Prefer a smart sampling plan: more side-view slices for ligaments and menisci; enough front and top-down views for arthritis, fluid, and bone. Decode each scan once. Leave time budget before any five-model blend.
3. **One strong exam model before blends.** The model should see slices in order (short “video-like” or attention-over-slices designs). Beat a high full-58 expert bar with **one** model before blending folds or views.
4. **Build the fast student at the same time.** Same cache and same labels; smaller or faster model for the efficiency final. Do not leave that prize as an afterthought.
5. **Treat submission as part of the design.** Use known DICOM folders, decode on GPU once, do not blindly search the whole tree, and time the run on more than three tiny placeholder exams. At test time: images and series info only.

Later posts build and measure this plan step by step.

---

## Series roadmap

| Idea from the tabular playbook | What it means for knee MRI | Post |
|---|---|---|
| Core foundations | Fast Kaggle loop + full-58 expert yardstick | 01 (this post) |
| Smarter EDA | Exams, view planes, multilingual reports, hospital shift | 02 |
| Diverse baselines | New simple baselines (from scratch, not the old stack) | 03 |
| Feature engineering | Smart sampling of the MRI volume | 04 |
| Pseudo-labeling | Fair text teacher (rule 2.6.b) + quality gate, then lock | 05 |
| Hill climb / stacking | Only after one model is strong; stay under 9 hours | 06 |
| Extra training | More random seeds / all training data after the yardstick says yes | 07 |
| Combine / ship | Best-AUC final + efficiency student | 08 |

Code and posts live in one public repository: [RSNA_Knee_Abnormality_Detection_Model](https://github.com/Girish011/RSNA_Knee_Abnormality_Detection_Model).

---

## Attribution

Method sources:

- [Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) (Onodera, Viel, Titericz, Deotte)
- [Winning with Generative AI-Assisted Coding](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/) (Deotte)

Competition: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection). Rules: [Competition Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) section 2.6.b (external data and tools), including the host notes on commercial LLMs for training-time report work and on banning KneeCoT.

Numbers from older trials are **history used for redesign**. They are not this series’ published leaderboard standing.
