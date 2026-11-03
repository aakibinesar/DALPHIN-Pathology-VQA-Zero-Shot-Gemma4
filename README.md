# BEETLE breast-tissue segmentation: pathology foundation models, ensembles and a blend with the organisers' nnU-Net

Code and write-up for my entries to [BEETLE](https://beetle.grand-challenge.org/) (Grand Challenge): 4-class tissue segmentation (other, non-invasive epithelium, invasive epithelium, necrosis) on 170 hidden H&E regions from three centres. The leaderboard metric is `overall_dice`.

**Status: ongoing.** Last updated 2026-11-03. All submissions to date are listed in [RESULTS.md](RESULTS.md).

**Results so far (public leaderboard, hidden test set)**

| entry | overall_dice |
|---|---|
| organisers' released nnU-Net 5-fold ensemble, run by me, no training | 0.8666 |
| my own models alone (single models and averages, see RESULTS.md) | 0.8240 to 0.8464 |
| **50% my model (Phikon-v2) + 50% the organisers' nnU-Net** | **0.8839** |

The best score needs the organisers' model. **My own models alone never beat the organisers' reproduced result**; the best of them reaches 0.8464. The blend weights were fixed before any score was seen, and I did not tune them on the leaderboard. I have made 12 submissions so far.

## 1. Task and constraints
- Train on the public BEETLE development set (Zenodo, CC BY-NC-SA 4.0); the 170 evaluation regions (about 2048 to 2560 px) have hidden labels, so every test number comes from uploading a zip of label PNGs.
- Limit: 12 submissions per 28 days.
- Compute: Kaggle free tier only (about 30 GPU hours per week; T4 GPUs; 12 h per session). Slides are read in Kaggle notebooks by HTTP range request, so the 120 GB of slides never touch my disk.
- No external data. Slides named after TCGA cases were left out of training. Splits are by patient.

## 2. Models (`src/`, `kernels/`)
| name | backbone | what was trained | training data | test score (alone) |
|---|---|---|---|---|
| v2 | Phikon ViT-B/16 (Owkin) | last 4 blocks + light CNN-skip decoder; stage 1 on 72 slides, stage 2 on 154 | 154 slides, 26,343 patches | 0.8388 |
| v3 | same, continued | stain (H&E) jitter in optical-density space | 218 slides | 0.8240 |
| pv2 | Phikon-v2 (DINOv2 ViT-L/16) | last 6 of 24 blocks + same decoder; stain jitter | 218 slides | 0.8269 |
| pv2abl | Phikon-v2 | ablation of pv2: original 154 slides, no jitter | 154 slides, 26,343 patches | never submitted alone |
| pv2big | Phikon-v2 | last 6 of 24 blocks; no jitter; original 154 slides plus 113 further slides | 43,772 patches | 0.8464 |

- **Patches:** 512 px at 0.5 µm/px, class-balanced centres per slide for training; a patient-disjoint validation set of 23 slides (1,715 patches) for model selection.
- **Training:** AdamW, cosine schedule, fp16; a class-rarity sampler; backbone learning rate 3e-5 (with a delay for the large model). The checkpoint was chosen by mean per-class Dice on the validation set (for v3 and pv2 together with a held-out stain-shift set).
- **Inference:** overlapping 512 px windows (stride 256, cosine taper), four-fold flip/rotation averaging, arg-max over the four classes.
- **Organisers' model:** the released 5-fold nnU-Net ensemble from the BEETLE repository, run by me on Kaggle (`kernels/22_infer`); I did not train it. With flip averaging its score was 0.8667, so flip averaging changes nothing measurable.
- **Ensembles and blends:** per-pixel class probabilities are saved as uint8 and averaged on CPU (`ensemble/`). Weights were fixed in advance. Saved probabilities reproduce the submitted zips on at least 99.985% of pixels.

## 3. What I found
- **Validation did not predict the test.** In-distribution validation mean-class Dice went from 0.851 (v2) to 0.912 (pv2big), but the own-only test scores stayed between 0.824 and 0.846.
- **Averaging two own models helped; four did not.** v2 + pv2: 0.8459; v2 + v3 + pv2 + pv2abl: 0.8429. Equal weights on weaker members cost a little.
- **The extra data did not change the overall score, but changed its profile.** pv2big (113 further slides) scored 0.8464 against 0.8459 for the earlier two-model average, a difference far below what 170 regions can resolve. Compared with that average it is much better on necrosis (0.6123 against 0.5342) and a little better on the Biopticka centre (0.8335 against 0.8209) and on invasive tumour (+0.011), and worse on the UWMed centre (-0.018), non-invasive epithelium (-0.012) and other tissue (-0.005).
- **Stain jitter did not help.** v3 scored lower than v2 at all three centres.
- **Where we lose:** v2 was better than the organisers' model on necrosis (+0.040), on the UWMed centre (+0.029) and on ILC invasive tumour (+0.017), and worse on the Biopticka centre (-0.080) and on NST invasive tumour (-0.050). So most of the gap is one centre, which fits a domain shift.
- **Stain statistics (weak support, not proof):** the test regions are farther in colour statistics from the nearest training slide (median 0.66) than training slides are from each other (0.43). The per-region correlation with disagreement with the organisers is weak (Spearman 0.18).
- **Agreement with the organisers' predictions is not a reliable proxy.** Phikon-v2 agreed most with them (0.870 of pixels, against 0.864 for v2) and scored lower than v2 (0.8269 against 0.8388), so the leaderboard is the only measurement I trust.
- **Blends:** a 50/50 blend with the organisers' model adds +0.002 (v2 alone), +0.009 (average of v2 and pv2) and +0.017 (pv2big) over their model alone. A blend with the organisers at 70% scored 0.8789; I do not use it as a headline because my model carries less than half.

### The last blend (submission 12)
Submission 12 is 50% pv2big + 50% the organisers' nnU-Net (0.8839). pv2big is the Phikon-v2 model trained on the original 154 slides plus 113 further development slides, without stain jitter. Alone it scored 0.8464: higher than each of my earlier single models (v2 0.8388, v3 0.8240, Phikon-v2 with jitter 0.8269) and 0.0005 above the average of two models (0.8459), which is within noise. I chose it for the blend because of a rule I fixed before submitting it: if it scored above 0.8459 alone, blend it 50/50 with the organisers' model. The blend is 0.0173 above the organisers alone (0.8666) and 0.0086 above the earlier 25/25/50 blend (0.8753). It is above the organisers' model in every class and at every centre: necrosis +0.122, non-invasive epithelium +0.063, invasive epithelium +0.050, other tissue +0.002; Biopticka +0.013, SCDC +0.009, UWMed +0.032.

## 4. Limits
- 170 regions from three centres: differences of about 0.01 or less should not be read as real. I did not estimate confidence intervals.
- "Best of N" is optimistic: 12 submissions in total, all in RESULTS.md.
- The headline number is a blend that contains the organisers' released model.
- I used part of the 396 development slides and no external data; free-tier compute limited training to a few epochs per model. Top entries on the leaderboard score higher.

## 5. Repository layout
| path | content |
|---|---|
| `src/` | the two segmentation modules (Phikon ViT-B, Phikon-v2) |
| `kernels/20_fetch`, `22_infer`, `23_tta` | fetch of the evaluation regions and organisers' model; their nnU-Net run (no training) |
| `kernels/30_wsiprobe`, `31_prep`, `33_prep2`, `38_prep3`, `39_prep4`, `42_prep5`, `45_prep6` | slide probing and patch extraction (slide selections in `selection*.json`) |
| `kernels/41_pack`, `43_pack2`, `46_pack3` | packing of the patches into a few large shard files |
| `kernels/32_train*`, `62_train_pv2` | trainers (ViT-B; Phikon-v2) with `build.sh` generators and local tests |
| `kernels/36_infer_seg`, `64_infer_pv2`, `66_probs_vitb` | inference and saving of probabilities |
| `kernels/50_blend`, `65_blend_pv2`, `ensemble/` | blends with the organisers' model; CPU ensembling from saved probabilities |
| `kernels/34_check`, `35_scale`, `37_maskstats`, `51_stain`, `52_stain2`, `53_priors`, `63_pv2check` | data checks and the stain / class-prior analyses |
| `select_shift.py`, `shift.json` | the held-out stain-shift patients |
| `RESULTS.md` | every submission, with date and weights |

## 6. Reproducing
Set `KAGGLE_USER`, then generate a notebook, for example the Phikon-v2 trainer: `cd kernels/62_train_pv2 && ./build.sh <name> ...` (the header of each `build.sh` lists its arguments), and push it with `kaggle kernels push -p kernels/62_train_pv2 --accelerator NvidiaTeslaT4`. The notebooks read the slides and the organisers' model from Zenodo and from earlier notebook outputs; adapt the slugs to your account. Weights and probabilities are not included.

## 7. Data and licence note
BEETLE data and the organisers' model belong to the organisers (CC BY-NC-SA 4.0) and are not included. Phikon and Phikon-v2 are used under Owkin's non-commercial licence. No data, weights or probabilities are redistributed here.

## 8. Credits
- BEETLE challenge, data, evaluation and the released nnU-Net baseline: the BEETLE organisers (Grand Challenge, Zenodo record 16812932). nnU-Net: Isensee et al.
- Backbones: Phikon and Phikon-v2 (Owkin). Libraries: PyTorch, transformers, nnU-Net.
- All code here was written for this project; no code from other participants' submissions was used.
