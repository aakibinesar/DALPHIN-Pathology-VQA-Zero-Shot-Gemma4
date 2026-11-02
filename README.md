# DALPHIN pathology VQA: zero-shot open vision-language models on a pathology benchmark

Code and write-up for my entries to the [DALPHIN](https://arxiv.org/abs/2605.03544) benchmark (Grand Challenge): six question types about 300 pathology cases (1,236 images).
All entries are **zero-shot**: no training or fine-tuning of any model. The best scores come from Gemma-4-31B-it, loaded in 4-bit on Kaggle's free two-T4 machine.

**Best scores (public leaderboard, full set)**

| task | metric | smallest model (Qwen2.5-VL-3B) | best entry | best configuration |
|---|---|---|---|---|
| Tissue / organ | hierarchical organ score | 0.154 | **0.650** | Gemma-4-31B, organisers' prompt only, image token budget 560 |
| Neoplastic status | F1 (positive class) | 0.683 | **0.909** | Gemma-4-31B with the model's own diagnosis answer (independent run) as text context |
| Behaviour | Cohen's kappa | 0.038 | **0.663** | Gemma-4-31B, text context of its own earlier answers (see section 6: this board is noisy) |
| Diagnosis | BioBERT similarity | 0.590 | **0.734** | Gemma-4-31B, neoplasm answer as text context, organisers' prompt only |
| Multiple choice | accuracy | 0.407 | **0.697** | plurality vote of four runs with different inputs |
| Free response | BioBERT similarity | 0.433 | **0.580** | Gemma-4-31B, organisers' prompt only |

Each number is the best of several attempts (tissue 8, neoplastic status 7, behaviour 7, diagnosis 8, multiple choice 6, free response 6; the list of attempts is in section 5). The comparison with the organisers' own baselines and other participants' entries is on the public leaderboards.

## 1. Task and constraints
- Six tasks per case, in this order: tissue, neoplastic status, diagnosis, behaviour, multiple choice, free response. Answers are free text or a fixed label.
- Ground truth is hidden; every score comes from uploading a submission file to the challenge platform. Each board allows 10 submissions per 28 days.
- Compute: Kaggle free tier only (a weekly quota of about 30 GPU hours; two T4 GPUs, 15 GB each, 12 h per session).
- Method, not model size, is where I could contribute: no model was trained, so the work is the prompting, the answer cleaning, the evaluation harness and a set of controlled comparisons.

## 2. Pipeline (`dalphin_pipeline.py`)
- **Prompts:** the organisers' preamble (tissue type, image description, "one short answer") plus the question; for some runs one extra instruction sentence (e.g. "answer with exactly one of: benign, uncertain, in situ, malignant").
- **Images:** the case's overview and region-of-interest images at their native size (about 1,000 px). Gemma-4 gives every image a fixed token budget of 280 by default; raising it to 560 matches the native size (tissue +0.04).
- **Answer cleaning:** fixed rules per task (yes/no, the four behaviour labels, a single letter, first line for tissue, brackets removed for diagnosis), written to mirror the organisers' published curation rules. A validator copies the format checks of the organisers' evaluation code, so a file that would fail on the platform fails locally first.
- **Model loading:** Gemma-4-31B-it is quantised to 4-bit (NF4, bitsandbytes) while loading and spread over both T4s; about 16 s per question at budget 560.
- **Context runs (`kernels/52_full/make_hybrid_body.py`):** for each case the model sees, as text only, earlier questions and answers of the same case. The earlier answers come from independent runs (the images are not repeated). This differs from the organisers' sequential scenario, where earlier images stay in the conversation. Leaderboard comments say "contextual" for these runs.
- **Voting (`vote.py`):** plurality over several runs, ties to the earlier run. It helped only where the runs differed in input.
- **Notebook generators:** `kernels/52_full/build.sh` assembles one self-contained Kaggle script from `dalphin_pipeline.py` and a body file; `RUNS.md` lists the recipe of every notebook that was built.

## 3. What helped
| change | effect |
|---|---|
| model size, 12B → 31B | tissue +0.15, the largest single gain |
| image token budget 280 → 560 (native size) | tissue 0.590 → 0.629 |
| organisers' prompt only, no extra instruction | free response 0.500 → 0.580, diagnosis and tissue slightly better |
| text context from the model's own earlier answers | neoplastic status 0.881 → 0.909 (diagnosis as context), diagnosis 0.719 → 0.734 (neoplasm as context) |
| plurality vote over runs with different inputs | neoplastic status 0.881 → 0.904, multiple choice 0.680 → 0.697 |

## 4. What did not help (measured, briefly)
- Gemma-4 thinking mode works but runs at about 3.5 tokens/s on two T4s: about 17.7 GPU hours for one task at a 1,024-token budget, so it was not affordable.
- Putting the images before the text (the model card's advice): behaviour 0.515 against 0.560.
- MedGemma-27B-it (4-bit, same pipeline): diagnosis 0.590 against 0.734 for Gemma-4-31B; its tissue answers agreed with Gemma's on only 16% of cases and were not submitted.
- Choosing the most central diagnosis among four of my own runs with the organisers' BioBERT model: 0.722, below the best single run (0.734).
- A four-run vote for neoplastic status: 0.901, below the best single run with context (0.909).

## 5. Attempts per board (scores in the order they were submitted)
- Tissue (8): 0.154, 0.337, 0.441, 0.590, 0.564, 0.629, 0.650, 0.645
- Neoplastic status (7): 0.683, 0.793, 0.876, 0.881, 0.904, 0.909, 0.901
- Behaviour (7): 0.038, 0.232, 0.324, 0.560, 0.663, 0.554, 0.515
- Diagnosis (8): 0.590, 0.616, 0.664, 0.713, 0.719, 0.734, 0.722, 0.590 (MedGemma)
- Multiple choice (6): 0.407, 0.470, 0.613, 0.680, 0.680, 0.697
- Free response (6): 0.433, 0.448, 0.483, 0.500, 0.480, 0.580

## 6. Limits
- 300 cases give a standard error of about 0.02 on most scores; differences of a few thousandths are not meaningful.
- **Behaviour is the noisiest board:** near-identical runs scored 0.515, 0.554 and 0.560, and a contextual run 0.663; I would not read the 0.663 as a stable gain.
- "Best of N" is optimistic by construction; section 5 lists every attempt so the reader can judge.
- The context runs use text-only history and answers from earlier independent runs, which is not the organisers' sequential protocol.
- Results are for 4-bit models on free-tier hardware; larger or full-precision models, other prompts and reasoning modes may do better, as the top entries on the leaderboards show.

## 7. Repository layout
| path | content |
|---|---|
| `dalphin_pipeline.py`, `test_pipeline.py` | prompts, answer cleaning, validator, and their tests (the tests need the organisers' `dalphin_metadata.csv`, see section 9) |
| `vote.py`, `merge_*.py`, `clean_medgemma_diag.py` | voting, merging of split runs, cleaning of long MedGemma diagnosis answers |
| `kernels/52_full/` | `build.sh` and the body generators for the Kaggle notebooks; `test_hybrid_chain.py` checks the context chain with a stub model |
| `kernels/51_images/` | notebook that fetches the benchmark images |
| `kernels/54_mbr/` | the BioBERT-centrality selection notebook |
| `experiments/` | generators for the experiments in section 4 (thinking mode, image order, MedGemma) |
| `RUNS.md` | settings of every notebook that was built |

## 8. Reproducing
1. Download the benchmark images and the question metadata from the organisers (section 9); put `dalphin_metadata.csv` next to `dalphin_pipeline.py`.
2. Set `KAGGLE_USER`, then build a notebook, for example the independent run for four tasks at the 560-token budget:
   `TASKS_ONLY=behavior,diagnosis,neoplasm,mc TOKBUDGET=560 MAXSIDE=1100 kernels/52_full/build.sh gemma31b-t560-a gemma31b 0 google/gemma-4/transformers/gemma-4-31b-it/1 float16 true`
   and push it with `kaggle kernels push -p kernels/52_full --accelerator NvidiaTeslaT4`.
3. The notebooks read the images from a Kaggle notebook output (`kernels/51_images`) and the model from Kaggle Models; adapt the slugs to your account.
4. Clean the answers, check them with the validator, upload them to the challenge.

## 9. Data and licence note
The benchmark images (CC BY-NC-ND 4.0) and the question metadata belong to the DALPHIN organisers and are not included. No labels are available to anyone outside the organisers. Model weights are used under their own terms (Gemma, Qwen, and the Health AI Developer Foundations terms for MedGemma).

## 10. Credits
- DALPHIN benchmark, data, evaluation code and baselines: Lems et al., *DALPHIN: Benchmarking Digital Pathology AI Copilots Against Pathologists on an Open Multicentric Dataset*, arXiv:2605.03544. My prompts, answer cleaning and validator follow the descriptions of the preprint and the organisers' public evaluation code.
- Models: Gemma-4 (Google), Qwen2.5-VL and Qwen3-VL (Alibaba), MedGemma-27B (Google). Libraries: transformers, bitsandbytes, PyTorch. The BioBERT sentence model used by the organisers' diagnosis metric (`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`) was used for the selection experiment.
- All code here was written for this project; no code from other participants' submissions was used.
