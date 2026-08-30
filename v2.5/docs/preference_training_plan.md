# Preference Training Plan

Date: 2026-08-25

> **SUPERSEDED SCOPE:** All new training, preference learning, generation evaluation, and module experiments must use the 7B Generator. The 3B Generator route is abandoned as of 2026-08-25. Existing 3B checkpoints and partial evaluations are historical artifacts only; no new 3B jobs may be submitted. The remaining 3B sections below are retained only for provenance.

## Active 7B-only route

- Base: `Qwen/Qwen2.5-VL-7B-Instruct`.
- SFT winner from the controlled module pilot: MLP `gate_proj/up_proj/down_proj`, rank 3, alpha 6.
- Active checkpoint: `outputs/7b-generator/best_val_loss` (final validation loss `2.930127`).
- Current job: `6585267`, a single-MIG 7B Base-vs-SFT held-out generation and blind-packet evaluation (submitted 2026-08-25; initially queued).
- Next: held-out 7B SFT generation audit; recompute frozen reference log-probabilities with the 7B SFT checkpoint; screen preference objectives on 7B; perform module-placement Go/No-Go only if 7B preference learning first shows a generation-level gain.
- Old 3B reference log-probabilities must never be reused for 7B.
- The stopped dual-model/joint-training decision remains in force.

## Archived 3B objective evidence

The 3B full-pair training runs completed, but the generation evaluation was stopped when the project moved to 7B-only. The auxiliary 24-image judge completed only SFT, DPO, and SimPO: qualified rates were 70.83%, 62.50%, and 58.33%; average humor scores were 4.542, 4.583, and 4.542. DPO is the provisional leader among evaluated preference methods by average humor/overall score, but it did not beat SFT on qualified rate. IPO, Anchored, and blind win-rate evaluation are incomplete. Therefore no scientifically valid overall objective winner was established on 3B, and these results only prioritize DPO as the first 7B preference baseline.

## Fixed current state

| Variable | Resolved value |
|---|---|
| `BEST_SFT_CHECKPOINT` | `outputs/newyorker_compact_v2_captioner_3b_qlora/best_val_loss` |
| `GENERATOR_CHECKPOINT` | `Qwen/Qwen2.5-VL-3B-Instruct` + the SFT adapter above |
| `HINT_MODEL_CHECKPOINT` | `Qwen/Qwen2.5-VL-7B-Instruct` + `outputs/newyorker_caption_aware_viewpoint_v3_7b_qlora/final_lora` |
| `DPO_TRAIN_DATA` | `data/processed/newyorker_published_dpo_reference_3b/dpo_train.jsonl` |
| `DPO_VALID_DATA` | `data/processed/newyorker_published_dpo_reference_3b/dpo_validation.jsonl` |
| `EVALUATION_ENTRYPOINT` | generation: `scripts/generate_lora_candidates.py`; blind group evaluation: `scripts/build_group_winrate_eval.py` + `scripts/report_group_winrate_eval.py` |
| `JUDGE_ENTRYPOINT` | strict auxiliary multidimensional judge: `scripts/preference_diagnostics/judge_humor_candidates_qwen.py`; primary comparative report: group win-rate pipeline above |
| Existing Generator LoRA | rank 16, alpha 32, dropout 0.05; `q_proj`, `k_proj`, `v_proj`, `o_proj`; 7,372,800 parameters |

The final rather than minimum-validation-loss 7B adapter is the current Hint Model checkpoint because the held-out blind evaluation found 33.3% acceptable captions with the final adapter versus 26.4% with the minimum-loss adapter. This is an empirical pipeline choice, not a claim that training loss selects poor checkpoints in general.

## Current Hint to Generator interface

The 7B model maps image to caption-aware compact JSON with `scene`, `type`, `target`, viewpoints, anchors, and an external-knowledge flag. `scripts/build_captioner_inputs_from_plans.py` validates that JSON and renders it into the 3B prompt inside `<joke_annotations>...</joke_annotations>`.

The published Generator preference pairs currently contain a fixed three-line `ANCHOR/CONTRAST/ANGLE` humor plan. Both chosen and rejected captions share exactly the same image and prompt. This satisfies the no-confounding requirement for Generator preference optimization. It is not silently rewritten into generated 7B hints during objective or placement screening. Deployment-style compact JSON hints enter only after the best Generator configuration is frozen and downstream Hint utility is measured.

Hint usefulness and basic DPO-data validity are accepted prerequisites and will not be rerun.

## Training entrypoints and implementation boundary

- Frozen SFT reference log probabilities: `scripts/precompute_dpo_reference_logps.py`.
- Current DPO trainer: `scripts/train_lora_dpo.py`.
- Exact module diagnosis: `scripts/preference_analysis/module_preference_gradient.py`.
- LoRA budget matching: `scripts/preference_analysis/match_lora_budget.py`.
- Existing generation and evaluation code remains unchanged and reusable.

Job `6578562` is only a full reference-log-probability plus one-optimizer-step smoke. It cannot produce a trained preference adapter.

## Successive experiment list

1. Smoke and screen DPO, SimPO, IPO, and positive-anchored preference with identical checkpoint, data, placement, budget, tokens, and seed.
2. Retain the best one or two objectives.
3. Run a low-cost module-placement Go/No-Go pilot under an approximately equal LoRA parameter budget: all-linear, attention, MLP, gradient-selected, and random-selected only.
4. Treat module gradients as an analysis signal. Continue expensive Fisher, SVD, dynamic-rank, or large layer-wise search only if gradient-selected is clearly and stably better than random-selected and has a parameter-efficiency advantage.
5. If the gate passes, validate the minimum additional diagnosis required and compare continuing the SFT adapter with a separate preference adapter. If it fails, retain the best simple placement and stop module search.
6. Train and evaluate the selected Generator configuration only.
7. Report three-seed final results for the selected objective and the five Go/No-Go placements.

### Module-search gate amendment

The project owner changed module selection to a low-cost Go/No-Go pilot on 2026-08-24. Exact layer-wise Fisher job `6578822` was stopped and will not be resubmitted before the gate. Its one-layer smoke also exposed a CUDA allocator/NVML internal assertion during exact base-weight autograd; this failed run produced no valid ranking and must not be interpreted as evidence for or against any module family.

The five pilot placements are parameter matched around 7.37M LoRA parameters: Attention `r=16` (7.373M), MLP `r=5` (7.050M), All-linear `r=4` (7.483M), and Gradient/Random-selected `r=73` (7.400M each). Gradient-selected contains the top 30 matrices from the already completed attention-LoRA tangent-gradient analysis: 18 `o_proj` and 12 `v_proj`. Random-selected is disjoint and matches those family counts exactly. This signal does not rank MLP modules; MLP is retained as a direct intervention baseline.

Go/No-Go is successive. First run one seed for all five placements. Only if Gradient-selected has a positive held-out point advantage over Random-selected do two additional matched seeds run. Expensive diagnosis passes only if the three-seed Gradient-minus-Random effect is positive on the preregistered primary blind group metric with an image-bootstrap 95% interval above zero, does not materially reduce grounding, and has higher improvement per million trainable parameters than Random and the simple placements. Otherwise the decision is No-Go: stop Fisher/SVD/dynamic-rank/layer search and use the best simple placement.

### Dual-model training stopped

The project owner stopped the subsequent Hint Model–Generator joint/sequential/alternating training plan on 2026-08-24. No `H0 -> G1 -> H1 -> G2`, Hint preference training, joint RL, or alternating refinement job may be submitted under this plan. Existing Hint checkpoints and the fixed Hint→Generator inference interface remain available for conditioning and evaluation only. The small downstream Hint-utility utility code added before this decision is dormant scaffolding; it is not an active experiment and does not authorize data generation or training.

## Evaluation contract

Every result-bearing run records the resolved config, seed, dataset hash, checkpoint, generation settings, trainable parameter count, memory/time/throughput, chosen and rejected absolute log probabilities, preference margin/accuracy, and multidimensional generation results. Primary claims require held-out blind group win rate with image-level confidence intervals; the same auxiliary judge may be used only for low-cost screening.

## Research basis

1. Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
2. Azar et al. (2024), *A General Theoretical Paradigm to Understand Learning from Human Preferences*, AISTATS 2024 (IPO). https://proceedings.mlr.press/v238/azar24a.html
3. Meng et al. (2024), *SimPO: Simple Preference Optimization with a Reference-Free Reward*, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/3a6bfa0d2b8cfce85f61f3c23c7f8b90-Abstract-Conference.html
4. Xu et al. (2024), *Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation*, ICML 2024. https://proceedings.mlr.press/v235/xu24h.html
5. Zhang et al. (2024), *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning*, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html
6. Hessel et al. (2023), *Do Androids Laugh at Electric Sheep? Humor Understanding Benchmarks from The New Yorker Caption Contest*, ACL 2023 Best Paper. https://aclanthology.org/2023.acl-long.41/
