# Preference Learning Repository Audit

Audit date: 2026-08-24 (Asia/Tokyo)

## 1. Scope and decision boundary

This repository is not an empty preference-learning project. It contains an older reranker-oriented V2.5 pipeline, several generations of New Yorker SFT data, working Qwen2.5-VL LoRA/QLoRA training and inference code, an unrun offline DPO implementation for the 3B captioner, and a newer experimentally validated 7B-planner-to-3B-captioner pipeline.

No preference training should be launched from the existing DPO config as-is. The repository currently contains two distinct policy scopes that must not be conflated:

1. Caption-policy preference learning: optimize the 3B response `caption` conditioned on `image + plan`.
2. Planner-policy preference learning: freeze the 3B captioner and optimize the 7B response `compact JSON plan` conditioned on `image` using downstream caption utility.

The counterfactual experiment currently supports investigating scope 2 first. It does not yet prove that either scope should receive a full DPO run.

## 2. Repository structure actually present

- `README.md`: inherited V2.5 reranker description. It is stale relative to the current 7B→3B work and references `/home/zhang.jianqiao/projects/v2.5/`, not the active workspace.
- `configs/`: SFT, QLoRA, old 3B DPO, reference-log-probability, teacher-generation, and reranker configs.
- `scripts/`: preprocessing, SFT/DPO training, generation, judge, reranker, blind evaluation, counterfactual evaluation, and cluster job helpers.
- `src/models/`: Qwen2.5-VL LoRA training and inference loaders.
- `src/training/`: multimodal SFT and offline DPO datasets/collators.
- `data/raw/newyorker_caption_ranking/`: New Yorker images, caption rankings, and released GPT-4o descriptions. The audit found 776 files below this raw-data tree.
- `data/processed/`: multiple non-equivalent data versions; see Section 6.
- `outputs/`: adapters, trainer state, generations, blind-evaluation artifacts, and counterfactual results.
- `jobs/`: PJM scripts for Genkai MIG smoke, training, generation, reference-log-probability computation, and evaluation.
- `tests/`: 42 previously reported tests covering data, adapter validation, pipeline inputs, counterfactual construction, and evaluation integrity.
- There is no top-level `train/`, `evaluation/`, `datasets/`, `results/`, `checkpoints/`, or `logs/` directory. Equivalent functions live under `scripts/`, `src/training/`, `outputs/`, and `data/processed/`.

The worktree already contains many modified and untracked experiment files. They are treated as existing user work and must not be deleted, reset, or broadly reformatted.

## 3. Current models and trainable components

### 3.1 Current 7B planner

Config: `configs/lora_sft_caption_aware_viewpoint_v3_7b_qlora.yaml`

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`.
- Task: image → caption-aware compact viewpoint JSON plan.
- Training method: NF4 4-bit QLoRA.
- LoRA rank: 8.
- LoRA alpha: 16.
- LoRA dropout: 0.05.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Bias: none.
- Trainable parameters reported in `outputs/newyorker_caption_aware_viewpoint_v3_7b_qlora/SUBMISSION_RECORD.md`: 5,046,272 / 4,697,304,064, or 0.1074%.
- Peak learning rate: 2e-5.
- Effective batch size: 8 (`batch_size=1`, `gradient_accumulation_steps=8`).
- Epochs/updates: 30 epochs, 300 optimizer steps.
- Vision input budget: `image_max_pixels=100352`.
- Maximum sequence length: 1536.

Adapter inspection: the serialized adapter keys are under `base_model.model.model.language_model.layers.*.self_attn.{q,k,v,o}_proj`. No adapter tensors were found for `visual`, `vision`, `merger`, or `projector` modules.

Consequently:

- Vision encoder: frozen.
- Multimodal projector/merger: frozen.
- Language backbone: frozen except attention LoRA matrices.
- MLP modules (`gate_proj`, `up_proj`, `down_proj`): frozen.
- `lm_head`: frozen.

Implementation: `src.models.qwen_vl_lora_loader.load_qwen_vl_with_lora` creates or loads the PEFT adapter and then explicitly sets every non-`lora_` parameter to `requires_grad=False`.

### 3.2 Current 3B captioner used in joint evaluation

Config: `configs/lora_sft_newyorker_compact_v2_3b_qlora.yaml`

- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Task: `image + three-line humor plan → caption`.
- Training method: NF4 4-bit QLoRA.
- LoRA rank/alpha/dropout: 16 / 32 / 0.05.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Vision encoder/projector/MLP/lm_head: frozen by the same loader logic.
- Checkpoint used by the current joint and counterfactual jobs: `outputs/newyorker_compact_v2_captioner_3b_qlora/best_val_loss`.

The relevant evidence is in:

- `jobs/genkai_eval_caption_aware_v3_joint_vs_direct.pjm`
- `jobs/genkai_eval_plan_counterfactual_v1.pjm`

### 3.3 Checkpoint selection

7B planner candidates:

- Auxiliary minimum-loss adapter: `outputs/newyorker_caption_aware_viewpoint_v3_7b_qlora/best_val_loss`.
- End-of-training adapter: `outputs/newyorker_caption_aware_viewpoint_v3_7b_qlora/final_lora`.

The final adapter is the current downstream selection because it outperformed the minimum-proxy-loss adapter in the blind 7B→3B test, despite a worse proxy validation loss. This is a downstream empirical selection, not proof of population-level superiority.

3B captioner:

- Current evaluation adapter: `outputs/newyorker_compact_v2_captioner_3b_qlora/best_val_loss`.
- A resumable trainer checkpoint exists at `outputs/newyorker_compact_v2_captioner_3b_qlora/checkpoint-453`.

Important configuration defect: `configs/lora_dpo_newyorker_compact_3b.yaml` points to `outputs/newyorker_compact_captioner_3b/best_val_loss`, but that directory currently has no adapter weights. The DPO config also uses the older compact-captioner path rather than the v2 3B adapter used in the current evaluation. It must not be submitted unchanged.

## 4. Training pipeline

### 4.1 SFT

Entrypoint: `scripts/train_lora_sft.py`

Key components:

- `HumorSFTTrainer.compute_loss`: ordinary supervised causal-LM loss with finite-loss and supervised-token guards.
- `ImageBalancedSFTDataset`: samples one caption per image per dataset pass to avoid contests with many captions dominating training.
- `AdapterCheckpointCallback`: saves `latest` and minimum-validation-loss LoRA adapters.
- `FixedGenerationCallback`: optionally creates fixed qualitative validation generations.
- `WallClockCheckpointCallback`: requests periodic recoverable checkpoints.
- `src.training.sft_dataset.HumorSFTDataset`: validates image/caption rows and constructs Qwen multimodal messages.
- `HumorSFTDataset.collate_fn`: masks prompt/image tokens and supervises only assistant response tokens.

The pipeline is reusable and should not be replaced.

### 4.2 Existing DPO

Entrypoint: `scripts/train_lora_dpo.py`

Key components:

- `src.training.dpo_dataset.PreferenceDataset`: requires `image`, `image_id`, `prompt`, `chosen`, and `rejected`.
- `DPOCollator`: encodes chosen and rejected responses independently with the same image/prompt context and masks prompt tokens.
- `sequence_logps`: sums assistant-token log probabilities; it does not length-normalize.
- `scripts.precompute_dpo_reference_logps.py`: precomputes frozen SFT reference log probabilities so only one VLM is resident during DPO.
- `scripts.train_lora_dpo.dpo_metrics`: implements reference-based DPO using summed sequence log probabilities.
- `ImageBalancedPreferenceDataset`: exposes one pair per image per pass.

This is a real offline, image-conditioned DPO implementation for caption responses. It is not planner DPO, does not implement SimPO, has no score-aware weighting, and has no image-conditional counterfactual loss.

It also lacks run metadata required by the new protocol: git commit, data hash, pair-type metrics, trainable-parameter budget, memory/time/throughput, chosen absolute likelihood, and general-capability preservation.

## 5. 7B SFT status

Training evidence:

- `outputs/newyorker_caption_aware_viewpoint_v3_7b_qlora/SUBMISSION_RECORD.md`
- `genkai_train_caption_aware_viewpoint_v3_7b_qlora.pjm.6546345.out`
- `methodexp.md`

Observed values:

- Minimum proxy validation loss: approximately 0.7818 near epoch 7.5.
- Final proxy validation loss: 0.90445.
- Final training loss log: approximately 1.352; aggregate reported train loss 1.539.
- Observed gradient norms were finite and broadly stable; a recorded high value was 1.654 with configured clipping at 1.0.
- No collator truncation was reported.
- Maximum allocated GPU memory reported near 10.25 GiB on one MIG partition.

Interpretation: optimization ran normally, but token-level proxy loss degraded after its minimum and was not aligned with downstream joint-caption utility. Additional epochs on the same 78 labels are not justified.

## 6. Dataset definitions and schemas

### 6.1 Current 7B caption-aware planner SFT

Paths:

- `data/processed/newyorker_compact_viewpoint_sft_caption_aware_v3/train.jsonl`: 78 rows, 78 unique images.
- `data/processed/newyorker_compact_viewpoint_sft_caption_aware_v3/validation_proxy.jsonl`: 24 rows, 24 unique images.

Top-level columns:

- `image`
- `image_id`
- `messages`
- `meta`

Message schema: multimodal user message containing image and the exact planner prompt, followed by an assistant message containing one JSON string.

Plan schema, enforced by the prompt and counterfactual code:

- `scene`
- `type`
- `target`
- `primary_view`
- `views`
- `anchors`
- `external_knowledge`

Metadata fields include `label_provenance`, `manual_review`, `prompt_version`, `task`, `teacher`, `teacher_caption_count`, and `teacher_caption_set_sha256`.

The validation split is explicitly a proxy using an older target style. It is not a clean independent gold planner validation set.

### 6.2 Current 3B compact-v2 captioner SFT

Paths:

- `data/processed/newyorker_compact_sft_v2/caption_train.jsonl`: 13,190 rows over 79 images.
- `caption_validation.jsonl`: 3,990 rows over 24 images.
- `caption_test.jsonl`: 4,415 rows over 24 images.
- `manifest.json`: construction provenance and warnings.

Top-level schema is again `image`, `image_id`, `messages`, and `meta`. Metadata contains continuous `score`, rank, votes, funny votes, and compact-plan provenance.

The 3B captions are conditioned on an older automatic three-line format (`ANCHOR`, `CONTRAST`, `ANGLE`), whereas the latest 7B emits caption-aware compact JSON that is converted into a compatible captioner prompt by `scripts/build_captioner_inputs_from_plans.py`.

### 6.3 Existing caption DPO pairs

Builder: `scripts/build_newyorker_dpo_pairs.py`

Paths:

- `data/processed/newyorker_compact_dpo/dpo_train.jsonl`: 13,190 pairs over 79 images.
- `dpo_validation.jsonl`: 3,990 pairs over 24 images.
- `dpo_test.jsonl`: 4,415 pairs over 24 images.

Schema:

- `image`, `image_id`, `prompt`, `chosen`, `rejected`, `meta`.
- Metadata: contest number, chosen/rejected ranks, rank gap, rejected votes, compact plan, and `pair_type=same_image_rank_gap`.

Construction rule: each top-3%-rank caption is paired with one caption at a fixed lower within-contest percentile. The source has continuous scores, but the pair file does not preserve `chosen_score`, `rejected_score`, `score_margin`, or multidimensional judge values.

This dataset is not yet suitable as the primary hard-pair dataset because:

- it creates many pairs per only 79 training images;
- it does not implement H1–H4 pair types;
- it does not match length, style, grammar, grounding, or meme markers;
- it can include easy/obvious negatives;
- it uses rank-derived preference rather than independent pairwise judgments;
- it is caption-policy data, not 7B planner-policy data.

### 6.4 Scores and judge information

Available scalar source signals:

- New Yorker `score`, `rank`, `votes`, and `funny_votes` in compact-v2 caption SFT rows.
- Historical reranker utilities expect `pos_score`, `neg_score`, and `score_gap`.

Available multidimensional judge implementation:

- `scripts/judge_sft_candidates_qwen.py` outputs integer 1–5 `image_specific`, `naturalness`, `humor`, `format`, and `overall` values for candidates.

Limitations:

- This judge uses the same broad Qwen model family and is not an independent calibrated human judge.
- Current 24-image group evaluation uses one GPT/Codex judge and fixed blind decisions, not stored per-caption multidimensional scores for the four counterfactual conditions.
- There is no current large, human-validated candidate pool with humor, grounding, originality, specificity, and hallucination dimensions.

## 7. Candidate generation, reranking, and evaluation

### 7.1 Candidate generation

- `scripts/generate_lora_sft.py`
- `src.models.qwen_vl_lora_inference.load_qwen_vl_lora_for_inference`
- `generate_candidates`: supports temperature, top-p, max tokens, repetition penalty, seed, and multiple return sequences.
- `run_generation`: records image, prompt, gold captions, and candidate list.

The generator does not currently record top-k, resolved generation config, checkpoint hash, or per-candidate RNG identity in each output file. Best-of-N diagnostics should add a manifest.

### 7.2 Existing reranker

- Config: `configs/humor_reranker.yaml`.
- Trainer: `scripts/train_humor_reranker.py`.
- Model classes: `PairDataset`, `PairCollator`, `RerankerHead`, and `HumorReranker` are defined inside the script rather than a reusable `src/reranker/` module.
- Backbone: CLIP ViT-B/32, frozen by default.
- Head: learned MLP over image/text embeddings and interaction features.
- Loss: weighted pairwise softplus loss.
- Split: `split_by_image`, which is appropriate for leakage control.

However, the processed reranker directories described by `README.md` are absent from this active workspace, and no trained reranker checkpoint was found under `outputs/`. Therefore there is code for a reranker, but no currently deployable reranker artifact.

### 7.3 Current evaluation

Absolute blind evaluation:

- `docs/CAPTION_AWARE_V3_BLIND_EVAL.md`
- `scripts/build_multisystem_blind_caption_comparison.py`
- `scripts/report_multisystem_blind_caption_comparison.py`

Counterfactual group evaluation:

- `scripts/build_counterfactual_plan_inputs.py`
- `scripts/build_group_winrate_eval.py`
- `scripts/report_group_winrate_eval.py`
- `docs/COUNTERFACTUAL_GROUP_WINRATE_EVAL.md`
- `outputs/newyorker_caption_aware_v3_counterfactual_eval/group_winrate_report.json`

Metrics already available:

- absolute good-caption and strong-caption rates;
- average ordinal score;
- fraction of images with at least one good candidate;
- Group Overall win rate;
- Group Best-Pick win rate;
- image-level paired bootstrap interval for absolute-rate differences;
- Wilson intervals for group win rates;
- A/B position-choice counts.

Missing metrics include calibrated multi-rater agreement, originality, hallucination, generic meme rate, distinct-n, self-BLEU/semantic diversity, general-capability preservation, and pair-type-specific accuracy.

## 8. Current best experiment and failure cases

Current downstream point-estimate leader:

`Qwen2.5-VL-7B final_lora planner → compact JSON → Qwen2.5-VL-3B best_val_loss captioner`

Absolute blind evaluation over 24 images × 3 captions:

- Joint final good-caption rate: 24/72 = 33.3%.
- Joint best-proxy-loss good-caption rate: 19/72 = 26.4%.
- Direct 3B good-caption rate: 16/72 = 22.2%.
- Joint-final minus direct image-level bootstrap interval: [-6.9, +27.8] percentage points.

Counterfactual Group-of-3 Overall results:

- Correct plan vs direct: 58.3%, Wilson 95% CI [38.8%, 75.5%].
- Correct vs fully swapped plan: 79.2%, [59.5%, 90.8%].
- Correct vs corrupted humor target: 75.0%, [55.1%, 88.0%].
- Swapped vs direct: 29.2%, [14.9%, 49.2%].
- Corrupted vs direct: 20.8%, [9.2%, 40.5%].

Supported conclusion: the 3B captioner uses both the plan and the proposed humor bridge. Unsupported conclusion: the present joint system is reliably better than direct 3B inference.

Observed failure modes:

1. Planner visual misidentification propagates into all downstream captions. The documented image 548 example incorrectly interpreted a rocket-like vehicle as being cut like a fish.
2. Plans sometimes restate an abnormal scene without producing a reusable double meaning or social-script mapping.
3. The 3B captioner follows incorrect plans strongly; wrong plans can be worse than no plan.
4. Many generated captions remain generic, literal, incoherent, or weakly tied to the image. In the absolute evaluation, 48/72 joint-final captions were below the good threshold.
5. The planner SFT set is extremely small and caption-aware labels may contain hindsight rationalization derived from gold captions.
6. The current evaluation has only 24 images and one model judge. Position is blinded, but the judge is not independent of all prior exposure.
7. The current three-caption group protocol is a scaled diagnostic inspired by, not a strict reproduction of, the NeurIPS 2024 Group-of-10 evaluation.

## 9. Preference-learning readiness assessment

### 9.1 What is already supported

- There is evidence that plan quality has downstream causal utility: correct plans substantially beat swapped and target-corrupted plans.
- There is a reusable multimodal chosen/rejected collator and mathematically recognizable offline DPO loss for caption-policy experiments.
- Scalar human-derived ranking scores exist in the New Yorker source data.
- The current generation and blind-evaluation pipelines can be extended without replacement.

### 9.2 Audit-time gaps and execution status

- Best-of-N: completed for N={1,2,4,8,16,32} on joint and direct policies; see `docs/preference_learning_diagnostics.md`.
- Image shuffle log-probability margin: completed on 61 image-diverse H2 pairs.
- Humor representation probe: completed on 53 image-disjoint triplets.
- Per-layer/per-module preference gradients: completed for the currently trainable q/k/v/o LoRA scope; MLP/vision/projector remain unmeasured.
- SFT-vs-preference gradient cosine: completed on 32 image-diverse H2 pairs.
- Hard-pair bias and confounders: completed for 485 conservative H2 pairs; H1/H3/H4 remain unavailable.
- Planner preference pairs based on frozen-3B downstream utility: do not exist.

### 9.3 Objective recommendation after diagnostics

No objective is selected as the winner yet.

- DPO is the required reference-based baseline.
- SimPO is a required short-caption, length-normalized, reference-free baseline.
- An mDPO-inspired image-conditional term now has high priority because image-shuffle sensitivity was weak; it must be labeled an experimental adaptation unless implemented exactly from the paper.
- Score-aware weighting is justified only after score reliability and pair-margin calibration are checked.
- Chosen-anchor regularization is justified only if the chosen absolute likelihood falls while the relative margin rises.

For planner-policy learning, ordinary caption DPO code cannot be reused unchanged. The response tokens must be compact plan tokens, and pair labels must be constructed from a frozen downstream 3B utility measurement plus a visual-factuality gate.

## 10. Minimal-intrusion implementation plan

1. Add diagnostic scripts under `scripts/preference_diagnostics/` while reusing `generate_lora_sft.py`, `qwen_vl_lora_loader.py`, `dpo_dataset.py`, and `src.utils.io`.
2. Keep all generated artifacts under `results/preference_diagnostics/`; do not alter SFT outputs.
3. Make diagnostics accept explicit policy scope (`captioner` or `planner`) in manifests.
4. Build score-preserving, same-image hard pairs with explicit H1–H4 types and bias reports; do not replace `data/processed/newyorker_compact_dpo/`.
5. Run CPU schema/unit tests first. GPU execution requires separate smoke jobs and is not part of this audit.
6. Only after diagnostic results exist, select policy scope, target modules, objective, adapter strategy, and training budget.

## 11. Authoritative references

1. Hessel et al. (2023), *Do Androids Laugh at Electric Sheep? Humor Understanding Benchmarks from The New Yorker Caption Contest*, ACL 2023 Best Paper. https://aclanthology.org/2023.acl-long.41/
2. Zhang et al. (2024), *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning*, NeurIPS 2024 Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html
3. Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
4. Meng et al. (2024), *SimPO: Simple Preference Optimization with a Reference-Free Reward*, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/e099c1c9699814af0be873a175361713-Abstract-Conference.html
5. Wang et al. (2024), *mDPO: Conditional Preference Optimization for Multimodal Large Language Models*. https://arxiv.org/abs/2406.11839
6. Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022. https://openreview.net/forum?id=nZeVKeeFYf9
7. Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html
