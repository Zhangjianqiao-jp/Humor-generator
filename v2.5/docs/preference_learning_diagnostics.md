# Preference Learning Diagnostics

Status date: 2026-08-24 (Asia/Tokyo)

This report records the completed Phase-1 evidence and the remaining data-annotation gaps. No DPO/SimPO training has started.

## 1. Current model status

The currently evaluated system is:

`image → Qwen2.5-VL-7B final planner LoRA → compact JSON plan → frozen Qwen2.5-VL-3B best-validation-loss captioner LoRA → caption group`

Both adapters are attention-only QLoRA. The vision encoder, multimodal merger/projector, MLP, and language-model head are frozen. Full repository details are in `docs/preference_learning_audit.md`.

## 2. Is preference optimization justified?

Current decision: the policy has a clear Best-of-N ranking opportunity, so a controlled preference-optimization pilot is scientifically justified. The current automatically derived H2-only pairs are not sufficient for the final run: they cover only 61 images, omit H1/H3/H4, and the auxiliary judge is context-sensitive. Formal preference training should wait for an independently or human-validated, image-disjoint pair version; a clearly labeled H2-only engineering pilot may be used only to validate the trainer.

Evidence in favor:

- Correct plans beat fully swapped plans at 79.2% Group Overall win rate.
- Correct plans beat target-corrupted plans at 75.0%.
- Swapped and corrupted plans perform worse than direct 3B inference.
- Therefore the 3B captioner uses the planner and the humor bridge has downstream utility.

Completed diagnostic evidence:

- Correct-plan joint inference beats direct inference only at a 58.3% point estimate; its earlier 95% interval includes 50%.
- Best-of-N now shows that both joint-conditioned and direct caption policies contain high-quality modes whose probability is low at N=1.
- Image-shuffle margins show no established advantage for the correct raw image when the compact plan is fixed.
- The late hidden layer contains the strongest linearly accessible humor signal, but funny-vs-weak separation is only moderate.
- Preference gradients inside the current attention LoRA concentrate in `o_proj`/`v_proj` and align positively with chosen-caption SFT gradients.

Decision rule:

- If Best-of-N rises strongly while mean quality stays relatively flat, preference ranking is a plausible bottleneck.
- If Best-of-32 remains weak, prioritize new SFT/contrastive representation shaping rather than DPO.
- If image-shuffle delta margins are near zero, vanilla DPO is unsafe as the sole objective; test an explicitly labeled mDPO-inspired conditional term.

## 3. Best-of-N capability

Implementation: `scripts/preference_diagnostics/best_of_n.py`

Artifacts:

- `results/preference_diagnostics/best_of_n_joint_vs_direct_v1/{joint,direct}/best_of_n.csv`
- `results/preference_diagnostics/best_of_n_joint_vs_direct_v1/{joint,direct}/best_of_n_humor.png`
- `results/preference_diagnostics/best_of_n_joint_vs_direct_v1/{joint,direct}/best_of_n_success_rate.png`
- `results/preference_diagnostics/best_of_n_joint_vs_direct_v1/paired_comparison.csv`

Design:

- N values: 1, 2, 4, 8, 16, 32.
- One max-N pool is sampled per image; smaller N values use nested prefixes of the same pool.
- Generation config, seed, model/adapter, input hashes, and judge score field are recorded.
- Metrics: mean prefix maximum, mean candidate score, and fraction of images with at least one caption above the fixed threshold.

Both 24-image, 32-candidate pools and their strict auxiliary scores are complete (768 validated scores per system).

| System | N | H_max | H_mean | P_good (score >= 4) |
|---|---:|---:|---:|---:|
| joint | 1 | 2.792 | 2.792 | 0.292 |
| joint | 4 | 3.708 | 2.844 | 0.750 |
| joint | 8 | 3.917 | 2.818 | 0.917 |
| joint | 32 | 3.958 | 2.776 | 0.958 |
| direct | 1 | 2.500 | 2.500 | 0.042 |
| direct | 4 | 3.625 | 2.740 | 0.708 |
| direct | 8 | 3.917 | 2.734 | 0.917 |
| direct | 32 | 4.042 | 2.723 | 1.000 |

For joint, H_max gains 1.167 points from N=1 to N=32 while H_mean changes by -0.016. For direct, H_max gains 1.542 while H_mean gains only 0.223. This is the expected signature of a useful low-probability generation mode and supports preference/reranking work rather than another identical positive-only SFT pass.

The 50,000-sample paired bootstrap comparison finds a joint-minus-direct N=1 H_max difference of +0.292 with 95% interval [-0.125, 0.750]. The N=1 P_good difference is +0.250 with interval [0.042, 0.458]. At N=8 both systems have identical H_max and P_good point estimates; at N=32 the H_max difference is -0.083 with interval [-0.208, 0.000]. Thus the planner-conditioned prompt may improve one-shot success under this scorer, but it does not increase the observed high-N capability ceiling.

A dedicated strict auxiliary scorer used fixed two-candidate chunks, validated every score in [1, 5], saved each completed image atomically, and supported restart. The original general judge silently omitted candidates and used 0/1 despite requesting 1--5; those outputs were rejected. A batch-size-4 smoke also produced materially different absolute scores from batch size 2 on the same four captions. Consequently these absolute scores are auxiliary diagnostics, not approved preference labels. Independent blind group win rate or human review is required before training-data construction.

## 4. Image dependency

Implementation: `scripts/preference_diagnostics/image_shuffle.py`

It computes the chosen-minus-rejected summed response log-probability margin under the correct image and a deterministic one-to-one wrong-image assignment, then reports their difference.

The earlier plan-counterfactual test already establishes plan sensitivity, but it is not the same diagnostic: it changes the plan and evaluates generated caption groups, whereas this script changes the image and measures policy log probabilities for fixed caption pairs.

Formal result on one eligible H2 pair from each of 61 distinct training images:

- Mean delta margin: 0.133.
- Median delta margin: 0.370.
- Bootstrap 95% interval for the mean: [-0.491, 0.788].
- Positive deltas: 33/61 = 54.1%.
- Wilson 95% interval for the positive fraction: [41.7%, 66.0%].

Interpretation: with the compact plan held fixed, the 3B captioner shows no established preference-margin advantage for the correct raw image over a shuffled image. This is evidence of weak independent image checking by the 3B caption policy. It is not evidence that the full system contains no visual information, because the plan itself was generated from and describes the correct image. For caption-policy training, vanilla DPO alone is therefore at risk of learning plan/text shortcuts; an image-conditional ablation has high priority.

## 5. Humor representation probe

Implementation: `scripts/preference_diagnostics/humor_representation_probe.py`

- Requires image-grounded examples labeled exactly `funny`, `weak`, and `literal`.
- Pools assistant-response token representations at early, middle, late, and final layers.
- Uses an image-disjoint train/test split.
- Fits only a linear classifier.
- Reports accuracy, macro-F1, and macro one-vs-rest AUROC.

Probe input construction: `scripts/preference_diagnostics/build_probe_examples.py`.

The current controlled set contains 53 images and 159 examples: 53 each of `funny`, `weak`, and `literal`. Funny/weak captions are the best conservative H2 pair per image. Literal examples are released GPT-4o visual descriptions; eight additional images were excluded because the literal text was more than 65% different in character length from the funny/weak pair. This is not equivalent to a human-written literal-caption control, so the formal analysis reports both three-class metrics and a stricter funny-vs-weak binary probe.

GPU executable smoke passed on eight images. The formal image-disjoint run then completed on all 53 images: 40 train images (120 examples) and 13 test images (39 examples).

| Layer | 3-class accuracy | 3-class macro AUROC | Funny-vs-weak accuracy | Funny-vs-weak AUROC |
|---|---:|---:|---:|---:|
| early | 0.718 | 0.846 | 0.615 | 0.577 |
| middle | 0.718 | 0.864 | 0.538 | 0.536 |
| late | **0.821** | **0.907** | **0.615** | **0.658** |
| final | 0.667 | 0.832 | 0.500 | 0.447 |

Interpretation: the late representation contains a strong linearly accessible signal for separating literal, weak, and funny text, but some of that result can reflect the residual literal-description style difference. The stricter funny-vs-weak result is only moderate in the late layer and weak elsewhere. Therefore the model has evidence of a humor-related ranking signal, but this diagnostic does not establish that strong humor generation is already solved. The held-out funny-vs-weak test contains only 26 examples, so this result needs more human-controlled literal/weak annotations or repeated image-disjoint splits before a strong representational claim.

## 6. Preference pair construction

Implementations:

- `scripts/preference_diagnostics/build_preference_pairs.py`
- `scripts/preference_diagnostics/pair_bias_analysis.py`

CPU result from current New Yorker scalar-score data:

- Input captions: 13,190 over 79 training images.
- Held-out test IDs excluded explicitly.
- Conservative length/style-matched output: 485 pairs.
- H1: 0.
- H2: 485.
- H3: 0.
- H4: 0.

H1/H3/H4 are not measured because the current source rows do not contain reliable grounding, originality, specificity, and literal-caption annotations. They are not interpreted as absent from the task.

Bias report: `results/preference_diagnostics/current_rank_pairs/pair_bias_report.md`.

Observed surface differences are small: chosen captions average 47.63 characters and rejected captions 46.92; no configured heuristic shortcut crossed the current warning threshold. This does not validate visual grounding or humor semantics.

These 485 pairs are a diagnostic candidate set, not an approved DPO training set. They come from only 61 pair-producing images and cover only H2.

## 7. Preference gradient concentration

Implementation: `scripts/preference_diagnostics/module_gradient_analysis.py`

Outputs:

- `module_gradient_scores.csv`
- `module_group_summary.csv`
- `layer_module_gradient_heatmap.png`
- `summary.json`

Metrics:

- RMS batch Frobenius gradient norm.
- Norm divided by square root of trainable parameter count.
- Norm relative to the trainable adapter-weight norm.

Coverage guard: the current adapter exposes only q/k/v/o LoRA parameters. Therefore the first run can compare attention layer/module gradients but cannot rank MLP, vision, projector, or lm_head modules. Those families will be marked `not_measured`, not assigned zero importance. A later diagnostic adapter or non-quantized analysis is required for an attention-vs-MLP selection claim.

Formal first-pass result on 32 image-diverse H2 pairs:

| Module family | Normalized gradient norm | Relative gradient norm | Parameters |
|---|---:|---:|---:|
| `o_proj` | 0.05319 | 5.6628 | 2,359,296 |
| `v_proj` | 0.04970 | 4.0396 | 1,327,104 |
| `k_proj` | 0.01326 | 1.0920 | 1,327,104 |
| `q_proj` | 0.01213 | 1.3218 | 2,359,296 |

Within the current attention adapter, preference sensitivity concentrates much more strongly in `o_proj` and `v_proj` than in `q_proj` and `k_proj`. Several high-normalized-gradient locations occur in later `o_proj` layers, but layer 0 `v_proj` is also high, so the pattern is not simply “train only late layers.” These are gradients on only 32 H2 pairs and only existing LoRA weights.

## 8. SFT-versus-preference gradient alignment

Implementation: `scripts/preference_diagnostics/module_gradient_analysis.py --compute-sft-alignment`.

The chosen-response token-average SFT NLL gradient was compared with the negative chosen-minus-rejected margin gradient on the same 32 image-diverse H2 pairs. Results over the currently trainable attention LoRA parameters:

| Module family | Aggregated cosine | Negative layer cosines |
|---|---:|---:|
| `q_proj` | 0.691 | 0/36 |
| `k_proj` | 0.689 | 0/36 |
| `v_proj` | 0.693 | 0/36 |
| `o_proj` | 0.682 | 0/36 |

The global concatenated cosine is 0.686; all 144 measured layer×module cosines are positive (range 0.612 to 0.767). For this H2 subset and the existing attention adapter, preference gradients broadly reinforce rather than undo the chosen-caption SFT direction. This does not establish alignment for MLP, vision/projector modules, or H1/H3/H4 objectives. Continuing the current adapter is therefore a defensible baseline, while a separate preference adapter remains useful for rollback and controlled comparison rather than being mandated by observed conflict.

## 9. Recommended target modules

No global target-module winner is selected yet.

The current attention-only setting is the known stable baseline, not an evidence-based optimum. The first measured selective candidate inside that scope is `o_proj + v_proj`, which should be compared against all four attention projections under a matched LoRA-parameter budget. MLP-only and all-linear remain unmeasured and cannot be ranked until a diagnostic adapter covers them. All configurations must be compared under approximately equal trainable-parameter budgets.

## 10. Recommended objective

Recommended objective matrix after diagnostics:

1. DPO: required reference-based baseline with the frozen SFT policy as reference.
2. SimPO: equally important baseline because captions are short and summed sequence log probabilities carry length effects; use the same validated pairs and update budget as DPO.
3. Conditional multimodal preference: now high priority because image-shuffle dependence was weak. Label the local implementation experimental unless it exactly reproduces mDPO.
4. Score-aware weighting: postpone until scalar-score calibration and margin reliability are measured.
5. Chosen anchor: enable only if chosen absolute log probability falls while relative preference margin increases.

Planner-policy and caption-policy experiments must remain separate. The scientifically preferred first planner experiment freezes the 3B and labels same-image plan pairs by downstream caption-group utility plus a hard visual-factuality gate.

## 11. Recommended first training experiments

Phase-1 diagnostics are complete. Before a result-bearing preference run, freeze an image-disjoint pair-data version with human or independent blind validation and add H1/H3/H4. Do not train on auxiliary judge scores from this diagnostic.

If diagnostics support preference optimization, the first controlled experiments should be:

1. Frozen SFT baseline.
2. Optional trainer-only H2 pilot, explicitly excluded from scientific claims.
3. DPO with the current stable attention-only `q/k/v/o` adapter scope on validated pairs.
4. SimPO with the same pair data and approximately matched update/token budget.
5. Only then, budget-matched attention-only, MLP-only, all-linear, and selective-module ablations.
6. Conditional multimodal preference with correct-vs-shuffled image terms and explicit H3/H4 data.

Every result must report humor, grounding, originality, specificity, diversity, generic-meme rate, hallucination, length, pair-type metrics, trainable parameters, memory, time, throughput, and preservation checks.

## 12. Reproducibility status

- Audit written: yes.
- Diagnostic config written: yes, `configs/preference_diagnostics_captioner_3b.yaml`.
- CPU compilation and CLI smoke: passed.
- Focused diagnostic tests: 11 passed, including strict judge completeness/index handling and paired Best-of-N comparison.
- Full repository regression after all current diagnostic changes: 56 passed, 15 upstream/runtime warnings.
- GPU diagnostic smoke: job `6577397` passed.
- Formal image-shuffle and current-adapter gradient run: job `6577415` passed.
- Best-of-N one-image executable smoke: job `6577431` passed.
- Humor-probe eight-image executable smoke: job `6577465` passed.
- Formal humor representation probe: job `6577566` passed.
- SFT-versus-preference alignment smoke and formal run: job `6577590` passed.
- Formal joint-vs-direct Best-of-N candidate generation and validated scoring: completed; final scoring jobs `6577756` (joint) and `6577730` (direct) passed.
- Preference training: not started.

## 13. Authoritative references

1. Hessel et al. (2023), *Do Androids Laugh at Electric Sheep?*, ACL 2023 Best Paper. https://aclanthology.org/2023.acl-long.41/
2. Zhang et al. (2024), *Humor in AI*, NeurIPS 2024 Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html
3. Rafailov et al. (2023), *Direct Preference Optimization*, NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
4. Meng et al. (2024), *SimPO*, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/e099c1c9699814af0be873a175361713-Abstract-Conference.html
5. Wang et al. (2024), *mDPO: Conditional Preference Optimization for Multimodal Large Language Models*. https://arxiv.org/abs/2406.11839
