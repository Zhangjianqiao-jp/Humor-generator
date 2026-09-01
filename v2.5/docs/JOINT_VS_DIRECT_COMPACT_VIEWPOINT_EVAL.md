# Compact-Viewpoint 7B + 3B Joint Inference Evaluation

## Decision

Do not start joint DPO from the current two-model pipeline. The new joint system did not beat the same 3B captioner used without a plan, and manual inspection found material visual-grounding errors in the generated plans. Preference optimization cannot repair a planner that describes the wrong visual mechanism.

## Reproducibility

- PJM job: `6474451`
- Test images: 24 held-out New Yorker cartoons, one row per image
- Planner: `Qwen2.5-VL-7B-Instruct` + `outputs/newyorker_compact_viewpoint_7b_qlora/best_val_loss`
- Captioner: `Qwen2.5-VL-3B-Instruct` + `outputs/newyorker_compact_v2_captioner_3b_qlora/best_val_loss`
- Planner prompt: `prompts/7b_image_to_compact_viewpoint.txt`
- Planner prompt SHA-256: `2b5bf47892a606f395a7e21152101d185b427c1cebda9608d1819a475154204a`
- Planner decoding: deterministic, one plan per image, seed 4242
- Caption decoding: three candidates per image and system, seed 7777
- Evaluation: blind single-rater 0-3 rubric; score >=2 is good
- Bootstrap: paired resampling over the 24 images, 20,000 repeats

The joint and direct conditions use the same image, 3B adapter, candidate count, seed, and sampling parameters. The only intended difference is whether the compact JSON plan is supplied.

## Structural Gates

- Planner outputs: 24/24 valid compact-viewpoint JSON objects
- Planner prompt leakage: 0
- Joint captions: 72/72 structurally valid
- Direct captions: 72/72 structurally valid
- Caption prompt leakage: 0
- Both LoRA adapters passed finite-value and A/B pairing validation

## Blind Results

| Metric | Joint 7B -> 3B | Direct 3B | Joint - Direct |
| --- | ---: | ---: | ---: |
| Good candidates | 7/72 | 9/72 | -2 |
| Good-candidate rate | 9.72% | 12.50% | -2.78 pp |
| Mean score | 0.639 | 0.625 | +0.014 |
| Images with >=1 good candidate | 7/24 | 9/24 | -2 |
| Image hit rate | 29.17% | 37.50% | -8.33 pp |
| Strong captions (score 3) | 0 | 0 | 0 |

- Good-candidate-rate difference 95% bootstrap CI: `[-12.50 pp, +6.94 pp]`
- Image-hit-rate difference 95% bootstrap CI: `[-37.50 pp, +20.83 pp]`
- Per-image best-candidate wins: joint 6, direct 10, ties 8

The intervals include zero. This test does not establish a statistically reliable difference, but it also provides no evidence that the planner improves the captioner.

## Failure Analysis

The planner learned JSON syntax better than reliable visual grounding. At least 8 of the 24 held-out plans contain a clear core-scene or joke-mechanism error. Examples include:

- `nycc_548`: rocket-shaped vehicle described as a car with a shark fin.
- `nycc_578`: a crowd entering with food described as police carrying a suspect.
- `nycc_579`: the crowned animal-skin scene described as a live crowned dog on a couch.
- `nycc_590`: the cave/animal-costume scene described as a giraffe in a tunnel with flying animals.
- `nycc_603`: the mountaintop restaurant context and its joke mechanism are not captured.
- `nycc_617`: the hell/devil setting is reduced to naked people watching a burning building, followed by an unsupported metaphor.
- `nycc_632`: the matryoshka-doll office visual is reduced to a generic teacher/student scene.
- `nycc_652`: the UFO/sundae visual is described as a boat filled with candy.

There is also an interface distribution shift. The 3B captioner was trained with a three-line `ANCHOR/CONTRAST/ANGLE` plan, while this experiment supplies the newly requested compact JSON inside a richer instruction wrapper. This does not invalidate the measured deployed pipeline, but it prevents attributing all failures solely to the planner.

## Next Gate

Before any joint preference optimization:

1. Render the same generated JSON into the 3B-native three-line format and rerun the paired test. This is inference-only and isolates the interface mismatch.
2. Add a shuffled-plan condition. A useful planner must beat both direct inference and a plan taken from another image.
3. Audit and repair planner labels/grounding, emphasizing hard visual relations instead of schema imitation.
4. Train a plan-aware reranker on `(image, plan, caption)` with same-image low-ranked captions, model-generated hard negatives, and shuffled-plan counterfactual negatives.
5. Consider 3B-only DPO only after correct-plan inference beats direct and shuffled-plan inference on held-out images.

Joint DPO over both models is not recommended: the sampled JSON is a discrete interface, so caption preference does not provide a stable gradient or clear credit assignment to the 7B planner. Train the planner with matched-versus-shuffled plan contrastive supervision and train caption preferences on the 3B separately.

## Artifacts

- `outputs/newyorker_compact_viewpoint_joint_vs_direct_3b/planner_test_generations.jsonl`
- `outputs/newyorker_compact_viewpoint_joint_vs_direct_3b/joint_inputs.jsonl`
- `outputs/newyorker_compact_viewpoint_joint_vs_direct_3b/joint_captions.jsonl`
- `outputs/newyorker_compact_viewpoint_joint_vs_direct_3b/direct_3b_captions.jsonl`
- `outputs/newyorker_compact_viewpoint_joint_vs_direct_3b/blind_candidates.jsonl`
- `outputs/newyorker_compact_viewpoint_joint_vs_direct_3b/blind_scores.json`
- `outputs/newyorker_compact_viewpoint_joint_vs_direct_3b/gpt_blind_evaluation.json`

## References

- Rafailov et al., *Direct Preference Optimization*, https://arxiv.org/abs/2305.18290
- Hu et al., *LoRA*, https://arxiv.org/abs/2106.09685
- Dettmers et al., *QLoRA*, https://arxiv.org/abs/2305.14314
- Hessel et al., *The New Yorker Caption Contest Dataset*, https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html
