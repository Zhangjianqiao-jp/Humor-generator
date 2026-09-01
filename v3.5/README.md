# Humor Generator v3.5

v3.5 is an independent, fail-closed experiment tree for testing whether a
frozen 7B HOMER-style Planner can communicate conflict and associative
imagination to a frozen image-conditioned 7B Generator more effectively in
continuous states than in text. It never imports or executes v2.5/v3.0 code.

## Scientific scope

Two systems are deliberately separated:

1. `text_homer` is a method/data reproduction of public HOMER components with
   a pinned Qwen2.5-VL substitution. Its caption stage follows HOMER and uses
   description + selected conflict/path, without the raw image.
2. Communication experiments keep the project Generator's SFT interface:
   `image + instruction + optional plan/prefix -> caption`. All communication
   conditions therefore receive the same raw image and frozen receiver.

The second family includes full-plan text, 8-token-per-channel budget text,
token embeddings, a channel-preserving StateBridge adaptation, Learned bridge,
and Typed bridge. Learned/Typed use exactly 24 slots and the same trainable
parameter count. StateBridge's official 64-token homogeneous-agent setting is
an appendix baseline, not mislabeled as the parameter/bandwidth-matched main
comparison.

## Non-negotiable gates

- Both 7B policies remain frozen; only a bridge may be trainable.
- Every Planner trace contains strict HOMER conflict/local/global schemas,
  exact causal hidden-state/token alignment, actual generated semantics, and a
  SHA-256 digest. Formal traces additionally pin the clean Git commit and the
  dataset, prompt-source, and frozen-adapter manifest hashes.
- Training uses description-nearest, different-conflict hard negatives.
- The SFT receiver always receives the image. Text teacher and latent student
  use the same caption targets and the same three plan channels.
- No preference learning starts until latent communication has a stable,
  absolute-quality-preserving held-out gain.

## Data partitions

The immutable manifest currently contains 810 image clusters:

| split | clusters | role |
|---|---:|---|
| train | 602 | bridge fitting only |
| validation | 64 | checkpoint selection only |
| internal_test | 97 | sealed primary test |
| official_hia_unseen_test | 24 | official HIA images unseen by the SFT adapters |
| official_hia_seen_diagnostic | 23 | official HIA images seen during SFT; diagnostic only |

All 47 official HIA test images are excluded from bridge fitting. The 23
adapter-seen images must never be reported as uncontaminated held-out evidence.
The 121 primary unseen images yield 1,210 captions per system in the 10-seed
diversity run; this is not misreported as 1,000 independent images.

## Reproduce the engineering gates

```bash
cd /home/pj26000152/ku60000936/projects/Humor-generator/v3.5
./scripts/bootstrap_environment.sh
.venv/bin/pytest -q
.venv/bin/python scripts/check_environment.py
.venv/bin/python scripts/check_v35_isolation.py
.venv/bin/python scripts/verify_frozen_artifacts.py
.venv/bin/python scripts/verify_clustered_dataset.py
```

The real-trace GPU engineering smoke has passed. The first training stage is
only three serial 64-train/24-validation SFT-receiver pilots: Learned+KL,
Typed+KL, Typed without KL. Subsets use seeded hash sampling. After optimization,
one GPU job creates fixed 3-seed Text-HOMER, StateBridge, and learned-bridge
validation generations plus an anonymous mirrored Group-of-3 pilot packet;
automation stops for independent rating. Group-of-10 remains the confirmatory
paper endpoint. Full-data and Base-receiver
experiments are evidence-gated follow-ups, not an automatic matrix.

## Evaluation

The primary evaluation uses Group-of-10, matching the scale of the NeurIPS
2024 Humor-in-AI benchmark; Group-of-3 is retained only as a legacy sensitivity
analysis. Formal packets use both A/B orientations, randomize caption order,
and keep primary and mechanistic comparisons in separate correction families. Reports
include overall win rate, best-pick win rate, candidate `good/weak/bad`, seed
variance, hierarchical image/rater bootstrap intervals, Krippendorff nominal
alpha, and Holm-adjusted comparisons. Ten-candidate diversity reports include
the official all-mpnet-base-v2 SBERT diversity and EAD metrics, plus descriptive
Distinct-1/2, self-BLEU, TF-IDF distance, Vendi score, and human angle coverage;
diversity is also recomputed on good-only candidates.

## Authoritative references

- Shang et al., HOMER, ICLR 2026: https://openreview.net/pdf?id=SzaRhPom4o
- HOMER official implementation: https://github.com/Shang-hub/HOMER-Official-Implementation
- Du et al., InterLat, ACL 2026: https://aclanthology.org/2026.acl-long.1248/
- Peng et al., StateBridge, COLM 2026: https://arxiv.org/abs/2608.13317
- Zhang et al., Humor in AI, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf
- Hessel et al., Electronic Sheep, ACL 2023: https://aclanthology.org/2023.acl-long.41/
- Tevet & Berant, diversity evaluation, EACL 2021: https://aclanthology.org/2021.eacl-main.25/
- Friedman & Dieng, The Vendi Score, TMLR 2023: https://arxiv.org/abs/2210.02410
