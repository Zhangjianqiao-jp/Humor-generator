# Humor Generator v3.5

v3.5 is an independent, fail-closed experiment tree for testing whether a
frozen, pretrained Qwen2.5-VL-7B Planner can communicate conflict and
associative imagination to a frozen, pretrained Qwen2.5-VL-7B Generator more
effectively in continuous states than in text. Only the bridge is trainable in
the current route; the old Planner/Generator SFT adapters are quarantined
artifacts and are not loaded by the current protocol. It never imports or
executes v2.5/v3.0 code.

Every method claim is classified as implemented, literature-supported but
adapted, project-specific, or not yet run in
[`docs/METHOD_CITATION_EVIDENCE_ZH.md`](docs/METHOD_CITATION_EVIDENCE_ZH.md).
The annotation-by-annotation reproduction decision and current route changes
are recorded in
[`docs/HOMER_ANNOTATION_RECONCILIATION_ZH.md`](docs/HOMER_ANNOTATION_RECONCILIATION_ZH.md).

## Scientific scope

Two systems are deliberately separated:

1. `public_code_exact` preserves the prompt text and stage order of the pinned
   HOMER repository. The paper reports a 2+3+2 budget (7), while the released
   `generator.py` executes three generator requests, so an online run has 8
   requests; both facts are recorded instead of being conflated.
2. `pretrained_7b_bridge` uses the same staged protocol with the local pinned
   Qwen2.5-VL-7B revision as a model substitution. Both Planner and Generator
   are frozen with `adapter: null`; only the bridge is trainable. See
   [`docs/HOMER_PRETRAINED_7B_PROTOCOL_ZH.md`](docs/HOMER_PRETRAINED_7B_PROTOCOL_ZH.md)
   and the detailed audit in
   [`docs/HOMER_REPRODUCTION_EVALUATION_AUDIT_ZH.md`](docs/HOMER_REPRODUCTION_EVALUATION_AUDIT_ZH.md).

The tested base model is an explicit factor: additional GPT-4o, Claude-4,
Qwen-VL or LLaVA-1.5 runs must be separate `model_variant`s with their own
traces and identical Pass@K protocol. Candidates from different models are
never pooled into one denominator.

The communication family includes full-plan text, 8-token-per-channel budget
text, token embeddings, a channel-preserving StateBridge adaptation, Learned
bridge, and Typed bridge. Learned/Typed use exactly 24 slots and the same
trainable parameter count. StateBridge's official 64-token homogeneous-agent
setting is an appendix baseline, not mislabeled as the parameter/bandwidth-
matched main comparison. Planner traces for the current route must be rebuilt
with the full HOMER stage order; historical SFT traces cannot be reused.

The current caption-stage bridge route is separate from the historical A5
trainer.  Run `scripts/cache_pretrained_homer_context.py` after the
adapter-free Planner trace gate: it replays official summary, retrieval,
conflict/entity selection and seeded DFS path sampling from the raw cached
Planner responses.  Then run `scripts/build_pretrained_bridge_dataset.py` and
use `configs/pilot/cross_attention_caption_pretrained_public.yaml`.  This
route uses the pinned public `##Caption`/`##Explanation` prompt for both the
text teacher and latent student; the student receives the same description
with the selected plan removed and the bridge memory injected.  The old
`cross_attention_caption_pretrained.yaml`, `formal_bridge.py` legacy prompt,
and `latent_bridge_v35` data are historical and cannot support a current HOMER
claim.
The complete contract, target policy, provenance fields and fail-closed order
are documented in [`docs/HOMER_PUBLIC_BRIDGE_ROUTE_ZH.md`](docs/HOMER_PUBLIC_BRIDGE_ROUTE_ZH.md).

The source-aware public-population inventory is the immutable
[`manifests/homer_population_public_release_362.json`](manifests/homer_population_public_release_362.json)
lineage manifest.  It is anchored to the public HIA dataset revision
`1cd70477b6a99a473690a25a2fed359f75184c64` and contains exactly 362 contests
(271/44/47 description splits) with 1,086 retained source caption rows.  The
HIA paper's 365-contest claim is kept separate from both this machine-readable
public release and the 385 records in the released HOMER evaluator file.  The
362-ID list is explicitly recorded in
[`manifests/homer_public_release_362_allowlist.json`](manifests/homer_public_release_362_allowlist.json);
the data directory is ignored and recreated by
`scripts/create_homer_public_release_362.py`.

The earlier v2 copy remains a parent lineage artifact for the 878 image-byte
repair.  It is not the current population manifest.  Canonical 365 remains an
unresolved, fail-closed claim until an authoritative 365-ID allow-list is
published.

The current public-code route uses the separately implemented
`OfficialCodeRetrieval` adapter (official regex-priority search, query-fitted
TF-IDF, WordNet entity scoring and edge-tree construction). The older
`HomerRetrievalAugmenter` remains historical and must not be used to label a
run as public-code-compatible.

The HOMER paper writes the caption control as Ω ∈ NS × LA, but neither the
paper nor the pinned public caption prompt publishes a finite, machine-verifiable
NS/LA vocabulary. We therefore expose a separate, versioned project extension,
`project_omega_grid_v1` (6 narrative strategies × 6 linguistic styles = 36
combinations), in [`src/humor_generator_v35/homer/omega.py`](src/humor_generator_v35/homer/omega.py).
The exact public-code baseline omits Ω; `--omega NS|LA` must be supplied to
enable the extension, and its vocabulary, prompt, seed and hash are recorded in
provenance. This prevents a useful project control from being misreported as
an official HOMER enumeration.

## Non-negotiable gates

- Both 7B policies remain frozen; only a bridge may be trainable.
- Every Planner trace contains the full public-code stage ledger, strict
  conflict/local/global schemas, exact causal hidden-state/token alignment,
  actual generated semantics, and a SHA-256 digest. Formal traces additionally
  pin the clean Git commit, dataset, prompt-source, and pretrained-model
  manifest hashes; no adapter hash is part of the current route.
- Training uses description-nearest, different-conflict hard negatives.
- The pretrained receiver always receives the image. Text teacher and latent
  student use the same caption targets and the same three plan channels.
- No preference learning starts until latent communication has a stable,
  absolute-quality-preserving held-out gain.

## Data partitions

The current historical bridge manifest contains 810 contest clusters, but they
are not guaranteed to be unique images: source-aware collision auditing finds
multiple HIA/ES images sharing a contest-number key (949 distinct image hashes
across the split rows). Its representative-row policy and adapter-seen labels
were created for the superseded SFT receiver route, so it is not yet the data
manifest for the pretrained-only protocol:

The v2 rebuild updates only the four affected 878 image-hash fields (three
caption rows and one trace row) from the historical hash to the current disk
bytes and records the old/new hash lineage. This is a historical parent data
version, not an in-place manifest edit. The current executable route is the
independently sealed `public_release_362` population above; the paper-level
canonical 365 claim remains explicitly unresolved and does not block this
adapted route.

| split | clusters | role |
|---|---:|---|
| train | 602 | bridge fitting only |
| validation | 64 | checkpoint selection only |
| internal_test | 97 | sealed primary test |
| official_hia_unseen_test | 24 | historical SFT-unseen subset; not the current pretrained split |
| official_hia_seen_diagnostic | 23 | historical SFT-seen diagnostic; not used for current claims |

For the current pretrained-only route, the public-release population gate is
now ready for an adapted 362-contest baseline.  Current pretrained Planner
hidden-state traces are still a separate gate; no bridge training may use
`trace_inputs.jsonl` as if it were a trace index.  Old generation outputs are
historical artifacts and cannot be reused.

## Reproduce the engineering gates

```bash
cd /home/pj26000152/ku60000936/projects/Humor-generator/v3.5
./scripts/bootstrap_environment.sh
.venv/bin/pytest -q
.venv/bin/python scripts/check_environment.py
.venv/bin/python scripts/check_v35_isolation.py
.venv/bin/python scripts/verify_frozen_artifacts.py
.venv/bin/python scripts/verify_clustered_dataset.py
.venv/bin/python scripts/build_homer_population_manifest.py
.venv/bin/python scripts/create_homer_population_v2.py
.venv/bin/python scripts/verify_homer_population_v2.py
.venv/bin/python scripts/create_homer_public_release_362.py
.venv/bin/python scripts/verify_homer_public_release_362.py
.venv/bin/python scripts/verify_homer_evaluation_assets.py
```

The original real-trace GPU engineering smoke passed. Phase A3 and the
channel-isolated A4 semantic-recovery pilot have completed with bridge-only
updates and both 7B policies frozen. A4 is `pilot_inconclusive`, not a latent
success or failure: it has no caption-quality result. The current executable
route is now the pretrained-only HOMER audit/data gate. The exact public
362-contest population is verified, so an adapted online baseline may run;
current pretrained hidden-state traces and bridge/semantic gates remain
required before any bridge training is submitted. Caption pilots remain locked
until those gates pass.

The exact population and its distinction from the unresolved paper-level 365
claim are documented in
[`docs/HOMER_PUBLIC_RELEASE_362_ZH.md`](docs/HOMER_PUBLIC_RELEASE_362_ZH.md).

The first Phase A3
replacement smoke (`6689653`) stopped before forward/backward because a valid
492-token HOMER target exceeded the old 384-token bound; after raising the bound
to 768, replacement smoke `6706516` passed on a full H100. This historical smoke
record closes only an engineering gate, not a scientific semantic result.

The 7B choice is a controlled capability hypothesis, not an assumption that
pretraining is sufficient. The HOMER paper reports non-zero but substantially
lower Qwen-VL (7B) pass@K than its stronger bases, while the NeurIPS 2024 HIA
study reports similarly modest 7B baselines. Therefore the current route freezes
the local pretrained Qwen2.5-VL-7B and trains only the bridge; it does not claim
that this model alone matches GPT-4o/HOMER. Exact claims require the same
stage/prompt/evaluator track and must not use the historical SFT adapters.

## Evaluation

Evaluation is split into two non-interchangeable tracks. The HOMER-comparable
primary track uses five candidates, unbiased Pass@1/3/5 and five repeated
trials against the official human-caption groups. The project extension uses
Group-of-10, mirrored A/B packets, multiple independent judges, absolute
`good/weak/bad` labels, seed variance and image-clustered bootstrap intervals;
it is not a claim of exact HOMER reproduction. Group-of-3 is retained only as
a legacy sensitivity analysis. Formal packets randomize caption order and keep
primary and mechanistic comparisons in separate correction families. Diversity
reports include the official all-mpnet-base-v2 SBERT diversity and EAD metrics,
plus descriptive Distinct-1/2, self-BLEU, TF-IDF distance, Vendi score, and
human angle coverage; diversity is also recomputed on good-only candidates.

For the HOMER-comparable track, the paper label is GPT-5 and the pinned official
evaluator script uses the concrete API model ID `gpt-5-chat-latest` at
temperature 0. The original HIA benchmark is a separate protocol: Group Overall
uses GPT-4-Turbo with Hessel descriptions and Group Best Pick uses GPT-4o-vision
with raw images. These evaluator identities, prompts and asset hashes are kept
separate in `manifests/homer_official_assets.json`; they must not be reported as
one interchangeable score. A different evaluator model may be used only as a
pre-registered substitution with the same candidates, Pass@K, repetitions,
reference groups, temperature and prompt; the actual model ID/snapshot and
`evaluator_substitution=true` must be recorded, and the result is then an
adapted-protocol result rather than canonical GPT-5.

## Authoritative references

- Shang et al., HOMER, ICLR 2026: https://openreview.net/pdf?id=SzaRhPom4o
- HOMER official implementation: https://github.com/Shang-hub/HOMER-Official-Implementation
- Du et al., InterLat, ACL 2026: https://aclanthology.org/2026.acl-long.1248/
- Peng et al., StateBridge, COLM 2026: https://arxiv.org/abs/2608.13317
- Zhang et al., Humor in AI, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf
- Hessel et al., Electronic Sheep, ACL 2023: https://aclanthology.org/2023.acl-long.41/
- Tevet & Berant, diversity evaluation, EACL 2021: https://aclanthology.org/2021.eacl-main.25/
- Friedman & Dieng, The Vendi Score, TMLR 2023: https://arxiv.org/abs/2210.02410
