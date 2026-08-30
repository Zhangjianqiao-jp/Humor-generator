# Humor Generator v3.0

This directory is an isolated continuation of the archived legacy system with
one hard ordering constraint:

1. reproduce every publicly specified component of HOMER;
2. record undisclosed inputs as blockers rather than silently inventing them;
3. only then compare text and latent communication.

The legacy tree is frozen at annotated tag
`v2.5-legacy-freeze-20260830` (commit `3927f18`).  Executable v3 code is
statically checked so that it cannot import or execute legacy scripts.  The two
7B adapters are copied into git-ignored `artifacts/checkpoints/` and verified
against `manifests/frozen_7b_adapters.json` before every GPU job.

Create and audit the independent Python 3.12 environment with:

```bash
cd /home/pj26000152/ku60000936/projects/Humor-generator/v3.0
./scripts/bootstrap_environment.sh
.venv/bin/python scripts/check_environment.py
.venv/bin/python scripts/check_v3_isolation.py
```

The official standard descriptions and joke CSV are pinned in
`manifests/homer_official_assets.json`. Fetch and byte-verify them locally with:

```bash
.venv/bin/python scripts/fetch_homer_official_assets.py
```

The experiment uses the locally cached
`Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5`
as an explicit model substitution. The gate verifies this snapshot and the
official data assets. A passing gate means method/data reproducibility with a
pinned substitute; it does not claim exact parity with HOMER's undisclosed
Qwen-VL weights:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_reproduction_gate.py
```

Primary references:

- Shang et al., *On the Wings of Imagination: Conflicting Script-based
  Multi-role Framework for Humor Caption Generation*, ICLR 2026,
  arXiv:2602.06423.
- Du et al., *Enabling Agents to Communicate Entirely in Latent Space*, ACL
  2026.
- Peng et al., *StateBridge*, COLM 2026.
- Hao et al., *Coconut*, COLM 2025.

See `docs/HOMER_REPRODUCTION_LEDGER_ZH.md` for exact/unknown/extended fields.

## Formal bridge ordering

1. Rebuild and verify the image-clustered 648/81/81 split.
2. Generate strict Conflict/Local/Global Planner traces and freeze each tensor
   with SHA-256.
3. Train only Learned and Typed bridges; both 7B policies have zero trainable
   parameters.
4. Compare Text-HOMER, training-free StateBridge, Learned and Typed using the
   same held-out clusters and anonymous Group-of-3 generations.
5. Preference learning remains disabled until a latent method shows a stable
   held-out gain over Text-HOMER.
