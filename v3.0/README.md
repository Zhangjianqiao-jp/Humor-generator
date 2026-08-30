# Humor Generator v3.0

This directory is a clean-room continuation of `v2.5` with one hard ordering
constraint:

1. reproduce every publicly specified component of HOMER;
2. record undisclosed inputs as blockers rather than silently inventing them;
3. only then compare text and latent communication.

Formal training is disabled. The only permitted execution is the engineering
smoke:

```bash
cd /home/pj26000152/ku60000936/projects/Humor-generator/v3.0
PYTHONPATH=src ../v2.5/.venv-genkai/bin/python scripts/engineering_smoke.py
```

The official standard descriptions and joke CSV are pinned in
`manifests/homer_official_assets.json`. Fetch and byte-verify them locally with:

```bash
../v2.5/.venv-genkai/bin/python scripts/fetch_homer_official_assets.py
```

The experiment uses the locally cached
`Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5`
as an explicit model substitution. The gate verifies this snapshot and the
official data assets. A passing gate means method/data reproducibility with a
pinned substitute; it does not claim exact parity with HOMER's undisclosed
Qwen-VL weights:

```bash
PYTHONPATH=src ../v2.5/.venv-genkai/bin/python scripts/check_reproduction_gate.py
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
