# Continue HOMER typed-latent bridge experiment

Work only in `/home/pj26000152/ku60000936/projects/Humor-generator/v2.5` on branch `latent`.

Read `docs/HOMER_LATENT_REBUILD_ZH.md`, `docs/SFT_WORKING_AGREEMENT.md`, the current git diff, configs, scripts, PJM logs, and job state before acting. Preserve every unrelated dirty/untracked user change. Do not reset, delete, or commit unrelated files. Do not force-push.

Primary immediate task:

1. Inspect full-H100 smoke job `6638177`. Disappearance from `pjstat` is not success. Require scheduler/history exit 0, no traceback/OOM/NaN in `genkai_smoke_homer_typed_latent_bridge.pjm.6638177.out`, and both `outputs/latent_communication/homer_typed_bridge_smoke/best.pt` and `manifest.json`.
2. If smoke fails, diagnose and make the smallest scoped fix, rerun CPU/unit checks, commit/push only latent-branch files, and submit at most one new smoke. Do not submit formal training.
3. If smoke passes, inspect `show_rsc` and all permitted b/c/MIG resource groups. Choose earliest estimated completion with exactly one GPU. Do not use MIG again for bridge backward because three MIG attempts have already produced allocator failures. Avoid duplicate jobs: inspect recorded job IDs, live jobs, and completed artifacts before `pjsub`.
4. Submit and quietly monitor formal `scripts/train_homer_latent_bridge.py` training. Record job ID immediately. Require `best.pt`, `latest.pt`, manifest, finite train/validation NLL, frozen-policy gate, and job exit 0. A lower train loss alone is not scientific success.
5. Do not read or evaluate sealed legacy test47. Do not call 1,044 HOMER rows 1,044 independent images: there are 819 contest clusters and 225 cross-source duplicate rows. Do not bypass the 1,000-independent-held-out gate in `scripts/evaluate_homer_base_vs_sft.py` for a formal claim. A small development generation may be used only as an explicitly non-formal pipeline check.
6. After a valid bridge checkpoint, prepare the Base-Qwen7B vs SFT-7B 10-candidate, 3-seed, blind Group-of-3 evaluation, but stop and document the unresolved 1,000 independent held-out image shortage rather than leaking training images into formal testing.

Scientific basis: HOMER (ICLR 2026, arXiv:2602.06423), Du et al. InterLat (ACL 2026, 2026.acl-long.1248), Peng et al. StateBridge (COLM 2026, arXiv:2608.13317), Hao et al. Coconut (COLM 2025, arXiv:2412.06769), Humor in AI (NeurIPS 2024), Hessel et al. (ACL 2023).

Completion evidence for the automated task is a validated formal bridge checkpoint at `outputs/latent_communication/homer_typed_bridge/best.pt`. If authority or valid held-out data is missing, stop with a concise diagnostic instead of weakening the preregistered gate.
