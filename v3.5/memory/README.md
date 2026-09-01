# v3.5 Project Memory

This directory is the compact, evidence-linked handoff for future agents. It
does not archive chat transcripts or private reasoning. Read files in this
order:

1. `working_state.yaml`: volatile state and the next executable action;
2. `project_memory.yaml`: stable goals, contracts, gates, and experiment plan;
3. `retrieval_index.yaml`: topic-to-file routing;
4. `episodes.jsonl`: append-only evidence and lessons from material events.

## Memory policy

The design separates factual, experiential, procedural, and working memory.
Only evidence-backed information that changes a future decision is admitted.
Every volatile claim carries `observed_at` and `must_revalidate`; a scheduler
status is never treated as durable truth. Superseded claims are marked rather
than silently rewritten. Failed experiments are retained with their lesson,
while raw terminal chatter and duplicate summaries are excluded.

Update rules:

- update `working_state.yaml` after a phase transition or scheduler change;
- append one episode only for a material result, failure, or decision;
- update stable memory only when a contract or preregistered decision changes;
- link local evidence and authoritative references;
- run `.venv/bin/python scripts/validate_project_memory.py` before committing;
- never infer success from job disappearance or training loss alone.

This implements a practical version of selective memory addition/deletion,
linked-note organization, episodic reflection, and explicit memory operations.
It is intentionally deterministic rather than an unvalidated autonomous memory
writer.

## References

- Memory in the Age of AI Agents (2025): https://arxiv.org/abs/2512.13564
- Agentic Memory / AgeMem (2026): https://arxiv.org/abs/2601.01885
- A-MEM (2025): https://arxiv.org/abs/2502.12110
- Memory Management and Experience-Following (2025): https://arxiv.org/abs/2505.16067
- Reflexion, NeurIPS 2023: https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html
