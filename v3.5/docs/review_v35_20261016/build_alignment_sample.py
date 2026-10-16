"""Rebuild a fixed, training-only qualitative audit sample; never writes labels."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED = 'v35_alignment_audit_20261016:'
rows_path = ROOT / 'data/processed/homer_pretrained_7b_bridge_362/train.jsonl'
context_path = ROOT / 'data/cache/homer_pretrained_7b_homer_context/index.jsonl'
rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
contexts = {r['cluster_id']: r for r in map(json.loads, context_path.read_text().splitlines())}
clusters = sorted({r['cluster_id'] for r in rows},
                  key=lambda c: hashlib.sha256((SEED+c).encode()).hexdigest())[:8]
records = []
for cluster in clusters:
    c = contexts[cluster]
    records.append(dict(cluster_id=cluster, description=c['standard_description'],
                        selected_conflict=c['selected_conflict'], selected_paths=c['selected_paths'],
                        free_association_text=c['free_association_text'],
                        captions=[dict(row_id=r['row_id'], caption=r['caption'])
                                  for r in rows if r['cluster_id'] == cluster]))
payload = dict(scope='fixed_training_only_sample_not_gold_annotations', selection_seed=SEED,
               selection='first_8_clusters_sorted_by_sha256_seed_plus_cluster_id',
               train_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
               context_sha256=hashlib.sha256(context_path.read_bytes()).hexdigest(), records=records)
(OUT/'alignment_sample.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(dict(clusters=len(records), captions=sum(len(r['captions']) for r in records))))
