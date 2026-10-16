#!/usr/bin/env python3
"""Audit every train/validation target using the local production processor."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
import yaml
from transformers import AutoProcessor
from humor_generator_v35.data.traces import read_jsonl
from humor_generator_v35.qwen_backend import QwenBackend
from humor_generator_v35.training.formal_bridge import _prompt_and_full
from humor_generator_v35.training.public_bridge import (
    load_context_index, public_text_caption_messages, public_latent_caption_messages,
    target_caption_text, validate_row_context,
)
from humor_generator_v35.training.selected_plan import PARTIAL_CAPTION_POLICY


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pilot/cross_attention_caption_selected_plan_v2.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config['protocol']['training_target_policy'] != PARTIAL_CAPTION_POLICY:
        raise ValueError('this audit requires the marked open-assistant target policy')
    manifest_path = ROOT / config['model']['model_manifest']
    manifest = json.loads(manifest_path.read_text())
    if (manifest['model_id'], manifest['revision']) != (config['model']['name'], config['model']['revision']):
        raise ValueError('processor manifest and configured model disagree')
    processor = AutoProcessor.from_pretrained(manifest['local_snapshot_path'], local_files_only=True)
    # Device-only stub: no model weights or receiver forwards are needed.
    backend = QwenBackend(torch.nn.Linear(1, 1), processor, lambda messages: (None, None))
    context_path = ROOT / config['data']['context_index']
    contexts = load_context_index(context_path)
    reports, errors = {}, []
    for split in ('train', 'validation'):
        path = ROOT / config['data']['dataset'] / f'{split}.jsonl'
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f'empty {split} data')
        counts = []
        full_lengths = {'text': [], 'latent': []}
        for row in rows:
            try:
                context = contexts[str(row['cluster_id'])]
                validate_row_context(row, context)
                answer = '##Caption: ' + target_caption_text(row)
                targets = []
                for condition, build in (('text', public_text_caption_messages), ('latent', public_latent_caption_messages)):
                    messages = build(context)
                    if any(block['type'] != 'text' for msg in messages
                           if isinstance(msg['content'], list) for block in msg['content']):
                        raise ValueError('unexpected non-text receiver input')
                    prompt, full, target = _prompt_and_full(backend, messages, answer, partial_answer=True)
                    if processor.tokenizer.decode(target[0], skip_special_tokens=False) != answer:
                        raise ValueError('target round-trip mismatch')
                    if target.eq(processor.tokenizer.eos_token_id).any():
                        raise ValueError('target contains EOS')
                    if full['input_ids'].shape[1] != prompt['input_ids'].shape[1] + target.shape[1]:
                        raise ValueError('prompt/target length mismatch')
                    if not torch.equal(full['input_ids'][:, :prompt['input_ids'].shape[1]], prompt['input_ids']):
                        raise ValueError('prompt prefix changed on target append')
                    if target.shape[1] > int(config['training']['max_target_tokens']):
                        raise ValueError('target exceeds configured token limit')
                    targets.append(target)
                    full_lengths[condition].append(int(full['input_ids'].shape[1]))
                if not torch.equal(targets[0], targets[1]):
                    raise ValueError('teacher/student target token mismatch')
                counts.append(int(targets[0].shape[1]))
            except (ValueError, RuntimeError, KeyError) as error:
                errors.append({'split': split, 'row_id': row.get('row_id'), 'error': str(error)})
        reports[split] = dict(rows=len(rows), checked=len(counts), clusters=len({r['cluster_id'] for r in rows}),
                              dataset_sha256=sha(path), max_target_tokens=max(counts, default=None),
                              max_full_tokens={k: max(v, default=None) for k,v in full_lengths.items()})
    result = dict(status='fail' if errors else 'pass', config_sha256=sha(args.config),
                  context_sha256=sha(context_path), model_manifest_sha256=sha(manifest_path),
                  revision=manifest['revision'], splits=reports, errors=errors,
                  scope='train_validation_tokenization_only_no_weights_no_test_set')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
