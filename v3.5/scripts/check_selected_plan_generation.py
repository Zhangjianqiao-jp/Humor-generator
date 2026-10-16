#!/usr/bin/env python3
"""Real quantized Qwen cache/hook diagnostics; initial bridge, train rows only."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from humor_generator_v35.qwen_backend import QwenBackend, model_device, find_decoder_layers
from humor_generator_v35.latent.cross_attention import ReceiverDrivenCrossAttentionBridge
from humor_generator_v35.training.public_bridge import load_context_index, public_latent_caption_messages, public_text_caption_messages, validate_row_context
from humor_generator_v35.training.selected_plan import encode_selected_plan, selected_plan_sha256
from humor_generator_v35.homer.caption_parsing import parse_caption_strict


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def cache_probe(backend, bridge, memory, messages):
    model = backend.model
    encoded = backend.encode(messages, add_generation_prompt=True)
    assert not any(k.startswith("pixel") for k in encoded)
    prompt = encoded["input_ids"].shape[1]
    layers = find_decoder_layers(model)
    initial_hooks = [len(layer._forward_hooks) for layer in layers]
    observed = []
    def observe(_module, _inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        observed.append(int(hidden.shape[1]))
    handle = layers[bridge.layer_indices[0]].register_forward_hook(observe)
    try:
        with (bridge.inject(model, memory) if memory is not None else nullcontext()):
            result = model.generate(**encoded, do_sample=False, max_new_tokens=8,
                                    use_cache=True, repetition_penalty=1.0,
                                    return_dict_in_generate=True, output_logits=True)
    finally:
        handle.remove()
    assert [len(layer._forward_hooks) for layer in layers] == initial_hooks
    injection_updates = ([d.relative_update_norm for d in bridge.last_diagnostics]
                         if memory is not None else [])
    if memory is not None and (not injection_updates or not all(0 < x < float("inf") for x in injection_updates)):
        raise RuntimeError("bridge injection absent or nonfinite")
    steps = []
    # Replay the SAME token prefixes. Comparing separate free-running sequences
    # would confound floating-point drift with already-divergent histories.
    for i, cached in enumerate(result.logits):
        ids = result.sequences[:, :prompt+i]
        with (bridge.inject(model, memory) if memory is not None else nullcontext()):
            full = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                         use_cache=False, logits_to_keep=1).logits[:, -1, :]
        a, b = cached.float(), full.float()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise RuntimeError("nonfinite cache/replay logits")
        la, lb = a.log_softmax(-1), b.log_softmax(-1)
        steps.append({"step": i, "max_abs_logit_error": float((a-b).abs().max()),
                      "kl_cached_to_full": float((la.exp()*(la-lb)).sum(-1).mean()),
                      "same_argmax": bool((a.argmax(-1)==b.argmax(-1)).all())})
    assert [len(layer._forward_hooks) for layer in layers] == initial_hooks
    expected = [prompt] + [1]*(len(steps)-1)
    # Conservative engineering gate, not exact arithmetic equivalence. Both
    # baseline and injected runs use the same predeclared tolerance.
    passed = (len(steps) >= 2 and observed == expected and
              all(s["same_argmax"] and abs(s["kl_cached_to_full"]) <= 1e-3
                  and s["max_abs_logit_error"] <= 0.25 for s in steps))
    return {"status": "pass" if passed else "fail", "steps": steps,
            "observed_decoder_sequence_lengths": observed,
            "hooks_removed": True, "injection_relative_updates": injection_updates,
            "max_abs_logit_tolerance": 0.25,
            "kl_tolerance": 1e-3, "requires_identical_argmax": True}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config_path=ROOT/"configs/pilot/cross_attention_caption_selected_plan_v2.yaml"
    cfg=yaml.safe_load(config_path.read_text())
    source_paths=[Path(__file__), ROOT/"jobs/v35_generation_cache_smoke.pjm",
                  *sorted((ROOT/"src").rglob("*.py")), config_path]
    hashes={str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    (args.output/"source_manifest.json").write_text(json.dumps(hashes, indent=2)+"\n")
    context_path=ROOT/cfg["data"]["context_index"]
    contexts=load_context_index(context_path)
    rows=[json.loads(line) for line in (ROOT/cfg["data"]["dataset"]/"train.jsonl").read_text().splitlines() if line.strip()]
    selected=[]
    for cluster in ("nycc_555", "nycc_788"):
        row=next(r for r in rows if r["cluster_id"]==cluster)
        validate_row_context(row, contexts[cluster])
        selected.append(row)
    mc=cfg["model"]
    backend=QwenBackend.load(mc["name"], revision=mc["revision"], load_in_4bit=True,
                            min_visual_tokens=mc["min_visual_tokens"], max_visual_tokens=mc["max_visual_tokens"])
    torch.manual_seed(cfg["training"]["seed"])
    width=backend.model.get_input_embeddings().weight.shape[1]
    bc=cfg["bridge"]
    bridge=ReceiverDrivenCrossAttentionBridge(width,width,
        layer_indices=bc["layer_indices"], bottleneck_dim=bc["bottleneck_dim"],
        heads=bc["heads"], gate_init=bc["gate_init"], channel_fusion=bc["channel_fusion"],
        projection_mode=bc["projection_mode"]).to(model_device(backend.model)).eval()
    report={"scope":"initial_bridge_engineering_only_train_rows", "humor_quality_verified":False,
            "model_revision":mc["revision"], "config_sha256":digest(config_path),
            "context_sha256":digest(context_path), "cache_checks":[], "generations":[],
            "production_generate_use_cache":backend.model.generation_config.use_cache,
            "bridge_checkpoint":None, "max_new_tokens":192,
            "refusal_detection":"strict_parser_known_patterns_only"}
    for row in selected:
        cluster=row["cluster_id"]; context=contexts[cluster]
        states=encode_selected_plan(backend,context)
        parameter=next(bridge.parameters())
        memory={k:v.states.to(device=parameter.device,dtype=parameter.dtype) for k,v in states.items()}
        messages=public_latent_caption_messages(context)
        for condition, mem in (("no_memory",None),("initial_latent_bridge",memory)):
            check=cache_probe(backend,bridge,mem,messages)
            report["cache_checks"].append({"cluster":cluster,"condition":condition,**check})
        for condition in ("text_same_plan","no_memory","initial_latent_bridge"):
            msgs=public_text_caption_messages(context) if condition=="text_same_plan" else messages
            for seed in (20261017,20261018,20261019):
                with (bridge.inject(backend.model,memory) if condition=="initial_latent_bridge" else nullcontext()):
                    raw=backend.generate(msgs,temperature=1.0,max_new_tokens=192,seed=seed)
                parsed=parse_caption_strict(raw)
                item={"cluster":cluster,"condition":condition,"seed":seed,
                      "selected_plan_sha256":selected_plan_sha256(context),"raw_output":raw,
                      "caption":parsed.caption,"parse_status":parsed.status}
                report["generations"].append(item)
                with (args.output/"generations.jsonl").open("a") as f:
                    f.write(json.dumps(item,ensure_ascii=False)+"\n")
    report["generation_summary"]=[]
    for row in selected:
        for condition in ("text_same_plan","no_memory","initial_latent_bridge"):
            items=[g for g in report["generations"] if g["cluster"]==row["cluster_id"] and g["condition"]==condition]
            valid=[" ".join(g["caption"].casefold().split()) for g in items if g["parse_status"]=="ok"]
            report["generation_summary"].append({"cluster":row["cluster_id"],"condition":condition,
                "count":len(items),"parse_status_counts":dict(Counter(g["parse_status"] for g in items)),
                "duplicate_valid_captions":len(valid)-len(set(valid)),
                "duplicate_raw_outputs":len(items)-len(set(g["raw_output"] for g in items))})
    assert all(digest(ROOT/name)==h for name,h in hashes.items()), "source changed during run"
    report["status"]="pass" if all(c["status"]=="pass" for c in report["cache_checks"]) else "fail"
    # This status covers cache only; format and repeats are descriptive results.
    report["status_scope"]="cache_checks_only_format_is_descriptive"
    (args.output/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="generations"},indent=2))
    if report["status"]!="pass":
        raise SystemExit(1)

if __name__=="__main__":
    main()
