#!/usr/bin/env python3
"""Exact layer-by-module preference gradient, Fisher, utility, and alignment.

The script differentiates the chosen-minus-rejected sequence-log-probability
margin with respect to the *frozen SFT policy's base weight matrices*.  It does
not use SFT gradients as a proxy for preference gradients and it does not rank
unmeasured modules as zero.  Modules are processed in small chunks so a full
copy of every base-model gradient never has to reside on the GPU at once.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qwen_vl_lora_loader import load_qwen_vl_with_lora
from src.preference.diagnostics import select_image_diverse_rows, sha256, write_csv
from src.preference.module_analysis import (
    TRANSFORMER_MODULES,
    aggregate_module_groups,
    cumulative_selection,
    rank_statistics,
)
from src.training.dpo_dataset import DPOCollator, PreferenceDataset, model_inputs_from_batch, sequence_logps

LAYER_PATTERN = re.compile(r"(?:language_model|model)\.layers\.(\d+)")


def model_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def base_weight(module: torch.nn.Module) -> torch.nn.Parameter | None:
    base = getattr(module, "base_layer", module)
    weight = getattr(base, "weight", None)
    return weight if isinstance(weight, torch.nn.Parameter) else None


def module_kind(path: str) -> tuple[int | None, str, str]:
    layer_match = LAYER_PATTERN.search(path)
    module = next((name for name in TRANSFORMER_MODULES if path.endswith(name)), path.rsplit(".", 1)[-1])
    if ".visual." in path or path.startswith("visual."):
        group = "vision_encoder"
    elif "merger" in path or "projector" in path or "connector" in path:
        group = "multimodal_projector"
    elif path.endswith("lm_head"):
        group = "lm_head"
    else:
        group = "language_backbone"
    return (int(layer_match.group(1)) if layer_match else None), module, group


def discover_modules(model: Any, requested: set[str], include_auxiliary: bool) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen_weights: set[int] = set()
    for path, module in model.named_modules():
        layer, kind, group = module_kind(path)
        wanted = kind in requested and layer is not None and group == "language_backbone"
        auxiliary = include_auxiliary and group in {"vision_encoder", "multimodal_projector", "lm_head"}
        if not wanted and not auxiliary:
            continue
        weight = base_weight(module)
        if weight is None or weight.ndim != 2 or id(weight) in seen_weights:
            continue
        if not weight.is_floating_point():
            raise TypeError(
                f"Exact base-weight diagnosis requires floating weights; {path} has {weight.dtype}. "
                "Run this diagnostic without 4-bit loading."
            )
        seen_weights.add(id(weight))
        found.append(
            {
                "module_path": path,
                "layer": layer,
                "module": kind,
                "module_group": group,
                "weight": weight,
            }
        )
    found.sort(key=lambda row: (row["module_group"], -1 if row["layer"] is None else row["layer"], row["module_path"]))
    return found


def save_heatmap(path: Path, rows: list[dict[str, Any]], metric: str, title: str) -> None:
    measured = [row for row in rows if row["layer"] not in (None, "") and row["module"] in TRANSFORMER_MODULES]
    layers = sorted({int(row["layer"]) for row in measured})
    modules = list(TRANSFORMER_MODULES)
    width, height = 190 + 104 * len(modules), 125 + 30 * max(1, len(layers))
    image = Image.new("RGB", (width, height), "white")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default()
    draw.text((18, 16), title, fill="#10253f", font=font)
    lookup = {(int(row["layer"]), row["module"]): float(row[metric]) for row in measured}
    finite = [value for value in lookup.values() if math.isfinite(value)]
    low, high = (min(finite), max(finite)) if finite else (0.0, 1.0)
    for col, module in enumerate(modules):
        draw.text((105 + col * 104, 48), module.replace("_proj", ""), fill="#10253f", font=font)
    for ridx, layer in enumerate(layers):
        y = 70 + ridx * 30
        draw.text((25, y + 7), str(layer), fill="#10253f", font=font)
        for col, module in enumerate(modules):
            x = 95 + col * 104
            value = lookup.get((layer, module))
            if value is None or not math.isfinite(value):
                color, label = "#e5e7eb", "NA"
            else:
                ratio = (value - low) / max(high - low, 1e-30)
                color = (int(240 - 170 * ratio), int(247 - 90 * ratio), int(253 - 25 * ratio))
                label = f"{value:.1e}"
            draw.rectangle((x, y, x + 96, y + 24), fill=color, outline="#cbd5e1")
            draw.text((x + 3, y + 7), label, fill="#10253f", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_selection(path: Path, selected: list[dict[str, Any]], threshold: float, source: Path) -> None:
    value = {
        "selection": {
            "metric": "adaptation_utility",
            "cumulative_threshold": threshold,
            "source": str(source),
            "module_count": len(selected),
            "base_parameter_count": sum(int(row["parameter_count"]) for row in selected),
        },
        "model": {
            "lora": {
                "target_modules": [str(row["module_path"]) for row in selected],
                "rank": 1,
                "alpha": 2,
                "dropout": 0.05,
                "note": "Rank is a placeholder; use match_lora_budget.py before training.",
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def report(path: Path, summary: dict[str, Any], ranked: list[dict[str, Any]], groups: list[dict[str, Any]]) -> None:
    top = ranked[:20]
    lines = [
        "# Generator Preference Module Analysis",
        "",
        "This report measures exact gradients on the frozen SFT policy's floating-point base matrices. It is a diagnostic, not proof that the highest-ranked placement trains best; top, bottom, random, grouped, and utility-selected interventions are required.",
        "",
        f"- Pairs: {summary['pairs']} image-diverse examples",
        f"- Fisher damping: `{summary['fisher_lambda']}`",
        f"- Measured matrices: {summary['measured_modules']}",
        f"- Missing requested families: {', '.join(summary['missing_module_families']) or 'none'}",
        "",
        "## Module-family summary",
        "",
        "| Module | Preference gradient | Mean Fisher | Utility | Utility / parameter | Parameters |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(groups, key=lambda value: -float(value["adaptation_utility"])):
        lines.append(
            f"| {row['module']} | {row['raw_grad_norm']:.4e} | {row['mean_fisher']:.4e} | "
            f"{row['adaptation_utility']:.4e} | {row['utility_per_parameter']:.4e} | {row['parameter_count']} |"
        )
    lines.extend(
        [
            "",
            "## Top layer × module matrices",
            "",
            "| Rank | Layer | Module | PrefGrad | Fisher | Utility | Utility/param | Params | Cos(SFT,Pref) |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top:
        lines.append(
            f"| {row['rank']} | {row['layer']} | {row['module']} | {row['raw_grad_norm']:.4e} | "
            f"{row['mean_fisher']:.4e} | {row['adaptation_utility']:.4e} | "
            f"{row['utility_per_parameter']:.4e} | {row['parameter_count']} | {row['sft_pref_cosine']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guard",
            "",
            "Preference gradient says where the pairwise objective requests change. Base Fisher estimates sensitivity of the current SFT behavior. Utility rewards preference-sensitive and base-insensitive coordinates after damping. None of these quantities establishes causal training value until equal-budget placement interventions reproduce the ranking.",
            "",
            "## References",
            "",
            "- Rafailov et al. (2023), Direct Preference Optimization, NeurIPS 2023.",
            "- Kirkpatrick et al. (2017), Overcoming catastrophic forgetting in neural networks, PNAS (diagonal empirical Fisher/EWC).",
            "- Hu et al. (2022), LoRA: Low-Rank Adaptation of Large Language Models, ICLR 2022.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/preference_analysis"))
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--module-families", nargs="+", default=list(TRANSFORMER_MODULES))
    parser.add_argument("--layers-per-chunk", type=int, default=1)
    parser.add_argument("--max-layers", type=int, default=None, help="Limit language layers for a GPU smoke test.")
    parser.add_argument("--fisher-lambda", type=float, default=1e-6)
    parser.add_argument("--include-auxiliary", action="store_true")
    args = parser.parse_args()
    if args.layers_per_chunk < 1:
        raise ValueError("--layers-per-chunk must be positive")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg, data_cfg = config["model"], config["data"]
    lora = model_cfg["lora"]
    model, processor = load_qwen_vl_with_lora(
        model_name=model_cfg["model_name"],
        lora_rank=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias=str(lora.get("bias", "none")),
        device_map=str(model_cfg.get("device_map", "auto")),
        torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16")),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        adapter_path=args.adapter,
        is_trainable=False,
        image_min_pixels=data_cfg.get("image_min_pixels"),
        image_max_pixels=data_cfg.get("image_max_pixels"),
        load_in_4bit=False,
    )
    model.eval()
    dataset = PreferenceDataset(args.pairs)
    dataset.rows = select_image_diverse_rows(dataset.rows, args.max_pairs)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=DPOCollator(
            processor,
            max_seq_len=int(data_cfg.get("max_seq_len", 768)),
            require_reference=False,
            image_min_pixels=data_cfg.get("image_min_pixels"),
            image_max_pixels=data_cfg.get("image_max_pixels"),
        ),
        num_workers=0,
    )
    requested = set(args.module_families)
    unknown = requested - set(TRANSFORMER_MODULES)
    if unknown:
        raise ValueError(f"Unknown module families: {sorted(unknown)}")
    discovered = discover_modules(model, requested, args.include_auxiliary)
    if not discovered:
        raise RuntimeError("No requested floating-point matrices were discovered")
    layers = sorted({row["layer"] for row in discovered if row["layer"] is not None})
    if args.max_layers is not None:
        if args.max_layers < 1:
            raise ValueError("--max-layers must be positive")
        layers = layers[: args.max_layers]
        allowed_layers = set(layers)
        discovered = [row for row in discovered if row["layer"] in allowed_layers or row["layer"] is None]
    chunks: list[list[dict[str, Any]]] = []
    for start in range(0, len(layers), args.layers_per_chunk):
        layer_set = set(layers[start : start + args.layers_per_chunk])
        chunks.append([row for row in discovered if row["layer"] in layer_set])
    chunks.extend([[row] for row in discovered if row["layer"] is None])

    device = model_device(model)
    result_rows: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for chunk_index, chunk in enumerate(chunks, start=1):
        params = [row["weight"] for row in chunk]
        for parameter in params:
            parameter.requires_grad_(True)
        pref_sq = [torch.zeros_like(parameter, dtype=torch.float32, device="cpu") for parameter in params]
        fisher_sq = [torch.zeros_like(parameter, dtype=torch.float32, device="cpu") for parameter in params]
        norm_sum = [0.0] * len(params)
        dot_sum = [0.0] * len(params)
        pref_norm_sq_sum = [0.0] * len(params)
        sft_norm_sq_sum = [0.0] * len(params)
        cosine_sum = [0.0] * len(params)
        examples = 0
        for batch_index, batch in enumerate(loader, start=1):
            output = model(**model_inputs_from_batch(batch, device))
            logps, token_counts = sequence_logps(output.logits, batch["labels"].to(device))
            if int(batch["num_pairs"]) != 1:
                raise AssertionError("Exact per-example statistics require batch_size=1")
            margin = logps[0] - logps[1]
            pref_grads = torch.autograd.grad(margin, params, retain_graph=True, allow_unused=True)
            chosen_nll = -logps[0] / token_counts[0].clamp_min(1)
            sft_grads = torch.autograd.grad(chosen_nll, params, allow_unused=True)
            for index, (pref_grad, sft_grad) in enumerate(zip(pref_grads, sft_grads)):
                if pref_grad is None or sft_grad is None:
                    continue
                pref = pref_grad.detach().float()
                sft = sft_grad.detach().float()
                pref_cpu, sft_cpu = pref.cpu(), sft.cpu()
                pref_sq[index].add_(pref_cpu.square())
                fisher_sq[index].add_(sft_cpu.square())
                pref_norm_sq = float(pref.square().sum().cpu())
                sft_norm_sq = float(sft.square().sum().cpu())
                dot = float((pref * sft).sum().cpu())
                norm_sum[index] += math.sqrt(pref_norm_sq)
                dot_sum[index] += dot
                pref_norm_sq_sum[index] += pref_norm_sq
                sft_norm_sq_sum[index] += sft_norm_sq
                denominator = math.sqrt(pref_norm_sq * sft_norm_sq)
                cosine_sum[index] += dot / denominator if denominator else 0.0
            examples += 1
            del output, logps, pref_grads, sft_grads
            print(f"[module-analysis] chunk={chunk_index}/{len(chunks)} pair={batch_index}/{len(loader)}")
        for index, item in enumerate(chunk):
            parameter = params[index]
            count = parameter.numel()
            pref_mean_sq = pref_sq[index] / max(examples, 1)
            fisher_mean = fisher_sq[index] / max(examples, 1)
            utility = float((pref_mean_sq / (fisher_mean + args.fisher_lambda)).sum().item())
            raw = norm_sum[index] / max(examples, 1)
            weight_norm = float(parameter.detach().float().norm().cpu())
            denominator = math.sqrt(pref_norm_sq_sum[index] * sft_norm_sq_sum[index])
            result_rows.append(
                {
                    "layer": "" if item["layer"] is None else item["layer"],
                    "module": item["module"],
                    "module_group": item["module_group"],
                    "module_path": item["module_path"],
                    "raw_grad_norm": raw,
                    "normalized_grad_norm": raw / math.sqrt(count),
                    "relative_grad_norm": raw / (weight_norm + 1e-12),
                    "gradient_energy": float(pref_mean_sq.mean().item()),
                    "mean_fisher": float(fisher_mean.mean().item()),
                    "adaptation_utility": utility,
                    "utility_per_parameter": utility / count,
                    "parameter_count": count,
                    "weight_norm": weight_norm,
                    "sft_pref_cosine": dot_sum[index] / denominator if denominator else float("nan"),
                    "mean_pair_cosine": cosine_sum[index] / max(examples, 1),
                    "pairs": examples,
                }
            )
            parameter.requires_grad_(False)
        write_csv(args.output_dir / "module_gradient_statistics.partial.csv", result_rows)
        del pref_sq, fisher_sq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ranked = rank_statistics(result_rows)
    groups = aggregate_module_groups(result_rows)
    write_csv(args.output_dir / "module_gradient_statistics.csv", result_rows)
    write_csv(args.output_dir / "module_group_summary.csv", groups)
    write_csv(args.output_dir / "module_ranking.csv", ranked)
    write_csv(
        args.output_dir / "module_fisher.csv",
        [{key: row[key] for key in ("layer", "module", "module_path", "mean_fisher", "parameter_count", "pairs")} for row in result_rows],
    )
    alignment_rows = [
        {key: row[key] for key in ("layer", "module", "module_path", "sft_pref_cosine", "mean_pair_cosine", "pairs")}
        for row in result_rows
    ]
    write_csv(args.output_dir / "sft_pref_gradient_cosine.csv", alignment_rows)
    save_heatmap(args.output_dir / "layer_module_gradient_heatmap.png", result_rows, "normalized_grad_norm", "Preference gradient / sqrt(parameters)")
    save_heatmap(args.output_dir / "layer_module_utility_heatmap.png", result_rows, "utility_per_parameter", "Preference adaptation utility / parameter")
    save_heatmap(args.output_dir / "sft_pref_gradient_alignment.png", result_rows, "sft_pref_cosine", "SFT vs preference gradient cosine")

    selection80 = cumulative_selection(ranked, 0.8)
    selection90 = cumulative_selection(ranked, 0.9)
    write_selection(ROOT / "configs/preference/selective_modules_80.yaml", selection80, 0.8, args.output_dir / "module_ranking.csv")
    write_selection(ROOT / "configs/preference/selective_modules_90.yaml", selection90, 0.9, args.output_dir / "module_ranking.csv")
    observed = {row["module"] for row in result_rows if row["module"] in TRANSFORMER_MODULES}
    summary = {
        "model": model_cfg["model_name"],
        "adapter": str(args.adapter),
        "pairs": len(dataset),
        "pairs_sha256": sha256(args.pairs),
        "config_sha256": sha256(args.config),
        "measured_modules": len(result_rows),
        "requested_module_families": sorted(requested),
        "missing_module_families": sorted(requested - observed),
        "fisher_definition": "diagonal empirical Fisher from per-example chosen-caption token-normalized NLL gradients",
        "preference_gradient_definition": "per-example gradient of log pi(chosen|image,hint) - log pi(rejected|image,hint)",
        "utility_definition": "sum_i E[d_i^2] / (E[g_sft_i^2] + lambda)",
        "fisher_lambda": args.fisher_lambda,
        "selection_80_modules": len(selection80),
        "selection_90_modules": len(selection90),
        "validation_required": "Ranking is a hypothesis; validate top/bottom/random/group/selective placements under matched LoRA budgets.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report(ROOT / "docs/preference_module_analysis.md", summary, ranked, groups)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
