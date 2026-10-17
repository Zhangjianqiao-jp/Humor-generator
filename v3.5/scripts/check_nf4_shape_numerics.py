#!/usr/bin/env python3
"""Isolate shape-dependent linear arithmetic; no model, data, cache or bridge."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import torch
import bitsandbytes as bnb
from bitsandbytes.backends.cuda import ops


def error(a, b):
    delta = a.double() - b.double()
    if not torch.isfinite(delta).all():
        raise RuntimeError('Nonfinite linear output')
    return {'max_abs': float(delta.abs().max()), 'rms': float(delta.square().mean().sqrt())}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    records = []
    for seed in (17, 18, 19):
        torch.manual_seed(seed)
        # Synthetic matrices; representative dimensions, not model activations.
        for width in (512, 3584):
            layer = bnb.nn.Linear4bit(width, width, bias=False,
                compute_dtype=torch.bfloat16, compress_statistics=True, quant_type='nf4')
            layer.requires_grad_(False)
            layer = layer.cuda().eval()
            w = bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state).to(torch.bfloat16)
            for rows in (140, 155):
                x = torch.randn(rows, width, device='cuda', dtype=torch.bfloat16)
                single = x[-1:].clone()
                nf4_batch, nf4_single = layer(x)[-1:], layer(single)
                dense_batch = torch.nn.functional.linear(x, w)[-1:]
                dense_single = torch.nn.functional.linear(single, w)
                fp_batch = torch.nn.functional.linear(x.float(), w.float())[-1:]
                fp_single = torch.nn.functional.linear(single.float(), w.float())
                record = {'seed':seed, 'width':width, 'rows':rows,
                    'nf4_shape_error':error(nf4_batch,nf4_single),
                    'dense_bf16_shape_error':error(dense_batch,dense_single),
                    'dense_fp32_shape_error':error(fp_batch,fp_single),
                    'nf4_batch_vs_dense_bf16':error(nf4_batch,dense_batch),
                    'nf4_single_vs_dense_bf16':error(nf4_single,dense_single),
                    'nf4_same_shape_repeat':error(nf4_single,layer(single))}
                heuristic = getattr(ops, '_gemm_4bit_use_custom_cuda', None)
                if heuristic is not None:
                    record['custom_kernel_dispatch'] = {
                        str(m): bool(heuristic(torch.cuda.current_device(),torch.bfloat16,m,width,width))
                        for m in (1,rows)}
                records.append(record)
            del layer, w
    source = Path(inspect.getfile(ops))
    report = {'status':'completed_diagnostic_not_model_gate',
        'scope':'synthetic_same_weights_same_last_row_no_cache_no_bridge',
        'torch':torch.__version__, 'bitsandbytes':importlib.metadata.version('bitsandbytes'),
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'cuda_ops_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fp32_reference':'same dequantized BF16-rounded weights, TF32 disabled', 'records':records}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
