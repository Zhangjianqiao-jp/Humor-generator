#!/usr/bin/env python3
"""Synthetic counterexample: fitting known references need not recover source semantics."""
import argparse
import json
from pathlib import Path
import torch
from statebridge_alignment import align_message,orthogonal_map


def run():
    torch.set_num_threads(1)
    cases=[]
    for seed in [3,17,41]:
        g=torch.Generator().manual_seed(seed)
        n,d=8,32
        h=torch.randn(n,d,generator=g,dtype=torch.float64)
        e=torch.randn(n,d,generator=g,dtype=torch.float64)
        hc,ec=h-h.mean(0),e-e.mean(0)
        def whiten(x):
            u,s,vt=torch.linalg.svd(x,full_matrices=False)
            keep=s>1e-10
            return n**.5*(u[:,keep]@vt[keep]),int(keep.sum())
        hw,rh=whiten(hc);ew,re=whiten(ec)
        u,s,vt=torch.linalg.svd(ec,full_matrices=False)
        sqrt_cov=(vt.T*(s/n**.5))@vt
        rotation=orthogonal_map(hw.T@ew,True)
        reconstructed=hw@rotation@sqrt_cov+e.mean(0)
        exact=float((reconstructed-e).abs().max())
        assert rh==re==n-1 and exact<1e-10
        _,diag=align_message(h,e,e,regularization=1e-3,anchor_weight=0,proper_rotation=True)
        cases.append({'seed':seed,'tokens':n,'width':d,'source_rank':rh,'reference_rank':re,
            'unregularized_reconstruction_max_error':exact,
            'regularized_reconstruction_rmse':diag['reference_rmse_before_norm_and_anchor']})
    return {'scope':'synthetic_independent_random_states_and_references_not_qwen_or_semantic_evaluation',
        'claim':'reference reconstruction alone does not identify meaningful source-state information',
        'cases':cases}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    result=run()
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
