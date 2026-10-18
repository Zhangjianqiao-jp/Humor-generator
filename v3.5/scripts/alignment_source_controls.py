"""Source interventions; references must remain unchanged outside this function."""
import torch


def source_controls(hidden, seed=20261018):
    if hidden.ndim!=3 or hidden.shape[0]!=1 or hidden.shape[1]<2:
        raise ValueError('expected one message with at least two token positions')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    random=torch.randn(hidden.shape,generator=generator,dtype=torch.float32)
    random=random.to(hidden.device)
    # Only one overall RMS scalar is matched, not per-token values or directions.
    random=random*(hidden.float().square().mean().sqrt()/random.square().mean().sqrt().clamp_min(1e-8))
    permutation=torch.roll(torch.arange(hidden.shape[1],device=hidden.device),1)
    return {'random_aligned_plan':random.to(hidden.dtype),
            'permuted_aligned_plan':hidden[:,permutation,:]}, {
        'seed':seed,'random_conditioning':'shape_and_single_global_source_rms_only',
        'permutation':'fixed_cyclic_shift_one','permutation_indices':permutation.cpu().tolist()}
