"""Independent implementation of StateBridge's alignment equations.
Reference: arXiv:2608.13317v1; official revision 3f6bf5442c6e8848555a6132516e6d36f35444fb.
No model calls, training, prefix selection or generation are implemented here.
"""
import math
import torch


def symmetric_power(matrix, power):
    values, basis=torch.linalg.eigh((matrix+matrix.T)/2)
    return (basis*values.clamp_min(1e-6).pow(power).unsqueeze(0))@basis.T


def orthogonal_map(cross_covariance, proper_rotation=True):
    left, _, right_t=torch.linalg.svd(cross_covariance,full_matrices=False)
    signs=torch.ones(left.shape[1],device=left.device,dtype=left.dtype)
    if proper_rotation and torch.linalg.det(left@right_t)<0:
        signs[-1]=-1
    return (left*signs.unsqueeze(0))@right_t


def vocab_reference(vocabulary, chunk_size):
    total=0.0
    for start in range(0,len(vocabulary),chunk_size):
        block=vocabulary[start:start+chunk_size].float()
        if not torch.isfinite(block).all():raise ValueError('nonfinite vocabulary')
        total+=float(block.norm(dim=-1).double().sum())
    return total/len(vocabulary)


def nearest_vocab(vectors, vocabulary, chunk_size=4096):
    if chunk_size<1:raise ValueError('chunk size must be positive')
    queries=torch.nn.functional.normalize(vectors,dim=-1)
    best=torch.full((len(vectors),),-float('inf'),device=vectors.device,dtype=vectors.dtype)
    indices=torch.zeros(len(vectors),device=vectors.device,dtype=torch.long)
    for start in range(0,len(vocabulary),chunk_size):
        block=torch.nn.functional.normalize(vocabulary[start:start+chunk_size].to(vectors.dtype),dim=-1)
        scores,local=(queries@block.T).max(dim=-1)
        better=scores>best
        indices=torch.where(better,local+start,indices)
        best=torch.maximum(best,scores)
    return indices


@torch.no_grad()
def align_message(states, references, vocabulary, regularization=1e-3,
                  anchor_weight=.3, proper_rotation=True, chunk_size=4096):
    if not all(t.is_floating_point() for t in (states,references,vocabulary)):
        raise ValueError('floating tensors required')
    if states.ndim!=2 or states.shape!=references.shape or states.shape[0]<2:
        raise ValueError('require exactly aligned [tokens,width] tensors with at least two tokens')
    if vocabulary.ndim!=2 or vocabulary.shape[1]!=states.shape[1] or len(vocabulary)==0:
        raise ValueError('vocabulary dimension mismatch or empty vocabulary')
    if states.device!=references.device or states.device!=vocabulary.device:
        raise ValueError('all tensors must share device')
    if not math.isfinite(regularization) or regularization<=0 or not 0<=anchor_weight<=1 or chunk_size<1:
        raise ValueError('invalid regularization, anchor weight or chunk size')
    if not torch.isfinite(states).all() or not torch.isfinite(references).all():
        raise ValueError('nonfinite message/reference')
    # Float64 remains available for independent mathematical tests, float32 for real inputs.
    dtype=torch.float64 if states.dtype==torch.float64 else torch.float32
    h,e=states.to(dtype),references.to(dtype)
    hm,em=h.mean(0,keepdim=True),e.mean(0,keepdim=True)
    hc,ec=h-hm,e-em
    eye=torch.eye(h.shape[1],device=h.device,dtype=dtype)
    ch=hc.T@hc/len(h)+regularization*eye
    ce=ec.T@ec/len(e)+regularization*eye
    hw=hc@symmetric_power(ch,-.5);ew=ec@symmetric_power(ce,-.5)
    rotation=orthogonal_map(hw.T@ew,proper_rotation)
    reconstructed=hw@rotation@symmetric_power(ce,.5)+em
    target_norm=vocab_reference(vocabulary,chunk_size)
    if target_norm<=0:raise ValueError('vocabulary norm must be positive')
    calibrated=reconstructed*(target_norm/reconstructed.norm(dim=-1,keepdim=True).clamp_min(1e-6))
    nearest=None
    if anchor_weight>0:
        nearest=nearest_vocab(calibrated,vocabulary,chunk_size)
        aligned=(1-anchor_weight)*calibrated+anchor_weight*vocabulary[nearest].to(dtype)
    else:aligned=calibrated
    if not torch.isfinite(aligned).all():raise RuntimeError('nonfinite alignment result')
    diagnostics={'proper_rotation':proper_rotation,'determinant':float(torch.linalg.det(rotation)),
        'rotation_orthogonality_max_error':float((rotation.T@rotation-eye).abs().max()),
        'reference_rmse_before_norm_and_anchor':float((reconstructed-e).square().mean().sqrt()),
        'target_vocab_norm':target_norm,'regularization':regularization,'anchor_weight':anchor_weight,
        'nearest_vocab_ids':None if nearest is None else nearest.cpu().tolist(),
        'scope':'alignment_only_not_semantic_or_quality_guarantee'}
    return aligned.to(states.dtype),diagnostics
