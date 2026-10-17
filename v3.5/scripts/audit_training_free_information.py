#!/usr/bin/env python3
"""Exact toy counterexamples for claims about probability-weighted embeddings.
No model-quality claim and no production-path verification.
"""
import json
from fractions import Fraction as F


def weighted(p, embeddings):
    return sum(a*b for a,b in zip(p,embeddings))


def replay_distributions(tokens):
    # Causal toy next-token model: first distribution is fixed; later depends
    # on the previous observed token. Matches t-1 teacher-forced indexing.
    result=[]
    for i in range(len(tokens)):
        result.append((F(1,2),F(1,2)) if i==0 else
                      ((F(3,4),F(1,4)) if tokens[i-1]==0 else (F(1,4),F(3,4))))
    return result


def main():
    embeddings=[F(-1),F(0),F(1)]
    p=[F(1,2),F(0),F(1,2)];q=[F(0),F(1),F(0)]
    assert p!=q and weighted(p,embeddings)==weighted(q,embeddings)==0
    left=replay_distributions([0,1,0]);right=replay_distributions([0,1,1])
    assert left==right
    # Top-k renormalization error <= 2 R delta, for ||embedding|| <= R.
    probs=[F(3,5),F(3,10),F(1,10)];e=[F(-1),F(1),F(1)]
    delta=probs[-1];soft=weighted(probs,e)
    truncated=weighted([v/(1-delta) for v in probs[:2]],e[:2])
    assert abs(soft-truncated)<=2*delta
    result={'status':'pass','scope':'mathematical_toy_counterexamples_not_model_quality',
        'weighted_embedding_noninjective':True,'replay_blind_to_final_token':True,
        'topk_error_bound_example':{'error':str(abs(soft-truncated)),'bound':str(2*delta)},
        'norm_cancellation_example':{'input_norms':[1,1],'mixture_norm':0},
        'implication':'A continuous embedding need not preserve a distribution or the complete chosen message.'}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
