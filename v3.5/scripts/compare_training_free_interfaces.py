#!/usr/bin/env python3
"""Train-only, serial, training-free interface diagnostics; custom shared prompt."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from humor_generator_v35.qwen_backend import QwenBackend, model_device
from humor_generator_v35.training.public_bridge import load_context_index
from humor_generator_v35.training.selected_plan import selected_plan_channels, selected_plan_sha256
from humor_generator_v35.homer.caption_parsing import parse_caption_strict

SCHEMES={'text_controls':['no_memory','text_plan','exact_embeddings'],
         'soft_distribution':['soft_top64_replayed_plan'],
         'normalized_hidden':['norm_hidden_replayed_plan']}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def ids(tokenizer,text,device):
    return tokenizer(text,add_special_tokens=False,return_tensors='pt')['input_ids'].to(device)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scheme',choices=['all',*SCHEMES],default='all')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-clusters',type=int,default=16)
    ap.add_argument('--cluster-offset',type=int,default=0)
    args=ap.parse_args()
    if not 1<=args.max_clusters<=16: raise ValueError('diagnostic limit is 1..16 train clusters')
    if args.cluster_offset < 0 or args.cluster_offset+args.max_clusters > 16:
        raise ValueError('batch must stay inside the fixed 16-cluster diagnostic set')
    args.output.mkdir(parents=True,exist_ok=False)
    cfgpath=ROOT/'configs/pilot/cross_attention_caption_selected_plan_v2.yaml'
    cfg=yaml.safe_load(cfgpath.read_text());mc=cfg['model']
    trainpath=ROOT/cfg['data']['dataset']/'train.jsonl'
    rows=[json.loads(l) for l in trainpath.read_text().splitlines() if l.strip()]
    contextpath=ROOT/cfg['data']['context_index']; contexts=load_context_index(contextpath)
    selection_seed='v35_effectiveness_diagnostic_20261017:'
    clusters=sorted({r['cluster_id'] for r in rows},key=lambda c:hashlib.sha256((selection_seed+c).encode()).hexdigest())[args.cluster_offset:args.cluster_offset+args.max_clusters]
    arms=sum(SCHEMES.values(),[]) if args.scheme=='all' else SCHEMES[args.scheme]
    manifest={'scope':'train_only_custom_prompt_not_official_homer_reproduction',
        'arms':arms,'clusters':clusters,'cluster_offset':args.cluster_offset,'selection_seed':selection_seed,'seeds':[20261017,20261018],
        'training':False,'model_revision':mc['revision'],'precision':'nf4_bf16',
        'train_sha256':sha(trainpath),'context_sha256':sha(contextpath),
        'config_sha256':sha(cfgpath),'script_sha256':sha(Path(__file__)),
        'soft_method':'top64 renormalized next-token distributions under teacher-forced selected-plan replay; CIPHER-inspired, not original CIPHER',
        'hidden_method':'same-token final normalized hidden state with per-token norm calibrated to token embedding norm; not StateBridge',
        'recovery':'greedy JSON extraction; caption uses two sampled seeds; no humour judge'}
    manifest['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src').rglob('*.py'))}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    backend=QwenBackend.load(mc['name'],revision=mc['revision'],load_in_4bit=True,
        min_visual_tokens=mc['min_visual_tokens'],max_visual_tokens=mc['max_visual_tokens'])
    model=backend.model;tokenizer=backend.processor.tokenizer;device=model_device(model)
    embedding=model.get_input_embeddings()
    stats=[];records=[]
    for cluster in clusters:
        context=contexts[cluster]
        plan=json.dumps({'entities':list(context.selected_entities),'conflict':context.selected_conflict,
                        'paths':{k:list(v) for k,v in context.selected_paths.items()}},ensure_ascii=False)
        plan_ids=ids(tokenizer,plan,device);base=embedding(plan_ids)
        memories={'exact_embeddings':base}
        sender_diagnostics={}
        if any(a in arms for a in ['soft_top64_replayed_plan','norm_hidden_replayed_plan']):
            header=ids(tokenizer,'Selected plan:\n',device)
            sender_ids=torch.cat([header,plan_ids],dim=1)
            t=time.perf_counter()
            out=model(input_ids=sender_ids,attention_mask=torch.ones_like(sender_ids),use_cache=False,
                      output_hidden_states=True,logits_to_keep=plan_ids.shape[1]+1)
            if 'soft_top64_replayed_plan' in arms:
                # Logits at t-1 predict token t; do not use the next position by mistake.
                logits=out.logits[:,-plan_ids.shape[1]-1:-1,:].float()
                scores,indices=logits.topk(64,dim=-1)
                weights=scores.softmax(-1)
                sender_diagnostics['selected_token_in_top64_fraction']=float(indices.eq(plan_ids.unsqueeze(-1)).any(-1).float().mean())
                sender_diagnostics['selected_token_mean_probability']=float(logits.log_softmax(-1).gather(-1,plan_ids.unsqueeze(-1)).exp().mean())
                soft=(embedding(indices).float()*weights.unsqueeze(-1)).sum(-2)
                memories['soft_top64_replayed_plan']=soft.to(base.dtype)
            if 'norm_hidden_replayed_plan' in arms:
                hidden=out.hidden_states[-1][:,-plan_ids.shape[1]:,:].float()
                hidden=hidden*(base.float().norm(dim=-1,keepdim=True)/hidden.norm(dim=-1,keepdim=True).clamp_min(1e-8))
                memories['norm_hidden_replayed_plan']=hidden.to(base.dtype)
            torch.cuda.synchronize();encode_seconds=time.perf_counter()-t
            del out
        else: encode_seconds=0.0
        for task in ['recovery','caption']:
            instruction=('Return only a JSON object with keys entities, conflict, paths, copied from the supplied plan. If no plan is supplied return {}.' if task=='recovery' else
                         'Write one funny cartoon caption grounded in the description. Use the supplied plan if useful. Return exactly ##Caption: followed by the caption, then ##Explanation: followed by a brief explanation.')
            marker='V35_PLAN_SLOT_7F29'
            user='Description:\n'+context.standard_description+'\nSelected plan:\n'+marker+'\nTask:\n'+instruction
            rendered=backend.processor.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
            assert rendered.count(marker)==1
            left,right=rendered.split(marker)
            left_ids,right_ids=ids(tokenizer,left,device),ids(tokenizer,right,device)
            # Explicit segmented tokenization is identical across text/embedding arms.
            full_ids=torch.cat([left_ids,plan_ids,right_ids],dim=1)
            if 'exact_embeddings' in arms:
                normal=model(input_ids=full_ids,attention_mask=torch.ones_like(full_ids),use_cache=False,logits_to_keep=1).logits.float()
                direct=model(inputs_embeds=embedding(full_ids),attention_mask=torch.ones_like(full_ids),
                             position_ids=torch.arange(full_ids.shape[1],device=device).view(1,1,-1).expand(3,1,-1),use_cache=False,logits_to_keep=1).logits.float()
                err=float((normal-direct).abs().max())
                if not torch.allclose(normal,direct,atol=1e-5,rtol=1e-5): raise RuntimeError('exact embedding input control failed')
                stats.append({'cluster':cluster,'task':task,'exact_embedding_max_error':err})
            for arm in arms:
                if arm=='no_memory':
                    input_ids=torch.cat([left_ids,ids(tokenizer,'[No plan supplied]',device),right_ids],dim=1)
                    inputs={'input_ids':input_ids,'attention_mask':torch.ones_like(input_ids)}
                elif arm=='text_plan':
                    inputs={'input_ids':full_ids,'attention_mask':torch.ones_like(full_ids)}
                else:
                    vector=torch.cat([embedding(left_ids),memories[arm],embedding(right_ids)],dim=1)
                    inputs={'inputs_embeds':vector,'attention_mask':torch.ones(vector.shape[:2],device=device,dtype=torch.long)}
                for seed in ([20261017] if task=='recovery' else [20261017,20261018]):
                    torch.manual_seed(seed)
                    kwargs={'max_new_tokens':384 if task=='recovery' else 128,'do_sample':task=='caption',
                            'repetition_penalty':1.0,'use_cache':True,'return_dict_in_generate':True,'output_scores':True}
                    if task=='caption': kwargs.update(temperature=1.0,top_p=1.0)
                    torch.cuda.synchronize();start=time.perf_counter()
                    output=model.generate(**inputs,**kwargs)
                    torch.cuda.synchronize();seconds=time.perf_counter()-start
                    count=len(output.scores);tokens=output.sequences[:,-count:] if count else output.sequences[:,:0]
                    raw=tokenizer.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
                    item={'cluster':cluster,'arm':arm,'task':task,'seed':seed,'raw_output':raw,
                          'selected_plan_sha256':selected_plan_sha256(context),'plan_tokens':plan_ids.shape[1],
                          'input_tokens':inputs['attention_mask'].shape[1],'generated_tokens':count,
                          'generation_seconds':seconds,'shared_sender_encode_seconds':encode_seconds,
                          'sender_diagnostics':sender_diagnostics,'reached_token_cap':count==kwargs['max_new_tokens']}
                    if task=='caption':
                        parsed=parse_caption_strict(raw);item.update(caption=parsed.caption,parse_status=parsed.status)
                    else:
                        try:
                            value=json.loads(raw)
                            item.update(json_valid=True,exact_plan_recovery=value==json.loads(plan))
                        except ValueError: item.update(json_valid=False,exact_plan_recovery=False)
                    records.append(item)
                    with (args.output/'records.jsonl').open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
                    del output
            print(json.dumps({'cluster':cluster,'task':task,'records_written':len(records)}),flush=True)
    assert sha(Path(__file__))==manifest['script_sha256'],'script changed during run'
    assert all(sha(ROOT/p)==h for p,h in manifest['source_sha256'].items()),'source changed during run'
    summary={'status':'completed_not_quality_pass','scope':manifest['scope'],'records':len(records),'input_checks':stats,'by_arm':{}}
    for arm in arms:
        group=[r for r in records if r['arm']==arm];captions=[r for r in group if r['task']=='caption'];recovery=[r for r in group if r['task']=='recovery']
        summary['by_arm'][arm]={'captions':len(captions),'format_ok':sum(r['parse_status']=='ok' for r in captions),
            'recovery_attempts':len(recovery),'exact_plan_recovery':sum(r['exact_plan_recovery'] for r in recovery)}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
