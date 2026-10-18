#!/usr/bin/env python3
"""Reference-conditioning source interventions; frozen derivative of six-arm diagnostic.
Frozen derivative of compare_training_free_interfaces.py, SHA256 95283a4f6ef5660f9a94f95faa25d5ed380f8bcb39447557279fe28c1f60b5ef."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
import yaml
from statebridge_alignment import align_message, vocab_reference
from alignment_source_controls import source_controls
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from humor_generator_v35.qwen_backend import QwenBackend, model_device
from humor_generator_v35.training.public_bridge import load_context_index
from humor_generator_v35.training.selected_plan import selected_plan_channels, selected_plan_sha256
from humor_generator_v35.homer.caption_parsing import parse_caption_strict

SCHEMES={'alignment_controls':['no_memory','text_plan','exact_embeddings',
    'raw_hidden_replayed_plan','global_norm_hidden_replayed_plan','statebridge_replayed_plan','random_aligned_plan','permuted_aligned_plan','global_norm_token_embeddings']}



def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def ids(tokenizer,text,device):
    return tokenizer(text,add_special_tokens=False,return_tensors='pt')['input_ids'].to(device)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scheme',choices=['all',*SCHEMES],default='all')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-clusters',type=int,default=1)
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
        'hidden_method':'same-token final hidden state, raw / global vocabulary norm / full StateBridge-inspired alignment; fixed full plan replay, not official pipeline',
        'alignment_parameters':{'regularization':1e-3,'anchor_weight':.3,'proper_rotation':True},
        'alignment_sha256':sha(Path(__file__).with_name('statebridge_alignment.py')),
        'controls_sha256':sha(Path(__file__).with_name('alignment_source_controls.py')),
        'parent_runner_sha256':'6adfa9eea189805420e7a2e90e75da445b1cd34bc41816f58fb75e926eafaed9',
        'source_interventions':{'seed':20261018,'permutation':'cyclic_shift_one','reference':'same_actual_plan_tokens'},
        'reference_conditioning':'alignment uses the actual plan token embeddings; recovery is not proof of additional information',
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
        header=ids(tokenizer,'Selected plan:\n',device)
        sender_ids=torch.cat([header,plan_ids],dim=1)
        torch.cuda.synchronize();t=time.perf_counter()
        out=model(input_ids=sender_ids,attention_mask=torch.ones_like(sender_ids),use_cache=False,
                  output_hidden_states=True,logits_to_keep=1)
        hidden=out.hidden_states[-1][:,-plan_ids.shape[1]:,:].detach().clone()
        del out
        torch.cuda.synchronize();sender_seconds=time.perf_counter()-t
        memories['raw_hidden_replayed_plan']=hidden.to(base.dtype)
        vocabulary=embedding.weight.detach()
        target_norm=vocab_reference(vocabulary,4096)
        memories['global_norm_hidden_replayed_plan']=(hidden.float()*
            (target_norm/hidden.float().norm(dim=-1,keepdim=True).clamp_min(1e-6))).to(base.dtype)
        # Persist before the dense decomposition, so a timeout is distinguishable from generation failure.
        with (args.output/'alignment_events.jsonl').open('a') as f:
            f.write(json.dumps({'cluster':cluster,'event':'alignment_started',
                'plan_tokens':plan_ids.shape[1],'width':hidden.shape[-1],
                'sender_seconds':sender_seconds})+'\n')
        torch.cuda.synchronize();t=time.perf_counter()
        aligned,diagnostics=align_message(hidden[0],base[0],vocabulary,
            regularization=1e-3,anchor_weight=.3,proper_rotation=True)
        torch.cuda.synchronize();alignment_seconds=time.perf_counter()-t
        memories['statebridge_replayed_plan']=aligned.unsqueeze(0).to(base.dtype)
        sender_diagnostics={'alignment':diagnostics,'sender_seconds':sender_seconds,
            'alignment_seconds':alignment_seconds,'shared_preparation_not_per_arm_cost':True}
        encode_seconds=sender_seconds+alignment_seconds
        with (args.output/'alignment_events.jsonl').open('a') as f:
            f.write(json.dumps({'cluster':cluster,'event':'alignment_completed',
                'diagnostics':sender_diagnostics})+'\n')
        controls,control_spec=source_controls(hidden)
        control_diagnostics={}
        memories['global_norm_token_embeddings']=(base.float()*
            (target_norm/base.float().norm(dim=-1,keepdim=True).clamp_min(1e-6))).to(base.dtype)
        for control_name,control_hidden in controls.items():
            with (args.output/'alignment_events.jsonl').open('a') as f:
                f.write(json.dumps({'cluster':cluster,'event':'control_started','arm':control_name})+'\n')
            torch.cuda.synchronize();t=time.perf_counter()
            control_aligned,control_diag=align_message(control_hidden[0],base[0],vocabulary,
                regularization=1e-3,anchor_weight=.3,proper_rotation=True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            memories[control_name]=control_aligned.unsqueeze(0).to(base.dtype)
            control_diagnostics[control_name]={'alignment':control_diag,'seconds':elapsed}
            with (args.output/'alignment_events.jsonl').open('a') as f:
                f.write(json.dumps({'cluster':cluster,'event':'control_completed','arm':control_name,
                    'diagnostics':control_diagnostics[control_name]})+'\n')
            encode_seconds+=elapsed
        sender_diagnostics.update(source_control_spec=control_spec,source_control_results=control_diagnostics)
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
    assert sha(Path(__file__).with_name('alignment_source_controls.py'))==manifest['controls_sha256'],'controls changed during run'
    assert sha(Path(__file__).with_name('statebridge_alignment.py'))==manifest['alignment_sha256'],'alignment changed during run'
    assert all(sha(ROOT/p)==h for p,h in manifest['source_sha256'].items()),'source changed during run'
    summary={'status':'completed_not_quality_pass','scope':manifest['scope'],'records':len(records),'input_checks':stats,'by_arm':{}}
    for arm in arms:
        group=[r for r in records if r['arm']==arm];captions=[r for r in group if r['task']=='caption'];recovery=[r for r in group if r['task']=='recovery']
        summary['by_arm'][arm]={'captions':len(captions),'format_ok':sum(r['parse_status']=='ok' for r in captions),
            'recovery_attempts':len(recovery),'exact_plan_recovery':sum(r['exact_plan_recovery'] for r in recovery)}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
