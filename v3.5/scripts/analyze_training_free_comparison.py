#!/usr/bin/env python3
"""Strict paired analysis of diagnostic records; never judges humour quality."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re


def normalized(value):
    return ' '.join(value.casefold().split()) if isinstance(value,str) else None


def recovery_scores(raw, context, no_memory=False):
    try:
        value=json.loads(raw)
    except (ValueError,TypeError):
        return {'json_valid':False,'empty_object_correct':False,'entity_set_exact':False,
                'paths_exact':False,'conflict_normalized_exact':False,'full_plan_exact':False}
    target={'entities':context['selected_entities'],'conflict':context['selected_conflict'],
            'paths':context['selected_paths']}
    valid=isinstance(value,dict)
    entities=value.get('entities') if valid else None
    entities_valid=isinstance(entities,list) and all(isinstance(e,str) for e in entities)
    return {'json_valid':valid,'empty_object_correct':no_memory and value=={},
            'entity_set_exact':bool(entities_valid and len(entities)==len(set(entities)) and set(entities)==set(target['entities'])),
            'paths_exact':valid and value.get('paths')==target['paths'],
            'conflict_normalized_exact':valid and normalized(value.get('conflict'))==normalized(target['conflict']),
            'full_plan_exact':value==target}


def unwrap_json_fence(raw):
    """Secondary content-only measure: unwrap one whole-output fence, never extract substrings."""
    match=re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*',raw,re.DOTALL)
    return (match.group(1), True) if match else (raw, False)


def paired_interval(differences, repeats=10000):
    # Resample contests, keeping both candidate seeds together in their mean.
    mean=sum(differences)/len(differences)
    if len(differences)<2:return {'mean':mean,'interval95':None,'reason':'fewer_than_two_contests'}
    rng=random.Random(20261017)
    samples=sorted(sum(rng.choice(differences) for _ in differences)/len(differences) for _ in range(repeats))
    return {'mean':mean,'interval95':[samples[int(.025*repeats)],samples[min(repeats-1,int(.975*repeats))]],
            'resampling_unit':'contest','repeats':repeats,'scope':'descriptive_format_only_not_humour_or_confirmatory'}


def analyze(run_dirs, context_path):
    blob=context_path.read_bytes();context_hash=hashlib.sha256(blob).hexdigest()
    contexts={c['cluster_id']:c for c in (json.loads(l) for l in blob.decode().splitlines() if l.strip())}
    all_records={};all_clusters=[];reference=None;expected=set()
    invariants=['arms','seeds','model_revision','precision','train_sha256','context_sha256','config_sha256','script_sha256','source_sha256','alignment_sha256','alignment_parameters']
    for directory in run_dirs:
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['context_sha256']!=context_hash:raise ValueError('context hash mismatch')
        if reference is None:reference=manifest
        elif any(manifest.get(k)!=reference.get(k) for k in invariants):raise ValueError('incompatible run contracts')
        if set(all_clusters)&set(manifest['clusters']):raise ValueError('overlapping cluster batches')
        if len(set(manifest['clusters']))!=len(manifest['clusters']):raise ValueError('duplicate manifest cluster')
        all_clusters.extend(manifest['clusters'])
        for c in manifest['clusters']:
            if c not in contexts:raise ValueError('missing context')
            for arm in manifest['arms']:
                expected.add((c,arm,'recovery',20261017))
                expected.update((c,arm,'caption',seed) for seed in manifest['seeds'])
        for line in (directory/'records.jsonl').read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line);key=tuple(row[k] for k in ['cluster','arm','task','seed'])
            if key in all_records:raise ValueError('duplicate record key')
            if row['cluster'] not in manifest['clusters'] or row['arm'] not in manifest['arms']:
                raise ValueError('record outside batch population')
            all_records[key]=row
    if not reference:raise ValueError('no runs')
    missing=expected-set(all_records);extra=set(all_records)-expected
    if missing or extra:raise ValueError('incomplete or unexpected schedule: missing={} extra={}'.format(len(missing),len(extra)))
    output={'status':'complete_descriptive_analysis','contests':len(all_clusters),'records':len(all_records),
            'humour_quality_evaluated':False,'arms':{},'paired_caption_comparisons':[]}
    for arm in reference['arms']:
        rows=[r for k,r in all_records.items() if k[1]==arm]
        caps=[r for r in rows if r['task']=='caption'];recs=[r for r in rows if r['task']=='recovery']
        scores=[recovery_scores(r['raw_output'],contexts[r['cluster']],arm=='no_memory') for r in recs]
        unwrapped=[unwrap_json_fence(r['raw_output']) for r in recs]
        content_scores=[recovery_scores(raw,contexts[r['cluster']],arm=='no_memory') for r,(raw,_) in zip(recs,unwrapped)]
        output['arms'][arm]={'caption_count':len(caps),'caption_parse_status':dict(Counter(r['parse_status'] for r in caps)),
            'caption_token_cap_count':sum(bool(r['reached_token_cap']) for r in caps),
            'recovery_count':len(recs),'recovery_token_cap_count':sum(bool(r['reached_token_cap']) for r in recs),
            'recovery_field_counts':{k:sum(bool(v[k]) for v in scores) for k in scores[0]},
            'secondary_fence_normalized_recovery':{
                'scope':'posthoc_content_diagnostic_not_strict_format_compliance',
                'fences_removed':sum(removed for _,removed in unwrapped),
                'field_counts':{k:sum(bool(v[k]) for v in content_scores) for k in content_scores[0]}},
            'interpretation':'no-memory should return {}, not reconstruct missing plan' if arm=='no_memory' else 'field exactness is lexical/structural, not semantic judgement'}
        if arm!='text_plan' and 'text_plan' in reference['arms']:
            diffs=[];equal=0;pairs=0
            for c in all_clusters:
                per=[]
                for seed in reference['seeds']:
                    a=all_records[c,arm,'caption',seed];b=all_records[c,'text_plan','caption',seed]
                    per.append(int(a['parse_status']=='ok')-int(b['parse_status']=='ok'))
                    if a['parse_status']=='ok' and b['parse_status']=='ok':
                        equal+=normalized(a['caption'])==normalized(b['caption']);pairs+=1
                diffs.append(sum(per)/len(per))
            output['paired_caption_comparisons'].append({'arm':arm,'reference':'text_plan',
                'format_rate_difference':paired_interval(diffs),
                'normalized_exact_caption_matches':equal,'both_valid_pairs':pairs})
    return output


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,action='append',required=True)
    ap.add_argument('--context-index',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();result=analyze(args.run_dir,args.context_index)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
