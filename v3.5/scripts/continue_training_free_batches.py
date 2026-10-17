#!/usr/bin/env python3
"""Bounded serial PJM continuation; never submits the next batch before completion."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument('--first-job',type=int,required=True)
ap.add_argument('--output-prefix',required=True)
args=ap.parse_args()
script=root/'scripts/compare_training_free_interfaces.py'
expected=hashlib.sha256(script.read_bytes()).hexdigest()
job=args.first_job
deadline=time.monotonic()+72*3600
for batch in range(4):
    output=root/(args.output_prefix+'_batch'+str(batch))
    print('Waiting batch {} job {} output {}'.format(batch,job,output),flush=True)
    terminal_seen=None
    while True:
        if time.monotonic()>deadline: raise SystemExit('72-hour monitoring limit; no further submission')
        if hashlib.sha256(script.read_bytes()).hexdigest()!=expected:
            raise SystemExit('Comparison script changed; stop automatic submissions')
        summary=output/'summary.json'
        if summary.exists():
            result=json.loads(summary.read_text())
            manifest=json.loads((output/'manifest.json').read_text())
            if result['status']!='completed_not_quality_pass' or result['records']!=60:
                raise SystemExit('Incomplete batch; no next submission')
            if manifest['cluster_offset']!=batch*4 or manifest['script_sha256']!=expected:
                raise SystemExit('Batch provenance mismatch')
            # Wait for scheduler termination, not only report creation.
            stat=root/('v35_training_free_compare.pjm.{}.stats'.format(job))
            if stat.exists():
                content=stat.read_text()
                match=re.search(r'^\s*EXIT CODE\s*:\s*(\d+)\s*$',content,re.M)
                if match:
                    if int(match.group(1))!=0: raise SystemExit('Nonzero job exit')
                    break
        stat=root/('v35_training_free_compare.pjm.{}.stats'.format(job))
        if stat.exists() and terminal_seen is None: terminal_seen=time.monotonic()
        if terminal_seen is not None and time.monotonic()-terminal_seen>300:
            raise SystemExit('Terminal stats but no verified complete report; stop')
        time.sleep(60)
    print('Batch {} complete; quality not evaluated'.format(batch),flush=True)
    if batch<3:
        next_output=args.output_prefix+'_batch'+str(batch+1)
        if (root/next_output).exists(): raise SystemExit('Next output already exists; refuse duplicate')
        env='COMPARE_OUTPUT={},MAX_CLUSTERS=4,CLUSTER_OFFSET={}'.format(next_output,(batch+1)*4)
        text=subprocess.check_output(['pjsub','-x',env,'jobs/v35_training_free_compare.pjm'],cwd=str(root),universal_newlines=True)
        print(text,flush=True)
        match=re.search(r'Job (\d+) submitted',text)
        if not match: raise SystemExit('Submission status unclear; stop without retry')
        job=int(match.group(1))
print('All 16 train clusters completed serially; inspect records before quality conclusions',flush=True)
