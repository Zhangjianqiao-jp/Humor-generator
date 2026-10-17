"""Analysis must not silently drop malformed outputs or incomplete schedules."""
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
import hashlib

spec=importlib.util.spec_from_file_location('tf_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_training_free_comparison.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
CONTEXT={'selected_entities':['shark','periscope'],'selected_conflict':'Predator vs observer',
         'selected_paths':{'shark':['shark','boat'],'periscope':['periscope','submarine']}}

class AnalysisTests(unittest.TestCase):
    def test_empty_object_correct_only_for_no_memory(self):
        self.assertTrue(a.recovery_scores('{}',CONTEXT,True)['empty_object_correct'])
        self.assertFalse(a.recovery_scores('{}',CONTEXT)['empty_object_correct'])
        self.assertFalse(a.recovery_scores('{}',CONTEXT,True)['full_plan_exact'])
    def test_invalid_json_is_failure_not_missing(self):
        result=a.recovery_scores('##Caption: joke',CONTEXT)
        self.assertTrue(all(v is False for v in result.values()))
    def test_order_insensitive_entities_but_order_sensitive_paths(self):
        payload={'entities':['periscope','shark'],'conflict':'  PREDATOR vs observer ',
                 'paths':{'shark':['boat','shark'],'periscope':['periscope','submarine']}}
        result=a.recovery_scores(json.dumps(payload),CONTEXT)
        self.assertTrue(result['entity_set_exact'])
        self.assertTrue(result['conflict_normalized_exact'])
        self.assertFalse(result['paths_exact'])
        self.assertFalse(result['full_plan_exact'])
    def test_malformed_entity_values_not_accepted(self):
        result=a.recovery_scores('{"entities": [null, {}]}',CONTEXT)
        self.assertFalse(result['entity_set_exact'])
    def fixture(self, folder):
        context=folder/'contexts.jsonl'
        context.write_text(json.dumps(dict(CONTEXT,cluster_id='train1'))+'\n')
        run=folder/'run';run.mkdir()
        manifest={'clusters':['train1'],'arms':['no_memory'],'seeds':[20261017,20261018],
                  'context_sha256':hashlib.sha256(context.read_bytes()).hexdigest()}
        (run/'manifest.json').write_text(json.dumps(manifest))
        rows=[{'cluster':'train1','arm':'no_memory','task':task,'seed':seed,
               'raw_output':'{}','parse_status':'invalid_markers','caption':'','reached_token_cap':False}
              for task,seed in [('recovery',20261017),('caption',20261017),('caption',20261018)]]
        return run,context,rows
    def test_incomplete_schedule_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,context,rows=self.fixture(Path(tmp))
            (run/'records.jsonl').write_text('\n'.join(json.dumps(r) for r in rows[:-1]))
            with self.assertRaisesRegex(ValueError,'incomplete'):a.analyze([run],context)
    def test_duplicate_record_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,context,rows=self.fixture(Path(tmp))
            (run/'records.jsonl').write_text('\n'.join(json.dumps(r) for r in rows+[rows[0]]))
            with self.assertRaisesRegex(ValueError,'duplicate'):a.analyze([run],context)
    def test_complete_failures_stay_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,context,rows=self.fixture(Path(tmp))
            (run/'records.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            result=a.analyze([run],context)
            self.assertEqual(result['arms']['no_memory']['caption_count'],2)
            self.assertEqual(result['arms']['no_memory']['caption_parse_status']['invalid_markers'],2)
            self.assertFalse(result['humour_quality_evaluated'])
    def test_fence_unwrap_is_whole_output_only(self):
        self.assertEqual(a.unwrap_json_fence('```json\n{}\n```'),('{}',True))
        for raw in ['Here is JSON: ```json\n{}\n```','```json\n{}\n``` trailing',
                    '```python\n{}\n```','{}']:
            self.assertEqual(a.unwrap_json_fence(raw),(raw,False))
    def test_fenced_content_does_not_rewrite_strict_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,context,rows=self.fixture(Path(tmp))
            rows[0]['raw_output']='```json\n{}\n```'
            (run/'records.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            result=a.analyze([run],context)['arms']['no_memory']
            self.assertEqual(result['recovery_field_counts']['empty_object_correct'],0)
            self.assertEqual(result['secondary_fence_normalized_recovery']['field_counts']['empty_object_correct'],1)

    def test_single_contest_no_spurious_interval(self):
        self.assertIsNone(a.paired_interval([1.0])['interval95'])
    def test_alignment_contract_changes_rejected(self):
        for field in ['alignment_sha256','alignment_parameters']:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                run,context,rows=self.fixture(Path(tmp))
                (run/'records.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
                second=Path(tmp)/'second';second.mkdir()
                manifest=json.loads((run/'manifest.json').read_text())
                manifest[field]='changed'
                (second/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'incompatible run contracts'):
                    a.analyze([run,second],context)

if __name__=='__main__':unittest.main()
