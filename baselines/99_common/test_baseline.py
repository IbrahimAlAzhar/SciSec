"""CPU regression tests. All model responses below are synthetic fixtures."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import types
import unittest
import warnings
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
from firewall_core import (CSVJournal, LocalClient, Firewall, alignment_json,
                           review_json, audit_prompt, window_starts)
from run_baseline import load_rows, process, report, CALL_FIELDS, main, check_output_directory
from run_job import parser as job_parser

REVIEW = {'rating': 8, 'weaknesses': 'Missing external validation.',
          'limitations_assessment': 'Limited study size.'}
AUDIT = {'observation': 'Observation', 'thought': 'Assessment', 'conclusion': False}

class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(m['content']) for m in messages)//4+10))

class FixtureFirewall:
    def __init__(self, audits, fail=False):
        self.client = argparse.Namespace(calls=0)
        self.prompt_guard = None
        self.audits = iter(audits); self.fail = fail; self.prompts = []
    def fit_document(self, document, policy): return document, 1.0
    def review(self, document, key, repair=False):
        self.client.calls += 1
        return dict(REVIEW, rating=5 if repair else 8)
    def audit(self, document, action, key, stage='alignment'):
        self.client.calls += 1
        self.prompts.append(audit_prompt(document, action))
        if self.fail: raise RuntimeError('synthetic outage')
        return dict(AUDIT, conclusion=next(self.audits))


def row(**kw):
    return dict(paper_id='001', condition='A5_injection', family='instruction', split='test',
        label=1, text='Manuscript\nLimitations\nSmall sample.', historical_asr=1,
        historical_rating=8, historical_clean_rating=5,
        historical_weaknesses='Missing validation', historical_clean_weaknesses='Small sample', **kw)

class BaselineTests(unittest.TestCase):
    def test_output_directory_allows_launcher_monitor(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'gpu_memory.csv').write_text('timestamp,memory.used\n')
            (out/'output.txt').write_text('Server ready\n')
            check_output_directory(out)

    def test_output_directory_protects_existing_benchmark_artifacts(self):
        for name in ('rows_checkpoint.csv', 'intermediate_calls.csv', 'rows_final.csv',
                     'metrics.csv', 'metrics.json', 'manifest.json', 'unknown.csv'):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                (out/'gpu_memory.csv').touch()
                (out/name).touch()
                with self.assertRaises(SystemExit):
                    check_output_directory(out)

    def test_wrapper_forwards_mode_without_overwriting_model(self):
        for mode in ('replay', 'fresh'):
            args, rest = job_parser().parse_known_args([
                '--out-dir', '/tmp/test-run', '--mode', mode, '--max-papers', '2'])
            self.assertEqual(args.model, 'path/to/models/gemma3_27b_it')
            self.assertEqual(rest, ['--mode', mode, '--max-papers', '2'])

    def test_wrapper_explicit_model_and_mode_are_independent(self):
        args, rest = job_parser().parse_known_args([
            '--model', '/models/gemma', '--mode', 'replay', '--out-dir', '/tmp/test-run'])
        self.assertEqual(args.model, '/models/gemma')
        self.assertEqual(rest, ['--mode', 'replay'])

    def test_json_boolean_is_strict(self):
        for v in ['false', 0, 1, None]:
            with self.assertRaises(ValueError): alignment_json(json.dumps(dict(AUDIT, conclusion=v)))
        self.assertFalse(alignment_json(json.dumps(AUDIT))['conclusion'])

    def test_review_validation(self):
        for v in [True, 0, 11, 8.0, '8']:
            with self.assertRaises(ValueError): review_json(json.dumps(dict(REVIEW, rating=v)))
        self.assertEqual(review_json('```json\n'+json.dumps(REVIEW)+'\n```'), REVIEW)

    def test_tail_coverage(self):
        for length in [1, 510, 511, 1000, 4097]:
            covered = set()
            for s in window_starts(length, 510, 255): covered.update(range(s, min(length, s+510)))
            self.assertEqual(len(covered), length)

    def test_replay_block_does_not_restore_rating(self):
        result = process(row(), FixtureFirewall([True]))
        self.assertEqual(result['pred'], 1)
        self.assertTrue(np.isnan(result['r_after']))
        self.assertEqual(result['calls'], 1)
        self.assertEqual(result['final_allowed'], 0)

    def test_no_oracle_in_prompts(self):
        r = row(); r['historical_clean_weaknesses'] = 'SECRET_CLEAN_REFERENCE'
        fw = FixtureFirewall([False]); process(r, fw)
        self.assertNotIn('SECRET_CLEAN_REFERENCE', fw.prompts[0])
        self.assertNotIn('A5_injection', fw.prompts[0])

    def test_retry_must_pass_a_second_audit(self):
        good = process(row(), FixtureFirewall([True, False]), mode='fresh', repair=True)
        self.assertEqual((good['pred'], good['r_after'], good['calls']), (1, 5, 4))
        bad = process(row(), FixtureFirewall([True, True]), mode='fresh', repair=True)
        self.assertTrue(np.isnan(bad['r_after']))
        self.assertEqual(bad['status'], 'repair_blocked')

    def test_error_is_not_detection(self):
        result = process(row(), FixtureFirewall([], fail=True))
        self.assertTrue(np.isnan(result['pred']))
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['escalated'], 1)
        self.assertEqual(result['calls'], 1)

    def test_loader_ids_labels_missing_asr(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'wide.csv'
            pd.DataFrame([{'id':'001','input_text_cleaned':'Paper\nLimitations\nFew cases.',
                'insert__A0_clean':'filler','insert__A5_injection':'payload',
                'rating__A0_clean':7, 'weaknesses__A0_clean':'filler review'}]).to_csv(file,index=False)
            args = argparse.Namespace(wide_csv=str(file), detail_csv='', doc_col='',
                                       max_papers=0, conditions='', mode='replay')
            rows = load_rows(args)
            self.assertEqual([r['paper_id'] for r in rows], ['001']*3)
            self.assertEqual([r['label'] for r in rows], [0,0,1])
            self.assertTrue(np.isnan(rows[0]['historical_clean_rating']))
            self.assertTrue(np.isnan(rows[2]['historical_asr']))
            self.assertIn('payload', rows[2]['text'])

    def test_context_explicit_truncation(self):
        dummy = argparse.Namespace(max_len=5000, count=lambda s,u: len(s+u)//4)
        fw = Firewall(dummy)
        with self.assertRaises(ValueError): fw.fit_document('X'*40000, 'error')
        text, frac = fw.fit_document('A'*20000+'Z'*20000, 'head-tail')
        self.assertTrue(0 < frac < 1)
        self.assertTrue(text.startswith('A') and text.endswith('Z'))

    def test_http_accounting_auth_and_raw_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = CSVJournal(Path(tmp)/'calls.csv', CALL_FIELDS)
            client = LocalClient('http://localhost/v1', 'model', journal, FakeTokenizer())
            seen = []
            def respond(request):
                seen.append(request)
                return httpx.Response(200, json={'choices':[{'message':{'content':json.dumps(AUDIT)},'finish_reason':'stop'}]})
            client.c.close(); client.c = httpx.Client(transport=httpx.MockTransport(respond))
            self.assertEqual(client.ask('sys','input',alignment_json,'001','audit'), AUDIT)
            journal.close(); client.c.close()
            with (Path(tmp)/'calls.csv').open() as f:
                calls = list(csv.DictReader(f))
            self.assertEqual(len(calls), client.calls)
            self.assertEqual(json.loads(calls[0]['raw']), AUDIT)
            self.assertEqual(json.loads(seen[0].content)['temperature'], 0)

    def test_http_failure_keeps_raw_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = CSVJournal(Path(tmp)/'calls.csv', CALL_FIELDS)
            client = LocalClient('http://localhost/v1','model',journal,FakeTokenizer())
            client.c.close()
            client.c = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401)))
            with self.assertRaises(RuntimeError): client.ask('s','u',alignment_json,'k','a')
            self.assertEqual(client.calls,1)
            journal.close(); client.c.close()

    def test_report_confusion_harm_and_matched_clean(self):
        records = []
        for condition, label, detected in [('CLEAN',0,False), ('A0_clean',0,True),
                                           ('A5_injection',1,True), ('A1_omission',1,False)]:
            r = row(); r.update(condition=condition, label=label,
                               family={'CLEAN':'clean','A0_clean':'control','A5_injection':'instruction','A1_omission':'rhetorical'}[condition])
            records.append(process(r, FixtureFirewall([detected]), mode='fresh'))
        with tempfile.TemporaryDirectory() as tmp:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                result = report(records, Path(tmp), 'fresh')
            m = pd.read_csv(Path(tmp)/'metrics.csv').iloc[0]
            self.assertEqual((m.TP,m.FP,m.TN,m.FN),(1,1,1,1))
            self.assertEqual((m.precision,m.recall,m.f1,m.accuracy,m.FPR,m.harmrec),(.5,)*6)
            self.assertEqual((m.calls,m.calls_total),(2,8))
            self.assertEqual(m.rating_pairs,1)
            self.assertTrue(result.r_clean_matched.eq(8).all())
            self.assertEqual(json.loads((Path(tmp)/'metrics.json').read_text())['ALL']['fpr'], .5)

    def test_full_runner_with_mock_endpoint(self):
        # Exercises CLI -> wide reconstruction -> HTTP calls -> final artifacts.
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'wide.csv'; output = Path(tmp)/'run'
            output.mkdir()
            (output/'gpu_memory.csv').write_text('timestamp,memory.used\n')
            pd.DataFrame([{'id':'001','input_text_cleaned':'A manuscript with few cases.',
                'insert__A0_clean':'Neutral filler.', 'insert__A5_injection':'ATTACK_MARKER',
                'asr__A5_injection':'1'}]).to_csv(file,index=False)
            def respond(request):
                if request.method == 'GET': return httpx.Response(200,json={'data':[{'id':'model'}]})
                body = json.loads(request.content)
                if 'Role.ASSISTANT' in body['messages'][1]['content']:
                    value = dict(AUDIT, conclusion='ATTACK_MARKER' in body['messages'][1]['content'])
                else: value = REVIEW
                return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(value)}, 'finish_reason':'stop'}]})
            fake_module = types.ModuleType('transformers')
            fake_module.AutoTokenizer = argparse.Namespace(from_pretrained=lambda *a,**k: FakeTokenizer())
            real_client = httpx.Client
            def factory(*args, **kwargs):
                return real_client(*args, **kwargs, transport=httpx.MockTransport(respond))
            args = ['run_baseline.py','--mode','fresh','--model','model','--wide-csv',str(file),
                    '--detail-csv','','--out-dir',str(output)]
            with patch.dict('sys.modules', {'transformers':fake_module}), \
                 patch('sys.argv',args), patch('httpx.Client',factory), \
                 warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                self.assertEqual(main(),0)
            final = pd.read_csv(output/'rows_final.csv')
            self.assertEqual(final.pred.tolist(),[0,0,1])
            self.assertEqual(len(pd.read_csv(output/'intermediate_calls.csv')),6)
            self.assertTrue((output/'output.txt').exists())
            self.assertTrue((output/'manifest.json').exists())

if __name__ == '__main__': unittest.main(verbosity=2)
