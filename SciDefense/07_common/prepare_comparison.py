#!/usr/bin/env python3
"""Prepare V4 input exactly as the SFT/DPO all-paper loader and verify parity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import comparison_input as I


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', default='.')
    p.add_argument('--benign-csv', required=True)
    p.add_argument('--attacked-csv', required=True)
    p.add_argument('--reference-detail', required=True)
    p.add_argument('--sft-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--id-column', default='id')
    p.add_argument('--max-papers', type=int, default=1000000)
    p.add_argument('--seed', type=int, default=0)
    a=p.parse_args()
    a.text_column='input_text_cleaned'; a.sections_column='pdf_text'
    a.author_lim_column='author_men_lim'; a.title_column='title'
    a.slot='limitations_end'; a.max_chars=30000
    os.environ['AD_ROOT']=a.project_root
    I.setup_paths(a.project_root)
    import def_dataset as DS
    records=I.build(a, DS)
    reference=I.check_reference(records, a.reference_detail)
    # Independent fresh process uses the existing SFT/DPO loader and original V4 DS.
    script='''import json, sys
sys.path.insert(0, sys.argv[1])
import infer_ladder as L
from dataclasses import asdict
r=L.load_eval_set(sys.argv[2], sys.argv[3], id_column=sys.argv[4], max_papers=int(sys.argv[5]), seed=int(sys.argv[6]), all_papers=True)
sys.path.insert(0, sys.argv[7])
from comparison_input import record_hash
print("PARITY_HASH="+record_hash(r))
'''
    env=os.environ.copy()
    env['V4_DIR']=str(Path(a.project_root)/'defense/llm_agents_v4')
    env['PYTHONPATH']=os.pathsep.join(sys.path)
    child=subprocess.run([sys.executable,'-c',script,str(Path(a.sft_dir).resolve()),
                          a.benign_csv,a.attacked_csv,a.id_column,str(a.max_papers),str(a.seed),
                          str(Path(__file__).resolve().parent)],env=env,text=True,capture_output=True)
    print(child.stdout, end='')
    if child.returncode:
        raise SystemExit('SFT loader parity check failed:\n'+child.stderr)
    hashes=[l.split('=',1)[1] for l in child.stdout.splitlines() if l.startswith('PARITY_HASH=')]
    if hashes != [I.record_hash(records)]:
        raise SystemExit('V5 and the current SFT loader produced DIFFERENT records; stopped')
    source_files={k:{'path':str(Path(v).resolve()),'sha256':I.sha256(v)} for k,v in
                  [('benign_csv',a.benign_csv),('metrics_detail',a.attacked_csv),
                   ('def_dataset',DS.__file__),('sft_infer_ladder',str(Path(a.sft_dir)/'infer_ladder.py'))]}
    for name in ('align','runner','anchors'):
        mod=sys.modules.get(name)
        if mod and getattr(mod,'__file__',None):
            source_files[name]={'path':mod.__file__,'sha256':I.sha256(mod.__file__)}
    report=I.save_records(records,a.out_dir,source_files,reference,vars(a))
    report['current_sft_loader_exact_record_parity']=True
    (Path(a.out_dir)/'eval_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('[ok] exact record/text parity with current SFT/DPO loader; historical result order matches')


if __name__=='__main__': main()
