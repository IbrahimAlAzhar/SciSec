#!/usr/bin/env python3
"""Compare original V4 and the SFT/DPO runs only when ordered corpus identities match."""
import argparse
from pathlib import Path
import pandas as pd
from category_defense_metrics import ABLATIONS, DISPLAY, calculate, load_pairs

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--v4-detail',required=True)
p.add_argument('--sft-runs',required=True)
p.add_argument('--out',default='matched_comparison.csv')
a=p.parse_args()
paths=[('Original V4 agents, matched inputs',Path(a.v4_detail))]
paths += [(DISPLAY[ab],Path(a.sft_runs)/ab/'stage1_detail.csv') for ab in ABLATIONS]
rows=[]; reference=None
for label,path in paths:
    if not path.is_file(): raise SystemExit(f'Missing results: {path}')
    d=load_pairs(path)
    identity=list(d[['paper_id','condition','label']].itertuples(index=False,name=None))
    if reference is None: reference=identity
    elif identity!=reference: raise SystemExit(f'Corpus identities/order differ: {path}')
    rows.append(dict(Ablation=label,**calculate(d),Source=str(path)))
result=pd.DataFrame(rows)
cols=['Ablation','N','Acc','Prec','Rec','F1','FPR','HarmRec','Esc%','Calls']
print(result[cols].to_string(index=False,float_format=lambda x:f'{x:.3f}',na_rep='--'))
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
result.to_csv(a.out,index=False)
print(f'[saved] {a.out}')
print('Ordered identity/label parity checked. Historical result CSVs do not contain document text.')
