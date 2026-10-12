#!/usr/bin/env python3
"""Per-category metrics for ORIGINAL, UNSORTED infer_ladder stage1_detail.csv files.

Each category includes its attacked rows AND their immediately preceding clean rows.
This follows def_dataset.build_eval_set and avoids ambiguous joins on repeated paper IDs.
Requires pandas; no model calls or GPU. Metrics are document-level, not deduplicated.
"""
import argparse
from pathlib import Path
import re
import pandas as pd

ABLATIONS = ['zeroshot', 'dpo_only', 'sft', 'sft_dpo', 'sft_dpo_per_agent']
DISPLAY = {
    'zeroshot': 'Zero-shot cascade (no adapter)',
    'dpo_only': 'DPO only, no SFT',
    'sft': 'SFT, single adapter',
    'sft_dpo': 'SFT then DPO, single adapter',
    'sft_dpo_per_agent': 'SFT+DPO per agent (5 adapters)',
}


def divide(a, b):
    return a / b if b else float('nan')


def calculate(d):
    y, p = d['label'], d['pred']
    tp = int(((y == 1) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    result = dict(N=len(d), Positive=tp+fn, Negative=tn+fp,
                  TP=tp, FN=fn, TN=tn, FP=fp,
                  Acc=divide(tp+tn, len(d)), Prec=divide(tp, tp+fp),
                  Rec=divide(tp, tp+fn), F1=divide(2*tp, 2*tp+fp+fn),
                  FPR=divide(fp, fp+tn), Specificity=divide(tn, tn+fp),
                  FNR=divide(fn, tp+fn))
    harmful_col = 'harmful' if 'harmful' in d else ('asr' if 'asr' in d else None)
    if harmful_col:
        h = pd.to_numeric(d[harmful_col], errors='coerce')
        if not h[y == 1].isin([0, 1]).all():
            raise ValueError('harmful/asr must be 0 or 1 for every attacked row')
        harmful = (y == 1) & (h == 1)
        result['HarmN'] = int(harmful.sum())
        result['HarmRec'] = divide(int((harmful & (p == 1)).sum()), int(harmful.sum()))
    else:
        result.update(HarmN=float('nan'), HarmRec=float('nan'))
    for source, target, scale in [('escalated', 'Esc%', 100), ('n_llm_calls', 'Calls', 1)]:
        if source not in d:
            result[target] = float('nan')
            continue
        values = pd.to_numeric(d[source], errors='coerce')
        if values.isna().any() or (values < 0).any():
            raise ValueError(f'invalid {source} values')
        if source == 'escalated' and not values.isin([0, 1]).all():
            raise ValueError('escalated must be 0 or 1')
        result[target] = scale * values.mean()
    return result


def load_pairs(path):
    d = pd.read_csv(path, dtype={'paper_id': str, 'doc_id': str}).reset_index(drop=True)
    required = {'paper_id', 'condition', 'label', 'pred'}
    if required - set(d):
        raise ValueError(f'missing columns: {sorted(required - set(d))}')
    if d.empty or len(d) % 2:
        raise ValueError('expected a nonempty, even-length file of complete pairs')
    for c in ('label', 'pred'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
        if not d[c].isin([0, 1]).all():
            raise ValueError(f'missing/invalid {c}; expected 0 or 1')
        d[c] = d[c].astype(int)
    if d[['paper_id', 'condition']].isna().any().any():
        raise ValueError('missing paper IDs or conditions')
    categories = []
    for i in range(0, len(d), 2):
        clean, attack = d.iloc[i], d.iloc[i+1]
        if (clean['label'] != 0 or attack['label'] != 1 or
                clean['condition'] != 'A0_clean' or attack['condition'] == 'A0_clean' or
                clean['paper_id'] != attack['paper_id']):
            raise ValueError(f'rows {i+1}/{i+2} are not a clean/attack pair. '
                             'Use the original unsorted infer_ladder CSV; do not join repeated IDs.')
        if 'row_index' in d and clean['row_index'] != attack['row_index']:
            raise ValueError(f'row_index mismatch at rows {i+1}/{i+2}')
        categories.extend([attack['condition']] * 2)
    d['_category'] = categories
    d['_csv_record'] = range(1, len(d)+1)  # Data records, not physical lines in quoted CSV.
    return d


def category_key(name):
    m = re.match(r'A(\d+)', str(name))
    return (int(m.group(1)) if m else 1000000, str(name))


def latex_escape(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&').replace('#', r'\#')


def latex_table(frame):
    cols = ['Ablation', 'Category', 'N', 'Acc', 'Prec', 'Rec', 'F1', 'FPR', 'HarmRec', 'Esc%', 'Calls']
    lines = [r'\begin{tabular}{llrrrrrrrrr}', r'\toprule',
             ' & '.join(latex_escape(c) for c in cols) + r' \\', r'\midrule']
    for _, r in frame.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in ('Ablation', 'Category'): cells.append(latex_escape(v))
            elif pd.isna(v): cells.append('--')
            elif c == 'N': cells.append(str(int(v)))
            else: cells.append(f'{v:.1f}' if c == 'Esc%' else f'{v:.2f}' if c == 'Calls' else f'{v:.3f}')
        lines.append(' & '.join(cells) + r' \\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}', ''])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('files', nargs='*', help='One or more stage1_detail.csv paths')
    p.add_argument('--runs-dir', help='runs/gemma3-27b-it directory with ablation subdirectories')
    p.add_argument('--out-dir', default='category_metrics')
    a = p.parse_args()
    paths = [Path(x) for x in a.files]
    if a.runs_dir:
        for ab in ABLATIONS:
            path = Path(a.runs_dir) / ab / 'stage1_detail.csv'
            if path.is_file(): paths.append(path)
            else: print(f'[missing] {path}')
    paths = list(dict.fromkeys(x.resolve() for x in paths))
    if not paths: p.error('provide --runs-dir or stage1_detail.csv paths')
    results, audits, reference, sections = [], [], None, []
    for path in paths:
        try: d = load_pairs(path)
        except (ValueError, OSError) as e: raise SystemExit(f'{path}: {e}')
        keys = ['paper_id', 'condition', 'label']
        signature = list(d[keys].itertuples(index=False, name=None))
        if reference is None: reference = signature
        elif signature != reference:
            raise SystemExit('Ablations have different ordered paper/condition/label sequences. '
                             'Do not compare them as the same evaluation corpus.')
        ab = path.parent.name
        attacked = d[d['label'] == 1]
        counts = attacked['paper_id'].value_counts()
        repeat_ids = counts[counts > 1]
        audit = d[d['paper_id'].isin(repeat_ids.index)].copy()
        audit.insert(0, 'source_file', str(path))
        audit.insert(0, 'ablation', ab)
        audit_cols = ['ablation', 'source_file', '_csv_record', 'paper_id', 'condition', 'label', 'pred']
        if 'doc_id' in audit: audit_cols.append('doc_id')
        audits.append(audit[audit_cols])
        print(f'[audit] {ab}: {len(attacked)} pairs, {attacked.paper_id.nunique()} distinct IDs; '
              f'{len(repeat_ids)} repeated IDs, {int((repeat_ids-1).sum())} extra pair occurrences')
        groups = [('ALL', d)] + [(c, d[d['_category'] == c]) for c in sorted(d['_category'].unique(), key=category_key)]
        current = []
        for cat, group in groups:
            try: metrics = calculate(group)
            except ValueError as e: raise SystemExit(f'{path}: {e}')
            row = dict(Ablation=ab, Category=cat, **metrics, Source=str(path))
            results.append(row); current.append(row)
        # Category confusion matrices partition ALL, with each clean row used once.
        for c in ['N', 'TP', 'TN', 'FP', 'FN']:
            assert sum(r[c] for r in current[1:]) == current[0][c]
        cols = ['Category', 'N', 'Positive', 'Negative', 'Acc', 'Prec', 'Rec', 'F1', 'FPR', 'HarmN', 'HarmRec', 'Esc%', 'Calls']
        text = '\n' + DISPLAY.get(ab, ab) + '\n' + pd.DataFrame(current)[cols].to_string(index=False, float_format=lambda x: f'{x:.3f}', na_rep='--')
        print(text); sections.append(text)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(out/'category_metrics.csv', index=False)
    pd.concat(audits, ignore_index=True).to_csv(out/'repeated_paper_ids.csv', index=False)
    (out/'category_metrics.txt').write_text('\n'.join(sections)+'\n', encoding='utf-8')
    (out/'category_metrics.tex').write_text(latex_table(frame), encoding='utf-8')
    print(f'\n[saved] {out.resolve()}')
    print('Category metrics include matched clean rows. HarmRec uses only harmful attacked rows.')
    print('Undefined ratios are --/blank; no predictions or labels were deduplicated.')
    print('Repeated IDs require inspecting the original clean CSV; output IDs alone cannot prove duplicate documents.')


if __name__ == '__main__':
    main()
