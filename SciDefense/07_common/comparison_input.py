"""Input-only bridge to the SFT/DPO all-paper insertion evaluation.

No defense prompts, thresholds, or decision logic live here. Repeated paper IDs are
preserved as separate ordered occurrences, matching the original builder.
"""
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def setup_paths(root):
    root = Path(os.environ.get('AD_ROOT', root))
    for env, default in (
        ('WB_FEAS_DIR', 'lim_white_black_feas_dis_search_no_seed'),
        ('WB_COMP_DIR', 'lim_composite_attack'),
        ('WB_SRC_DIR', 'lim_white_black_transf'),
        ('WB_BB_DIR', 'lim_bb_multimodel_75_rows'),
        ('WB_T35_DIR', 'lim_track3_5_mech'),
        ('WB_UCA_DIR', 'lim_uca_hybrid'),
    ):
        candidate = Path(os.environ.get(env) or root / default)
        if candidate.is_dir():
            os.environ.setdefault(env, str(candidate))
            if str(candidate) not in sys.path:
                sys.path.append(str(candidate))
    if (root / 'whitebox').is_dir() and str(root / 'whitebox') not in sys.path:
        sys.path.append(str(root / 'whitebox'))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def plain(x):
    if isinstance(x, dict): return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [plain(v) for v in x]
    if hasattr(x, 'item'): x = x.item()
    if isinstance(x, float) and not __import__('math').isfinite(x): return None
    return x


def record_hash(records):
    content = json.dumps([plain(asdict(r)) for r in records], ensure_ascii=False,
                         sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def validate(records):
    if not records or len(records) % 2:
        raise ValueError('expected a nonempty, complete paired corpus')
    for i in range(0, len(records), 2):
        clean, attack = records[i:i+2]
        if (clean.label != 0 or attack.label != 1 or clean.condition != 'A0_clean'
                or attack.condition == 'A0_clean' or clean.paper_id != attack.paper_id
                or clean.row_index != attack.row_index):
            raise ValueError(f'invalid clean/attacked pair at record {i}')
        if not clean.document or not attack.document:
            raise ValueError(f'empty document at record {i}')


def build(args, DS):
    if args.max_papers < 1 or args.max_chars < 1:
        raise ValueError('max_papers and max_chars must be positive')
    if not args.attacked_csv:
        raise ValueError('--attacked-csv must point to the SFT/DPO metrics_detail.csv')
    attacked = DS.load_attacked(args.attacked_csv)
    if attacked.empty:
        raise ValueError('no usable attack inserts')
    keep = sorted({str(p) for p in attacked['paper_id']})
    # Exact all_papers=True branch of infer_ladder.load_eval_set; no split/subsampling.
    records = DS.build_eval_set(args.benign_csv, attacked, keep, args,
                                max_papers=args.max_papers, seed=args.seed)
    validate(records)
    return records


def check_reference(records, path):
    """Historical CSVs have no document text: verify their ordered identities/labels."""
    import pandas as pd
    ref = pd.read_csv(path, dtype={'paper_id': str, 'doc_id': str, 'condition': str})
    required = ['doc_id', 'paper_id', 'condition', 'label']
    if any(c not in ref for c in required):
        raise ValueError(f'reference must have {required}')
    if len(ref) != len(records):
        raise ValueError(f'corpus mismatch: {len(records)} records vs {len(ref)} reference rows')
    for i, (r, (_, old)) in enumerate(zip(records, ref.iterrows())):
        expected = (str(r.doc_id), str(r.paper_id), str(r.condition), int(r.label))
        actual = (str(old.doc_id), str(old.paper_id), str(old.condition), int(old.label))
        if actual != expected:
            raise ValueError(f'reference mismatch at data row {i+1}: {actual} != {expected}')
        if 'harmful' in ref and int(old.harmful) != int(bool(r.asr)):
            raise ValueError(f'harmful label mismatch at data row {i+1}')
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path),
            'rows': len(ref), 'matched': True,
            'scope': 'ordered doc_id/paper_id/condition/label and harmful if present; historical text unavailable'}


def load_records(path, DS):
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip(): records.append(DS.DocRecord(**json.loads(line)))
    validate(records)
    manifest = Path(path).parent / 'eval_manifest.json'
    if manifest.exists():
        info = json.loads(manifest.read_text())
        if info.get('records_hash') != record_hash(records):
            raise ValueError('frozen records do not match eval_manifest.json')
    return records


def save_records(records, out_dir, sources=None, reference=None, settings=None):
    from collections import Counter
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / 'eval_records.jsonl'
    with path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(plain(asdict(r)), ensure_ascii=False, allow_nan=False)+'\n')
    with (out/'defense_inputs.jsonl').open('w', encoding='utf-8') as f:
        for i, r in enumerate(records):
            f.write(json.dumps({'eval_index': i, 'document': r.document}, ensure_ascii=False)+'\n')
    counts = Counter(str(r.paper_id) for r in records if r.label == 1)
    report = dict(records_hash=record_hash(records), n_documents=len(records),
                  n_clean=sum(r.label == 0 for r in records), n_attacked=sum(r.label == 1 for r in records),
                  unique_paper_ids=len(counts), repeated_paper_ids={k:v for k,v in counts.items() if v>1},
                  conditions=dict(Counter(r.condition for r in records if r.label == 1)),
                  source_files=sources or {}, reference=reference, settings=settings or {},
                  full_insert_missing=[i for i,r in enumerate(records) if r.label == 1 and
                                       r.insert.strip() not in r.document])
    (out/'eval_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'[input] {report["n_clean"]} clean + {report["n_attacked"]} attacked; '
          f'{len(counts)} unique IDs; record hash {report["records_hash"]}')
    if report['repeated_paper_ids']:
        print('[input] repeated IDs preserved by row occurrence; see eval_manifest.json')
    return report


def load_for_runner(args, DS):
    if args.threat != 'insertion' or any((args.baseline, args.btr_dir, args.btr_results,
                                        args.argar_dir, args.conditions)):
        raise ValueError('sft-dpo input mode requires insertion and no alternate datasets/conditions')
    if args.eval_records:
        records = load_records(args.eval_records, DS)
        sources = {'prepared_records': {'path': str(Path(args.eval_records).resolve()),
                                        'sha256': sha256(args.eval_records)}}
    else:
        records = build(args, DS)
        sources = {key: {'path': str(Path(path).resolve()), 'sha256': sha256(path)}
                   for key, path in [('benign_csv', args.benign_csv), ('attacked_csv', args.attacked_csv)]}
    reference = check_reference(records, args.reference_detail) if args.reference_detail else None
    args.input_report = save_records(records, args.out_dir, sources, reference,
                                    {'input_mode': 'sft-dpo', 'selection': 'all matching paper occurrences'})
    return records
