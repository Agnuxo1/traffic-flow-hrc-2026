"""CPU regrade of a closing job from its artifacts: recomputes every decision, never submits."""
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pandas as pd

import final_logic as fl
from metric import H, PANELS2, normalised
from tfb_common import REL

JOBS = Path('/home/ai/servidor-ia/jobs')
CAMPAIGN = Path('/mnt/datos/competition-campaigns/traffic-umud-20261105/claude-hypothesis-20261106')
TOTAL_ROWS = 6980503
QUEUE_CELLS = 174000


def numpy_forward(ck, x):
    """Second route to the probabilities, without torch."""
    state = {k: v.detach().numpy().astype('float64') for k, v in ck['state_dict'].items()}
    z = normalised(x, ck['mean'], ck['std']).astype('float64')
    for i in range(ck['cfg']['depth']):
        a = 4 * i
        z = z @ state[f'{a}.weight'].T + state[f'{a}.bias']
        z = (z - z.mean(1, keepdims=True)) / np.sqrt(z.var(1, keepdims=True) + 1e-5) * state[f'{a + 1}.weight'] + state[f'{a + 1}.bias']
        z = z / (1 + np.exp(-z))
    a = 4 * ck['cfg']['depth']
    return 1 / (1 + np.exp(-(z @ state[f'{a}.weight'].T + state[f'{a}.bias'])))


def candidate_cells(path, base, key):
    """Candidate against base, byte by byte; both must end together. Returns the queue cells."""
    cells = []
    other_equal = True
    rows = 0
    with Path(path).open('rb') as new, Path(base).open('rb') as old:
        other_equal &= new.readline() == old.readline()
        for part in fl.queue_keys(key):
            queue = (part.task == 'queue').to_numpy()
            values = np.full(len(part), -1)
            for i in range(len(part)):
                a, b = new.readline(), old.readline()
                assert a and b, 'Candidate or base ended before the key'
                if queue[i]:
                    x, y = fl.fields(a), fl.fields(b)
                    assert x[fl.QUEUE_FIELD] in (b'0', b'1'), 'Invalid queue value'
                    values[i] = int(x[fl.QUEUE_FIELD])
                    del x[fl.QUEUE_FIELD], y[fl.QUEUE_FIELD]
                    other_equal &= x == y and a[len(a.rstrip(b'\r\n')):] == b[len(b.rstrip(b'\r\n')):]
                else:
                    other_equal &= a == b
            rows += len(part)
            if queue.any():
                cells.append(part.loc[queue, fl.KEY_COLUMNS].assign(queue_pred=values[queue]))
        assert new.readline() == b'' and old.readline() == b'', 'Candidate or base has rows beyond the key'
    return pd.concat(cells, ignore_index=True).set_index(fl.KEY_COLUMNS).queue_pred, bool(other_equal), rows


def independent_windows(arms_raw, series):
    """One window per panel and split through numpy; every decision must match the candidate."""
    mismatches = 0
    checked = 0
    for split in fl.SPLITS:
        for panel in PANELS2:
            P = fl.load_release(panel, split)
            root = REL / 'task2' / panel / split
            w = pd.read_csv(root / 'window_index.csv', dtype=fl.TEXT).iloc[0]
            history = pd.read_parquet(root / 'window_history.parquet')
            history['window_id'] = history.window_id.astype(str)
            target = pd.read_csv(root / 'sample_submission_queue.csv', dtype=fl.TEXT)
            target = target[target.window_id == w.window_id]
            origin = fl.window_origin(w, panel, split)
            d = P['days'].index(w.date)
            x = fl.feature_matrix(fl.overlay_history(P, d, origin, history[history.window_id == w.window_id]), d, fl.tick(origin))
            where = {str(v): i for i, v in enumerate(P['links'])}
            step = ((pd.to_datetime(target.timestamp, utc=True) - origin).dt.total_seconds() // 300).astype(int).to_numpy()
            for name, (models, threshold) in arms_raw.items():
                p = np.mean([numpy_forward(ck, x) for ck, _ in models], axis=0)
                got = (p[target.link_id.map(where).to_numpy(), step - 1] >= threshold).astype(int)
                index = pd.MultiIndex.from_arrays([[panel] * len(target), [split] * len(target), target.window_id, target.link_id, target.timestamp], names=fl.KEY_COLUMNS)
                mismatches += int((series[name].reindex(index).to_numpy() != got).sum())
                checked += len(got)
    return checked, mismatches


def main(job):
    work = JOBS / job / 'work'
    results = work / 'results'
    frozen_manifest = json.loads((CAMPAIGN / 'selector-final-v1/source-manifest.json').read_text())
    config = json.loads((CAMPAIGN / 'selector-final-v1/config/final.json').read_text())
    result = json.loads((results / 'result.json').read_text())
    choice = json.loads((results / 'frozen_choice.json').read_text())
    checks = dict(
        base=fl.digest(config['base']['path']) == config['base']['sha256'],
        source=all(fl.digest(work / k) == v for k, v in frozen_manifest.items()),
        config=fl.digest(work / 'config/final.json') == choice['config_sha256'] == fl.digest(CAMPAIGN / 'selector-final-v1/config/final.json'),
        thresholds=choice['thresholds'] == {k: v['threshold'] for k, v in config['arms'].items()},
        flags=(result['status'] == 'PREPARED_FOR_OWNER_REVIEW' and not result['submission_sent'] and not result['promoted']
               and not result['confirmation_opened'] and not result['official_improvement_verified']))
    arms = {name: (fl.load_arm(arm), arm['threshold']) for name, arm in config['arms'].items()}
    recomputed, _ = fl.build_predictions(arms)
    series = {}
    detail = {}
    for name, item in result['artifacts'].items():
        csv = results / f'traffic_candidate_{name}.csv'
        packed = csv.with_suffix('.zip')
        cells, other_equal, rows = candidate_cells(csv, config['base']['path'], REL / 'submission_key.csv')
        series[name] = cells
        with zipfile.ZipFile(packed) as z:
            assert z.namelist() == [csv.name]
            digest = hashlib.sha256()
            with z.open(csv.name) as stream:
                for block in iter(lambda: stream.read(1 << 24), b''):
                    digest.update(block)
        same = recomputed[name].reindex(cells.index)
        detail[name] = dict(rows=rows, queue_rows=len(cells), positive=int(cells.sum()), csv_sha256=fl.digest(csv), zip_sha256=fl.digest(packed),
                            decision_changes=int((same.to_numpy() != cells.to_numpy()).sum()))
        checks[f'{name}_hashes'] = detail[name]['csv_sha256'] == item['csv_sha256'] and detail[name]['zip_sha256'] == item['zip_sha256'] and digest.hexdigest() == item['csv_sha256']
        checks[f'{name}_other_tasks_identical'] = bool(other_equal)
        checks[f'{name}_all_cells'] = rows == TOTAL_ROWS and len(cells) == QUEUE_CELLS and cells.index.is_unique and not same.isna().any()
        checks[f'{name}_decisions_recomputed'] = detail[name]['decision_changes'] == 0
    checked, mismatches = independent_windows(arms, series)
    checks['independent_numpy_route'] = mismatches == 0
    failed = [k for k, v in checks.items() if not v]
    out = dict(status='FINAL_CPU_VERIFIED' if not failed else 'FINAL_CPU_VERIFICATION_FAILED', job_id=job, failed_checks=failed, checks=checks,
               candidates=detail, independent_cells=checked, independent_mismatches=mismatches, predictions=result['predictions'],
               submission_sent=False, promoted=False, confirmation_opened=False, official_score=False)
    (CAMPAIGN / 'verified').mkdir(exist_ok=True)
    fl.write_json(CAMPAIGN / f'verified/final-{job}.json', out)
    print(json.dumps({k: out[k] for k in ('status', 'failed_checks', 'candidates', 'independent_cells', 'independent_mismatches')}))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
