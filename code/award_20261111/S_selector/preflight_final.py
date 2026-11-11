"""Real CPU preflight: frozen inputs, feature parity, model parity and a full dry run. Writes no candidate."""
import glob
import json
import os
import resource
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pandas as pd

import final_logic as fl
import verify_final
from metric import PANELS2, exact_score
from npzmap import array
from rawpanel import read_layer
from tfb_common import REL

HERE = Path(__file__).resolve().parent
CONTROL = Path('/mnt/datos/competition-campaigns/traffic-umud-20261105/spatial-potts-v1/config/control_freeze.json')
FIRST_DAY = pd.Timestamp('2030-06-01')
TRAIN_LAST = 227
REPORT_FIRST = 243


def links_of(name):
    return [str(x) for x in json.loads(CONTROL.read_text())['topology_mapping'][name]['observed_links']]


def frozen_files(config):
    frozen = config['frozen_result']
    for key in ('choices', 'probabilities', 'matched'):
        assert fl.digest(frozen[key + '_path']) == frozen[key + '_sha256'], key
    assert fl.digest(config['base']['path']) == config['base']['sha256']
    choices = json.loads(Path(frozen['choices_path']).read_text())['choices']
    for name, arm in config['arms'].items():
        assert choices[name]['ensemble_threshold'] == arm['threshold'], name
    return True


def model_parity(config, arms):
    """The closing path reproduces the saved probabilities and the audited scores."""
    frozen = config['frozen_result']
    path = frozen['matched_path']
    day = np.asarray(array(path, 'day'))
    rows = np.flatnonzero(day > TRAIN_LAST)
    x = np.asarray(array(path, 'x_t')[rows])
    late = day[rows] >= REPORT_FIRST
    meta = {k: np.asarray(array(path, k)[rows])[late] for k in ('y', 'mask', 'group', 'panel', 'cond')}
    assert day.max() == 257
    out = {}
    with np.load(frozen['probabilities_path']) as saved:
        for name, (models, threshold) in arms.items():
            p = fl.ensemble(models, x)
            kept = saved[name].mean(axis=0)
            changed = int(((p >= threshold) != (kept >= threshold)).sum())
            overall = exact_score(p[late] >= threshold, meta['y'], meta['mask'], meta['group'], meta['panel'], meta['cond'])['overall']
            difference = float(np.abs(p - kept).max())
            assert difference <= 1e-5 and changed == 0, (name, difference, changed)
            assert abs(overall - frozen['ensemble_scores'][name]) <= 1e-12, (name, overall)
            out[name] = dict(max_probability_difference=difference, decisions_changed=changed, report_score=overall)
    return out


def train_feature_parity():
    """Public train windows through the closing path against the audited x_t rows. Open dates only."""
    path = json.loads((HERE / 'config/final.json').read_text())['frozen_result']['matched_path']
    panel_id = np.asarray(array(path, 'panel'))
    day = np.asarray(array(path, 'day'))
    origin = np.asarray(array(path, 'origin'))
    x_t = array(path, 'x_t')
    report = dict(windows=0, identical_to_reference=0, found_in_audited_rows=0, identical_to_audited_rows=0, max_difference=0.)
    for i, name in enumerate(PANELS2):
        root = REL / 'task2' / name / 'train'
        windows = pd.read_csv(root / 'window_index.csv', dtype=fl.TEXT)
        history = pd.read_parquet(root / 'window_history.parquet')
        history['window_id'] = history.window_id.astype(str)
        dates = sorted(windows.date.unique())
        links = links_of(name)
        base = REL / 'corridors' / name / 'train'
        masked, _ = read_layer(sorted(glob.glob(str(base / 'mainline_states_masked/mask_regime=*/*.parquet'))), dates, links, ['speed_kmh', 'flow_vph'])
        truth, _ = read_layer(sorted(glob.glob(str(base / 'mainline_states/*/*.parquet'))), dates, links, ['speed_kmh', 'flow_vph'])
        meta = pd.read_csv(REL / 'corridors' / name / 'network/links.csv').set_index('link_id')
        meta.index = meta.index.astype(str)
        P = dict(links=links, s=masked['speed_kmh'], q=masked['flow_vph'], vf=meta.loc[links, 'free_speed_kmh'].to_numpy(dtype='float32'),
                 lanes=meta.loc[links, 'lanes'].to_numpy(dtype='float32'), dow=pd.to_datetime(pd.Series(dates)).dt.dayofweek.to_numpy())
        where = {v: k for k, v in enumerate(dates)}
        for w in windows.itertuples():
            o = fl.window_origin(w, name, 'train')
            d = where[w.date]
            t0 = fl.tick(o)
            closing = fl.feature_matrix(fl.overlay_history(P, d, o, history[history.window_id == w.window_id]), d, t0)
            s = P['s'][d].copy()
            q = P['q'][d].copy()
            s[:, t0 - 12:t0] = truth['speed_kmh'][d][:, t0 - 12:t0]
            q[:, t0 - 12:t0] = truth['flow_vph'][d][:, t0 - 12:t0]
            reference = fl.feature_matrix(dict(P, s={d: s}, q={d: q}), d, t0)
            report['windows'] += 1
            report['identical_to_reference'] += int(np.array_equal(closing, reference))
            report['max_difference'] = max(report['max_difference'], float(np.abs(closing - reference).max()))
            global_day = (pd.Timestamp(w.date) - FIRST_DAY).days
            rows = np.flatnonzero((panel_id == i) & (day == global_day) & (origin == t0))
            if len(rows):
                report['found_in_audited_rows'] += 1
                report['identical_to_audited_rows'] += int(np.array_equal(closing, np.asarray(x_t[rows[:len(links)]])))
    assert report['identical_to_reference'] == report['windows'] == 80
    assert report['found_in_audited_rows'] > 0 and report['identical_to_audited_rows'] == report['found_in_audited_rows']
    return report


def link_identity():
    out = {}
    for split in fl.SPLITS:
        for name in PANELS2:
            P = fl.load_release(name, split)
            assert [str(x) for x in P['links']] == links_of(name), (name, split)
            template = pd.read_csv(REL / 'task2' / name / split / 'sample_submission_queue.csv', dtype=fl.TEXT, usecols=['link_id'])
            assert set(template.link_id) == set(links_of(name)), (name, split)
            out[f'{name}:{split}'] = len(P['links'])
    return out


def real_causality():
    name = PANELS2[0]
    P = fl.load_release(name, 'validation')
    root = REL / 'task2' / name / 'validation'
    w = pd.read_csv(root / 'window_index.csv', dtype=fl.TEXT).iloc[0]
    history = pd.read_parquet(root / 'window_history.parquet')
    history['window_id'] = history.window_id.astype(str)
    o = fl.window_origin(w, name, 'validation')
    d = P['days'].index(w.date)
    view = fl.overlay_history(P, d, o, history[history.window_id == w.window_id])
    t0 = fl.tick(o)
    before = fl.feature_matrix(view, d, t0)
    view['s'][d][:, t0 + 1:] = 1.
    view['q'][d][:, t0 + 1:] = 1.
    assert np.array_equal(before, fl.feature_matrix(view, d, t0))
    return True


def rehearse(series, config):
    """Real write, zip and byte comparison on a temporary file, removed afterwards."""
    temporary = HERE / 'preflight-rehearsal.csv'
    packed = temporary.with_suffix('.zip')
    began = time.time()
    try:
        fl.combine(series, temporary, config['base']['path'], config['base']['sha256'], REL / 'submission_key.csv')
        fl.archive(temporary)
        cells, equal, rows = verify_final.candidate_cells(temporary, config['base']['path'], REL / 'submission_key.csv')
        assert equal and rows == verify_final.TOTAL_ROWS and len(cells) == verify_final.QUEUE_CELLS
        return dict(rows=rows, other_tasks_identical=equal, csv_gib=temporary.stat().st_size / 2 ** 30, zip_gib=packed.stat().st_size / 2 ** 30,
                    seconds=round(time.time() - began))
    finally:
        temporary.unlink(missing_ok=True)
        packed.unlink(missing_ok=True)


def main():
    started = time.time()
    config = json.loads((HERE / 'config/final.json').read_text())
    assert not config['confirmation_opened'] and not config['portal_submission']
    result = dict(time=datetime.now(timezone.utc).isoformat(), cuda_used=False, candidate_kept=False, confirmation_opened=False,
                  frozen_files=frozen_files(config))
    arms = {name: (fl.load_arm(arm), arm['threshold']) for name, arm in config['arms'].items()}
    result['model_parity'] = model_parity(config, arms)
    print(json.dumps(result['model_parity']), flush=True)
    result['train_feature_parity'] = train_feature_parity()
    print(json.dumps(result['train_feature_parity']), flush=True)
    result['link_identity'] = link_identity()
    result['causality_on_real_window'] = real_causality()
    predictions, summary = fl.build_predictions(arms)
    result['dry_run'] = fl.describe(summary)
    result['key_coverage'] = {name: fl.combine(series, HERE / 'unused.csv', config['base']['path'], config['base']['sha256'],
                                               REL / 'submission_key.csv', write=False) for name, series in predictions.items()}
    result['write_rehearsal'] = rehearse(predictions['S'], config)
    result['resources'] = dict(seconds=round(time.time() - started), peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                               free_disk_gib=shutil.disk_usage('/home/ai/servidor-ia/jobs').free / 2 ** 30)
    assert result['resources']['peak_rss_gib'] < 8 and result['resources']['free_disk_gib'] > 5
    result['status'] = 'FINAL_CPU_PREFLIGHT_PASSED'
    fl.write_json(HERE / 'final-preflight.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'dry_run', 'key_coverage')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
