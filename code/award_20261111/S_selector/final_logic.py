"""Frozen closing logic: official history in, reviewable queue decisions out. Never submits."""
import glob
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from metric import H, PANELS2, model_for, normalised
from queue_features import ratio_hist, window_feats
from replica_loader import batches, load_panel_truth
from tfb_common import REL

SPLITS = ('validation', 'private')  # `train` holds the sealed days and is never loaded here
HISTORY_STEPS = 12
KEY_COLUMNS = ['panel', 'split', 'window_id', 'link_id', 'timestamp']
TEXT = {'window_id': str, 'link_id': str, 'timestamp': str, 'panel': str, 'split': str}
COLUMNS = [b'submission_id', b'task', b'speed_kmh', b'flow_vph', b'queue_pred', b'path_flow']
QUEUE_FIELD = COLUMNS.index(b'queue_pred')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def feature_matrix(P, d, t0):
    base, _ = window_feats(P, d, t0, None)
    ratios = ratio_hist(P, d, t0)
    position = np.arange(len(P['links']))
    history = np.concatenate([ratios[np.clip(position + o, 0, len(position) - 1)] for o in [-2, -1, 0, 1, 2]], axis=1)
    return np.nan_to_num(np.concatenate([base, history], axis=1), nan=0., posinf=10., neginf=-10.).astype('float32')


def tick(origin):
    assert origin.utcoffset() == pd.Timedelta(0), 'Origin must be expressed in UTC'
    return origin.hour * 12 + origin.minute // 5


def window_origin(w, panel, split):
    """Origin of an index row, tied to the panel, split and date that the row declares."""
    origin = pd.Timestamp(w.forecast_origin).tz_convert('UTC')
    assert w.panel == panel and w.split == split, 'Window index row from another panel or split'
    assert str(origin.date()) == str(w.date), 'Origin outside its declared date'
    return origin


def on_grid(stamps):
    """Five-minute stamps only: anything else would be rounded into a neighbouring cell."""
    assert ((stamps.dt.minute % 5 == 0) & (stamps.dt.second == 0) & (stamps.dt.microsecond == 0) & (stamps.dt.nanosecond == 0)).all(), 'Timestamp off the five-minute grid'


def load_release(panel, split, rel=REL, loader=load_panel_truth):
    """Masked month of an official split, refused if any stamp is off the grid."""
    assert split in SPLITS, 'Only the official test months are read here'
    files = sorted(glob.glob(str(Path(rel) / 'corridors' / panel / split / 'mainline_states_masked' / 'mask_regime=*' / '*.parquet')))
    assert files, (panel, split, 'missing masked parquet')
    for frame in batches(files, ['date', 'timestamp']):
        stamps = pd.to_datetime(frame.timestamp, utc=True)
        on_grid(stamps)
        assert (frame.date.astype(str).to_numpy() == stamps.dt.strftime('%Y-%m-%d').to_numpy()).all(), 'Row date differs from its timestamp'
    return loader(panel, split, truth=False)


def overlay_history(P, d, origin, history):
    """Official rows at exactly T-60, T-55, ..., T-5 over the released day; anything else is refused."""
    t0 = tick(origin)
    assert t0 >= HISTORY_STEPS, 'Window history would cross midnight'
    on_grid(pd.Series([origin]))
    stamps = pd.to_datetime(history.timestamp, utc=True)
    on_grid(stamps)
    before = (origin - stamps).dt.total_seconds().to_numpy()
    assert np.isin(before, 300 * np.arange(1, HISTORY_STEPS + 1)).all(), 'History outside T-60..T-5'
    ticks = (stamps.dt.hour * 12 + stamps.dt.minute // 5).to_numpy()
    where = {str(v): i for i, v in enumerate(P['links'])}
    rows = history.link_id.astype(str).map(where)
    assert rows.notna().all(), 'History link missing from the released links'
    rows = rows.to_numpy(dtype=int)
    assert len(set(zip(rows.tolist(), ticks.tolist()))) == len(history) == HISTORY_STEPS * len(P['links']), 'Incomplete window history'
    s = P['s'][d].copy()
    q = P['q'][d].copy()
    s[rows, ticks] = history.speed_kmh.to_numpy(dtype='float32')
    q[rows, ticks] = history.flow_vph.to_numpy(dtype='float32')
    return dict(P, s={d: s}, q={d: q})


def load_arm(arm):
    models = []
    for item in arm['models']:
        assert digest(item['path']) == item['sha256'], 'Frozen model changed: ' + item['path']
        # Own hash-checked checkpoints; they carry NumPy normalisers.
        ck = torch.load(item['path'], map_location='cpu', weights_only=False)
        net = model_for(torch, len(ck['mean']), ck['cfg'])
        net.load_state_dict(ck['state_dict'])
        net.eval()
        models.append((ck, net))
    return models


def ensemble(models, x):
    """Mean probability of the three models, as in selector_logic.score_arms."""
    with torch.no_grad():
        return np.stack([net(torch.as_tensor(normalised(x, ck['mean'], ck['std']))).sigmoid().numpy() for ck, net in models]).mean(axis=0)


def window_decisions(view, d, origin, target, arms):
    """Decisions of every arm for one window; the model covers every released link directly."""
    t0 = tick(origin)
    x = feature_matrix(view, d, t0)
    where = {str(v): i for i, v in enumerate(view['links'])}
    rows = target.link_id.map(where)
    assert rows.notna().all(), 'Template link without a model row'
    rows = rows.to_numpy(dtype=int)
    stamps = pd.to_datetime(target.timestamp, utc=True)
    assert not pd.DataFrame(dict(link=target.link_id.to_numpy(), stamp=stamps.to_numpy())).duplicated().any(), 'Template cell repeated'
    seconds = (stamps - origin).dt.total_seconds().to_numpy()
    steps = (seconds // 300).astype(int)
    assert (steps * 300 == seconds).all() and steps.min() >= 1 and steps.max() <= H, 'Template cell outside T+5..T+30'
    return {name: (ensemble(models, x)[rows, steps - 1] >= threshold).astype('int8') for name, (models, threshold) in arms.items()}


def build_predictions(arms, panels=PANELS2, splits=SPLITS, rel=REL, load=load_release):
    frames = {name: [] for name in arms}
    summary = []
    for split in splits:
        assert split in SPLITS, 'Only the official test months are read here'
        for panel in panels:
            P = load(panel, split)
            root = Path(rel) / 'task2' / panel / split
            windows = pd.read_csv(root / 'window_index.csv', dtype=TEXT)
            history = pd.read_parquet(root / 'window_history.parquet')
            history['window_id'] = history.window_id.astype(str)
            template = pd.read_csv(root / 'sample_submission_queue.csv', dtype=TEXT)
            days = {v: i for i, v in enumerate(P['days'])}
            for w in windows.itertuples():
                origin = window_origin(w, panel, split)
                d = days[str(w.date)]
                view = overlay_history(P, d, origin, history[history.window_id == w.window_id])
                target = template[template.window_id == w.window_id]
                assert len(target) == len(P['links']) * H, 'Template does not cover every link and horizon'
                decided = window_decisions(view, d, origin, target, arms)
                for name, values in decided.items():
                    frames[name].append(pd.DataFrame(dict(panel=panel, split=split, window_id=w.window_id, link_id=target.link_id.to_numpy(),
                                                          timestamp=target.timestamp.to_numpy(), queue_pred=values)))
                    summary.append(dict(arm=name, panel=panel, split=split, window_id=w.window_id, condition=w.condition,
                                        cells=int(len(values)), positive=int(values.sum())))
            print(json.dumps(dict(stage='inference', panel=panel, split=split, windows=len(windows))), flush=True)
    series = {}
    for name, parts in frames.items():
        frame = pd.concat(parts, ignore_index=True)
        assert not frame.duplicated(KEY_COLUMNS).any()
        series[name] = frame.set_index(KEY_COLUMNS).queue_pred
    return series, pd.DataFrame(summary)


def queue_keys(key):
    """Key frames of the release, read as text."""
    return pd.read_csv(key, usecols=['submission_id', 'task'] + KEY_COLUMNS, chunksize=100000, dtype=TEXT)


def fields(line):
    cells = line.rstrip(b'\r\n').split(b',')
    assert len(cells) == len(COLUMNS) and b'"' not in line, 'Unexpected base row'
    return cells


def combine(predictions, dest, base, base_sha256, key, write=True):
    """Copy the frozen base line by line, rewriting only the queue field of the queue rows."""
    assert digest(base) == base_sha256, 'Frozen historical base changed'
    assert predictions.index.is_unique
    rows = queue_rows = changed = 0
    used = []
    stream = Path(dest).open('wb') if write else None
    try:
        with Path(base).open('rb') as source:
            header = source.readline()
            assert header.rstrip(b'\r\n').split(b',') == COLUMNS, 'Unexpected base columns'
            if write:
                stream.write(header)
            for part in queue_keys(key):
                mask = (part.task == 'queue').to_numpy()
                values = np.full(len(part), -1)
                if mask.any():
                    wanted = pd.MultiIndex.from_frame(part.loc[mask, KEY_COLUMNS])
                    got = predictions.reindex(wanted)
                    assert not got.isna().any(), 'A required queue cell is missing'
                    values[mask] = got.to_numpy(dtype=int)
                    assert np.isin(values[mask], [0, 1]).all()
                    used.append(wanted)
                ids = part.submission_id.to_numpy(dtype='int64')
                tasks = part.task.to_numpy()
                for i in range(len(part)):
                    line = source.readline()
                    assert line, 'Base shorter than key'
                    cells = fields(line)
                    assert int(cells[0]) == ids[i] and cells[1].decode() == tasks[i], 'Base and key rows differ'
                    if mask[i]:
                        changed += int(float(cells[QUEUE_FIELD]) != values[i])
                        ending = line[len(line.rstrip(b'\r\n')):]
                        cells[QUEUE_FIELD] = str(values[i]).encode()
                        line = b','.join(cells) + ending
                    if write:
                        stream.write(line)
                rows += len(part)
                queue_rows += int(mask.sum())
            assert source.readline() == b'', 'Base longer than key'
    finally:
        if stream:
            stream.close()
    assert used, 'Key has no queue cells'
    keys = used[0].append(used[1:])
    assert keys.is_unique and len(keys) == len(predictions) == queue_rows, 'Key cells and predictions differ'
    assert keys.difference(predictions.index).empty and predictions.index.difference(keys).empty, 'Key cells and predictions differ'
    assert rows > queue_rows
    return dict(rows=rows, queue_rows=queue_rows, queue_cells_changed_from_base=changed, base_sha256=base_sha256,
                preserved_tasks=['state', 'physics-from-state', 'odme'], portal_submission=False)


def archive(csv):
    """Zip for upload by the owner, the format that Kaggle accepted for sub_v1."""
    path = Path(csv).with_suffix('.zip')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(csv, Path(csv).name)
    return path


def describe(summary):
    out = {}
    for name, frame in summary.groupby('arm'):
        out[name] = dict(positive_share=float(frame.positive.sum() / frame.cells.sum()),
                         empty_windows=int((frame.positive == 0).sum()), windows=int(len(frame)),
                         empty_by_condition={k: int((v.positive == 0).sum()) for k, v in frame.groupby('condition')},
                         by_condition={k: float(v.positive.sum() / v.cells.sum()) for k, v in frame.groupby('condition')},
                         by_split={k: float(v.positive.sum() / v.cells.sum()) for k, v in frame.groupby('split')})
    return out
