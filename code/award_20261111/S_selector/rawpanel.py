"""Read raw daily parquet for days 0..257 only. Later files are never opened."""
import glob, re
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq

REL = Path('/home/ai/proyectos/traffic-flow-bench/data/kaggle_public')
LAST_OPEN_DAY = 257
LAST_OPEN_DATE = '2031-02-13'


def dateof(path):
    return '-'.join(re.search(r'(\d{4})_(\d{2})_(\d{2})', Path(path).stem).groups())


def allowed_dates(files):
    dates = sorted({dateof(p) for p in files})
    assert len(dates) == 273, len(dates)
    return dates[:LAST_OPEN_DAY + 1]


def read_layer(files, dates, links, columns):
    """Return {column:[D,L,288]} for the listed dates; a file of any other date stays closed."""
    assert len(dates) <= LAST_OPEN_DAY + 1 and max(dates) <= LAST_OPEN_DATE, 'Reserved days stay closed'
    shape = (len(dates), len(links), 288)
    out = {c: np.zeros(shape, dtype=bool) if c == 'is_score_eligible' else np.full(shape, np.nan, dtype='float32') for c in columns}
    where = {d: i for i, d in enumerate(dates)}
    ids = pa.array(links)
    opened = []
    for path in files:
        date = dateof(path)
        if date not in where:
            continue
        table = pq.read_table(path, columns=['link_id', 'timestamp'] + list(columns), use_threads=False)
        idx = pc.index_in(pc.cast(table.column('link_id'), pa.string()), value_set=ids)
        keep = pc.is_valid(idx)
        table = table.filter(keep)
        idx = pc.index_in(pc.cast(table.column('link_id'), pa.string()), value_set=ids).to_numpy(zero_copy_only=False).astype(int)
        stamp = pc.cast(table.column('timestamp'), pa.string())
        assert pc.all(pc.equal(pc.utf8_slice_codeunits(stamp, 0, 10), date)).as_py()
        hour = pc.cast(pc.utf8_slice_codeunits(stamp, 11, 13), pa.int16()).to_numpy(zero_copy_only=False).astype(int)
        minute = pc.cast(pc.utf8_slice_codeunits(stamp, 14, 16), pa.int16()).to_numpy(zero_copy_only=False).astype(int)
        assert np.all(minute % 5 == 0)
        tick = hour * 12 + minute // 5
        for c in columns:
            if c == 'is_score_eligible':
                out[c][where[date], idx, tick] = table.column(c).to_numpy(zero_copy_only=False).astype(bool)
            else:
                out[c][where[date], idx, tick] = pc.cast(table.column(c), pa.float32(), safe=False).to_numpy(zero_copy_only=False)
        opened.append(path)
    assert len({dateof(p) for p in opened}) == len(dates)
    return out, opened


def load(name, links):
    root = REL / 'corridors' / name / 'train'
    masked = sorted(glob.glob(str(root / 'mainline_states_masked/mask_regime=*/*.parquet')))
    truth = sorted(glob.glob(str(root / 'mainline_states/*/*.parquet')))
    dates = allowed_dates(masked)
    assert dates == allowed_dates(truth)
    m, opened_m = read_layer(masked, dates, links, ['speed_kmh', 'flow_vph'])
    t, opened_t = read_layer(truth, dates, links, ['speed_kmh', 'flow_vph', 'is_score_eligible'])
    meta = pd.read_csv(REL / 'corridors' / name / 'network/links.csv').set_index('link_id')
    meta.index = meta.index.astype(str)
    return dict(panel=name, days=dates, links=list(links), s=m['speed_kmh'], q=m['flow_vph'], s_true=t['speed_kmh'], q_true=t['flow_vph'],
                elig=t['is_score_eligible'], vf=meta.loc[links, 'free_speed_kmh'].to_numpy(dtype='float32'), lanes=meta.loc[links, 'lanes'].to_numpy(dtype='float32'),
                dow=pd.to_datetime(pd.Series(dates)).dt.dayofweek.to_numpy(), opened=opened_m + opened_t)
