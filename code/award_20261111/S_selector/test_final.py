"""CPU tests for the closing source: causality, frozen decisions, untouched tasks, no sealed days."""
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

import final_logic as fl
import final_selector
import verify_final
from metric import H, model_for

HERE = Path(__file__).resolve().parent
LINKS = [f'L{i}' for i in range(6)]
DAY = '2031-03-02'
ORIGIN = pd.Timestamp(DAY + 'T08:00:00Z')
T0 = 96
rng = np.random.default_rng(11)


def panel():
    s = rng.uniform(40, 110, (2, len(LINKS), 288)).astype('float32')
    q = rng.uniform(500, 2000, (2, len(LINKS), 288)).astype('float32')
    return dict(panel='X', split='validation', days=['2031-03-01', DAY], links=LINKS, s=s, q=q,
                vf=np.full(len(LINKS), 105, 'float32'), lanes=np.full(len(LINKS), 3, 'float32'), dow=np.array([5, 6]))


def history(origin=ORIGIN, steps=range(-12, 0)):
    rows = []
    for k in steps:
        stamp = (origin + pd.Timedelta(minutes=5 * k)).strftime('%Y-%m-%dT%H:%M:%SZ')
        rows += [dict(window_id='W1', timestamp=stamp, link_id=link, speed_kmh=30. + k, flow_vph=900. + k) for link in LINKS]
    return pd.DataFrame(rows)


def target(origin=ORIGIN):
    rows = []
    for k in range(1, H + 1):
        stamp = (origin + pd.Timedelta(minutes=5 * k)).strftime('%Y-%m-%dT%H:%M:%SZ')
        rows += [dict(window_id='W1', timestamp=stamp, link_id=link, queue_pred='0') for link in LINKS]
    return pd.DataFrame(rows)


class Fixed:
    """Stands in for a network: a constant logit per call."""

    def __init__(self, logit):
        self.logit = logit

    def __call__(self, x):
        return torch.full((len(x), H), self.logit)


def stub(*probabilities):
    ck = dict(mean=np.zeros(95, 'float32'), std=np.ones(95, 'float32'))
    return [(ck, Fixed(float(np.log(p / (1 - p))))) for p in probabilities]


class Overlay(unittest.TestCase):
    def test_replaces_only_history(self):
        P = panel()
        view = fl.overlay_history(P, 1, ORIGIN, history())
        s = view['s'][1]
        np.testing.assert_array_equal(s[:, T0 - 12:T0], np.tile(30. + np.arange(-12, 0), (len(LINKS), 1)).astype('float32'))
        np.testing.assert_array_equal(s[:, T0:], P['s'][1][:, T0:])
        np.testing.assert_array_equal(s[:, :T0 - 12], P['s'][1][:, :T0 - 12])

    def test_refuses_row_at_or_after_origin(self):
        with self.assertRaises(AssertionError):
            fl.overlay_history(panel(), 1, ORIGIN, history(steps=range(-11, 1)))

    def test_refuses_rows_older_than_window(self):
        with self.assertRaises(AssertionError):
            fl.overlay_history(panel(), 1, ORIGIN, history(steps=range(-13, -1)))

    def test_refuses_history_shifted_off_the_grid(self):
        h = history()
        h['timestamp'] = (pd.to_datetime(h.timestamp, utc=True) + pd.Timedelta(seconds=30)).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        with self.assertRaises(AssertionError):
            fl.overlay_history(panel(), 1, ORIGIN, h)

    def test_refuses_origin_off_the_grid(self):
        with self.assertRaises(AssertionError):
            fl.overlay_history(panel(), 1, ORIGIN + pd.Timedelta(seconds=30), history(ORIGIN + pd.Timedelta(seconds=30)))

    def test_refuses_origin_not_in_utc(self):
        shifted = ORIGIN.tz_convert('Etc/GMT-1')
        with self.assertRaises(AssertionError):
            fl.overlay_history(panel(), 1, shifted, history())
        with self.assertRaises(AssertionError):
            fl.tick(shifted)

    def test_refuses_incomplete_history(self):
        with self.assertRaises(AssertionError):
            fl.overlay_history(panel(), 1, ORIGIN, history().iloc[1:])

    def test_refuses_unknown_link(self):
        h = history()
        h.loc[0, 'link_id'] = 'nope'
        with self.assertRaises(AssertionError):
            fl.overlay_history(panel(), 1, ORIGIN, h)


class Causality(unittest.TestCase):
    def test_nothing_after_origin_reaches_features(self):
        view = fl.overlay_history(panel(), 1, ORIGIN, history())
        before = fl.feature_matrix(view, 1, T0)
        view['s'][1][:, T0 + 1:] = 1.
        view['q'][1][:, T0 + 1:] = 1.
        view['s'][1][:, :T0 - 12] = 1.
        np.testing.assert_array_equal(before, fl.feature_matrix(view, 1, T0))

    def test_row_at_origin_is_used(self):
        view = fl.overlay_history(panel(), 1, ORIGIN, history())
        before = fl.feature_matrix(view, 1, T0)
        view['s'][1][:, T0] = 5.
        assert not np.array_equal(before, fl.feature_matrix(view, 1, T0))


class Decisions(unittest.TestCase):
    def test_ensemble_is_mean_of_probabilities(self):
        p = fl.ensemble(stub(.2, .5, .8), np.zeros((4, 95), 'float32'))
        np.testing.assert_allclose(p, .5, atol=1e-6)

    def test_threshold_applies_to_ensemble_mean(self):
        view = fl.overlay_history(panel(), 1, ORIGIN, history())
        out = fl.window_decisions(view, 1, ORIGIN, target(), dict(a=(stub(.1, .3, .95), .4), b=(stub(.1, .3, .95), .5)))
        assert out['a'].all() and not out['b'].any()

    def test_horizon_matches_template_cell(self):
        class Step:
            def __call__(self, x):
                return torch.tensor([[-9., -9., -9., -9., -9., 9.]]).repeat(len(x), 1)
        ck = dict(mean=np.zeros(95, 'float32'), std=np.ones(95, 'float32'))
        view = fl.overlay_history(panel(), 1, ORIGIN, history())
        t = target()
        out = fl.window_decisions(view, 1, ORIGIN, t, dict(a=([(ck, Step())], .5)))['a']
        last = (ORIGIN + pd.Timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
        np.testing.assert_array_equal(out.astype(bool), (t.timestamp == last).to_numpy())

    def test_refuses_cell_outside_horizon(self):
        view = fl.overlay_history(panel(), 1, ORIGIN, history())
        t = target()
        t.loc[0, 'timestamp'] = (ORIGIN + pd.Timedelta(minutes=35)).strftime('%Y-%m-%dT%H:%M:%SZ')
        with self.assertRaises(AssertionError):
            fl.window_decisions(view, 1, ORIGIN, t, dict(a=(stub(.5), .5)))


    def test_refuses_timestamp_off_the_grid(self):
        view = fl.overlay_history(panel(), 1, ORIGIN, history())
        t = target()
        t.loc[0, 'timestamp'] = (ORIGIN + pd.Timedelta(seconds=330)).strftime('%Y-%m-%dT%H:%M:%SZ')
        with self.assertRaises(AssertionError):
            fl.window_decisions(view, 1, ORIGIN, t, dict(a=(stub(.5), .5)))

    def test_refuses_same_cell_written_two_ways(self):
        view = fl.overlay_history(panel(), 1, ORIGIN, history())
        t = target()
        t.loc[1, 'link_id'] = t.loc[0, 'link_id']
        t.loc[1, 'timestamp'] = t.loc[0, 'timestamp'].replace('Z', '+00:00')
        with self.assertRaises(AssertionError):
            fl.window_decisions(view, 1, ORIGIN, t, dict(a=(stub(.5), .5)))

    def test_refuses_repeated_template_cell(self):
        view = fl.overlay_history(panel(), 1, ORIGIN, history())
        t = target()
        t.loc[1, ['link_id', 'timestamp']] = t.loc[0, ['link_id', 'timestamp']].to_numpy()
        with self.assertRaises(AssertionError):
            fl.window_decisions(view, 1, ORIGIN, t, dict(a=(stub(.5), .5)))


class Models(unittest.TestCase):
    def saved(self, root):
        cfg = dict(depth=1, width=8, dropout=.1)
        net = model_for(torch, 95, cfg)
        path = Path(root) / 'm.pt'
        torch.save(dict(state_dict=net.state_dict(), cfg=cfg, mean=np.zeros(95, 'float32'), std=np.ones(95, 'float32')), path)
        return path

    def test_refuses_changed_model(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.saved(root)
            with self.assertRaises(AssertionError):
                fl.load_arm(dict(models=[dict(path=str(path), sha256='0' * 64)]))

    def test_loaded_model_is_in_eval_mode(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.saved(root)
            (_, net), = fl.load_arm(dict(models=[dict(path=str(path), sha256=fl.digest(path))]))
            assert not net.training


class Build(unittest.TestCase):
    def release(self, root, **change):
        where = Path(root) / 'task2/X/validation'
        where.mkdir(parents=True)
        row = dict(window_id='W1', panel='X', split='validation', date=DAY, forecast_origin='2031-03-02T08:00:00Z', condition='queue_onset')
        pd.DataFrame([dict(row, **change)]).to_csv(where / 'window_index.csv', index=False)
        history().to_parquet(where / 'window_history.parquet')
        target().to_csv(where / 'sample_submission_queue.csv', index=False)

    def test_train_is_never_loaded(self):
        def loader(*a, **k):
            raise RuntimeError('loader must not run')
        with self.assertRaises(AssertionError):
            fl.build_predictions({}, panels=['X'], splits=('train',), load=loader)

    def test_only_official_test_months(self):
        assert fl.SPLITS == ('validation', 'private')

    def test_tiny_release_end_to_end(self):
        with tempfile.TemporaryDirectory() as root:
            self.release(root)
            series, summary = fl.build_predictions(dict(a=(stub(.9), .5)), panels=['X'], splits=('validation',), rel=root, load=lambda name, split: panel())
        assert len(series['a']) == H * len(LINKS) and series['a'].all()
        assert summary.positive.sum() == H * len(LINKS)
        assert fl.describe(summary)['a']['by_condition'] == {'queue_onset': 1.}


    def test_origin_in_another_zone_means_the_same_instant(self):
        with tempfile.TemporaryDirectory() as root:
            self.release(root, forecast_origin='2031-03-02T09:00:00+01:00')
            series, _ = fl.build_predictions(dict(a=(stub(.9), .5)), panels=['X'], splits=('validation',), rel=root, load=lambda name, split: panel())
        assert len(series['a']) == H * len(LINKS)

    def test_refuses_index_row_that_contradicts_itself(self):
        for change in (dict(date='2031-03-01'), dict(panel='Y'), dict(split='private')):
            with tempfile.TemporaryDirectory() as root:
                self.release(root, **change)
                with self.assertRaises(AssertionError):
                    fl.build_predictions(dict(a=(stub(.9), .5)), panels=['X'], splits=('validation',), rel=root, load=lambda name, split: panel())


class Release(unittest.TestCase):
    def masked(self, root, stamp, date='2031-03-02'):
        where = Path(root) / 'corridors/X/validation/mainline_states_masked/mask_regime=a'
        where.mkdir(parents=True)
        pd.DataFrame(dict(date=[date], timestamp=[stamp])).to_parquet(where / 'f.parquet')

    def test_reads_the_masked_layer_only(self):
        calls = []

        def loader(name, split, truth=True):
            calls.append(truth)
            return 'month'
        with tempfile.TemporaryDirectory() as root:
            self.masked(root, '2031-03-02T08:00:00Z')
            assert fl.load_release('X', 'validation', rel=root, loader=loader) == 'month'
        assert calls == [False]

    def test_refuses_stamp_off_the_grid(self):
        def loader(*a, **k):
            raise RuntimeError('loader must not run')
        with tempfile.TemporaryDirectory() as root:
            self.masked(root, '2031-03-02T08:00:30Z')
            with self.assertRaises(AssertionError):
                fl.load_release('X', 'validation', rel=root, loader=loader)

    def test_refuses_row_dated_apart_from_its_stamp(self):
        def loader(*a, **k):
            raise RuntimeError('loader must not run')
        with tempfile.TemporaryDirectory() as root:
            self.masked(root, '2031-03-03T08:00:00Z')
            with self.assertRaises(AssertionError):
                fl.load_release('X', 'validation', rel=root, loader=loader)

    def test_refuses_train(self):
        with self.assertRaises(AssertionError):
            fl.load_release('X', 'train')


class Combine(unittest.TestCase):
    def files(self, root):
        root = Path(root)
        base = pd.DataFrame(dict(submission_id=[1, 2, 3, 4], task=['state', 'queue', 'queue', 'odme'], speed_kmh=[1.5, 0, 0, 0],
                                 flow_vph=[2.5, 0, 0, 0], queue_pred=[0, 0, 1, 0], path_flow=[0, 0, 0, 7.25]))
        key = pd.DataFrame(dict(submission_id=[1, 2, 3, 4], task=base.task, panel=['X'] * 4, split=['validation'] * 4,
                                window_id=[None, 'W1', 'W1', None], link_id=[None, 'L0', 'L1', None], timestamp=[None, 't', 't', None]))
        base.to_csv(root / 'base.csv', index=False)
        key.to_csv(root / 'key.csv', index=False)
        index = pd.MultiIndex.from_tuples([('X', 'validation', 'W1', 'L0', 't'), ('X', 'validation', 'W1', 'L1', 't')], names=fl.KEY_COLUMNS)
        return root / 'base.csv', root / 'key.csv', pd.Series([1, 0], index=index, name='queue_pred')

    def test_changes_only_queue_cells(self):
        with tempfile.TemporaryDirectory() as root:
            base, key, series = self.files(root)
            dest = Path(root) / 'out.csv'
            info = fl.combine(series, dest, base, fl.digest(base), key)
            old, new = pd.read_csv(base), pd.read_csv(dest)
            assert list(old.columns) == list(new.columns) and info['queue_cells_changed_from_base'] == 2
            assert new.queue_pred.tolist() == [0, 1, 0, 0]
            pd.testing.assert_frame_equal(old.drop(columns='queue_pred'), new.drop(columns='queue_pred'))

    def test_kept_values_stay_byte_identical(self):
        with tempfile.TemporaryDirectory() as root:
            base, key, series = self.files(root)
            raw = base.read_bytes().replace(b'1.5', b'1.50000000000000012').replace(b'\n', b'\r\n')
            base.write_bytes(raw)
            dest = Path(root) / 'out.csv'
            fl.combine(series, dest, base, fl.digest(base), key)
            old, new = raw.splitlines(True), dest.read_bytes().splitlines(True)
            assert len(old) == len(new) and old[0] == new[0] and old[1] == new[1] and old[4] == new[4]
            assert new[2] == b'2,queue,0.0,0.0,1,0.0\r\n' and new[3] == b'3,queue,0.0,0.0,0,0.0\r\n'

    def test_refuses_quoted_base_row(self):
        with tempfile.TemporaryDirectory() as root:
            base, key, series = self.files(root)
            base.write_bytes(base.read_bytes().replace(b'1.5', b'"1.5"'))
            with self.assertRaises(AssertionError):
                fl.combine(series, Path(root) / 'out.csv', base, fl.digest(base), key)

    def test_refuses_repeated_key_cell(self):
        with tempfile.TemporaryDirectory() as root:
            base, key, series = self.files(root)
            key.write_text(key.read_text().replace(',L1,', ',L0,'))
            with self.assertRaises(AssertionError):
                fl.combine(series, Path(root) / 'out.csv', base, fl.digest(base), key)

    def test_refuses_base_shorter_or_longer_than_key(self):
        with tempfile.TemporaryDirectory() as root:
            base, key, series = self.files(root)
            full = base.read_bytes()
            for broken in (full[:full.rindex(b'4,odme')], full + b'5,odme,0,0,0,0\n'):
                base.write_bytes(broken)
                with self.assertRaises(AssertionError):
                    fl.combine(series, Path(root) / 'out.csv', base, fl.digest(base), key)

    def test_refuses_missing_cell(self):
        with tempfile.TemporaryDirectory() as root:
            base, key, series = self.files(root)
            with self.assertRaises(AssertionError):
                fl.combine(series.iloc[:1], Path(root) / 'out.csv', base, fl.digest(base), key)

    def test_refuses_changed_base(self):
        with tempfile.TemporaryDirectory() as root:
            base, key, series = self.files(root)
            with self.assertRaises(AssertionError):
                fl.combine(series, Path(root) / 'out.csv', base, '0' * 64, key)

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            base, key, series = self.files(root)
            fl.combine(series, Path(root) / 'out.csv', base, fl.digest(base), key, write=False)
            assert not (Path(root) / 'out.csv').exists()

    def test_archive_keeps_the_file(self):
        with tempfile.TemporaryDirectory() as root:
            csv = Path(root) / 'c.csv'
            csv.write_text('a\n1\n')
            with zipfile.ZipFile(fl.archive(csv)) as z:
                assert z.read('c.csv') == b'a\n1\n'


class Verify(unittest.TestCase):
    def candidate(self, root):
        base, key, series = Combine.files(self, root)
        dest = Path(root) / 'out.csv'
        fl.combine(series, dest, base, fl.digest(base), key)
        return dest, base, key

    def test_accepts_a_faithful_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            dest, base, key = self.candidate(root)
            cells, equal, rows = verify_final.candidate_cells(dest, base, key)
            assert equal and rows == 4 and cells.tolist() == [1, 0]

    def test_detects_missing_last_row(self):
        with tempfile.TemporaryDirectory() as root:
            dest, base, key = self.candidate(root)
            raw = dest.read_bytes()
            dest.write_bytes(raw[:raw.rindex(b'4,odme')])
            with self.assertRaises(AssertionError):
                verify_final.candidate_cells(dest, base, key)

    def test_detects_extra_row(self):
        with tempfile.TemporaryDirectory() as root:
            dest, base, key = self.candidate(root)
            dest.write_bytes(dest.read_bytes() + b'5,odme,0,0,0,0\n')
            with self.assertRaises(AssertionError):
                verify_final.candidate_cells(dest, base, key)

    def test_detects_changed_other_task(self):
        with tempfile.TemporaryDirectory() as root:
            dest, base, key = self.candidate(root)
            dest.write_bytes(dest.read_bytes().replace(b'7.25', b'7.5'))
            assert verify_final.candidate_cells(dest, base, key)[1] is False

    def test_detects_invalid_queue_value(self):
        with tempfile.TemporaryDirectory() as root:
            dest, base, key = self.candidate(root)
            dest.write_bytes(dest.read_bytes().replace(b'2,queue,0.0,0.0,1,', b'2,queue,0.0,0.0,2,'))
            with self.assertRaises(AssertionError):
                verify_final.candidate_cells(dest, base, key)


class Numpy(unittest.TestCase):
    def test_second_route_matches_torch(self):
        cfg = dict(depth=2, width=8, dropout=.1)
        net = model_for(torch, 95, cfg).eval()
        ck = dict(state_dict=net.state_dict(), cfg=cfg, mean=np.zeros(95, 'float32'), std=np.ones(95, 'float32'))
        x = rng.normal(size=(5, 95)).astype('float32')
        np.testing.assert_allclose(verify_final.numpy_forward(ck, x), fl.ensemble([(ck, net)], x), atol=1e-5)


class Config(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((HERE / 'config/final.json').read_text())

    def test_frozen_thresholds(self):
        assert self.c['arms']['S']['threshold'] == .5 and self.c['arms']['F']['threshold'] == .6

    def test_three_models_per_arm(self):
        assert all(len(a['models']) == 3 for a in self.c['arms'].values())

    def test_no_submission_or_sealed_days(self):
        assert self.c['confirmation_opened'] is False and self.c['portal_submission'] is False


class Main(unittest.TestCase):
    def test_choice_is_on_disk_before_inference_and_nothing_is_sent(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / 'results'
            seen = []

            def fake(arms):
                seen.append(json.loads((out / 'frozen_choice.json').read_text()))
                index = pd.MultiIndex.from_tuples([('X', 'validation', 'W', 'L', 't')], names=fl.KEY_COLUMNS)
                frame = pd.DataFrame([dict(arm='S', panel='X', split='validation', window_id='W', condition='queue_onset', cells=1, positive=1)])
                return {'S': pd.Series([1], index=index)}, frame

            def fake_combine(series, dest, *a):
                Path(dest).write_text('x\n')
                return dict(rows=1)
            cwd = Path.cwd()
            with mock.patch.object(sys, 'argv', ['final_selector.py', '--out', str(out)]), \
                    mock.patch.object(final_selector, 'check_sources', return_value='m'), \
                    mock.patch.object(final_selector, 'load_arm', return_value=[]), \
                    mock.patch.object(final_selector, 'build_predictions', fake), \
                    mock.patch.object(final_selector, 'combine', fake_combine):
                os.chdir(root)
                try:
                    final_selector.main()
                finally:
                    os.chdir(cwd)
            result = json.loads((out / 'result.json').read_text())
            assert seen[0]['thresholds'] == dict(S=.5, F=.6)
            assert result['submission_sent'] is False and result['promoted'] is False and result['confirmation_opened'] is False


if __name__ == '__main__':
    unittest.main()
