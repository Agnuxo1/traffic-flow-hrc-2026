"""Independent replica of the Task-2 queue predictor with a frozen, pre-registered protocol.

NOT a submission builder: this program never writes a Kaggle submission file.

Design (see ../config/preregistration.json, written BEFORE any result):
  * same predictor family as job 20b0502495c44cdf (MLP on per-link window features, same
    feature code), but an independent root seed (default 20261106 vs 20261105), an
    independent window sample, and the exact official aggregation of the score
    (window -> panel x condition -> panel -> family -> overall, equal means).
  * windows follow the published Task-2 rules: queue in horizon, history coverage >= 0.70,
    persistence IoU <= 0.90, onset/ongoing by the published visible-history rule.
  * two arms on identical windows/seeds/config:
      A  "masked"   : history features from the day's masked layer (what job 20b0 trains on)
      B  "testlike" : history rows t0-12..t0-1 taken from the unmasked layer, exactly what an
                      official window_history shows at validation/private time (row t0 stays masked)
    BOTH arms are scored on test-like selection inputs.
  * stage 1 (search) uses arm A only; the winning config is frozen on disk (sha256) before stage 2.
  * stage 2 trains 3 seeds x 2 arms; the reserved confirmation partition (last 15 train days)
    is read exactly once, after all 6 models are fixed.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from queue_features import window_feats, ratio_hist, labels   # copied unchanged from job 20b0502495c44cdf

T = 288
H = 6
PANELS2 = ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]
FAMILY = {"D12_I5_N": "D12_I5", "D12_I5_S": "D12_I5", "D7_I10_E": "D7_I10", "D7_I10_W": "D7_I10",
          "D7_I210_E": "D7_I210", "D7_I210_W": "D7_I210", "D7_I405_N": "D7_I405", "D7_I405_S": "D7_I405"}
COND = {"queue_onset": 0, "queue_ongoing": 1}
THRESHOLDS = [.2, .3, .4, .5, .6, .7]
MIN_COVERAGE = 0.70
MAX_PERSISTENCE_IOU = 0.90


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8'); tmp.replace(path)


# ----------------------------------------------------------------------------- windows
def view(P, d, t0, testlike):
    """Per-window panel view. testlike: history rows t0-12..t0-1 come from the unmasked layer."""
    s = P['s'][d].copy(); q = P['q'][d].copy()
    if testlike:
        s[:, t0 - 12:t0] = P['s_true'][d][:, t0 - 12:t0]
        q[:, t0 - 12:t0] = P['q_true'][d][:, t0 - 12:t0]
    return {'s': {d: s}, 'q': {d: q}, 'vf': P['vf'], 'lanes': P['lanes'], 'dow': P['dow'], 'links': P['links']}


def feature_matrix(P, d, t0):
    base, _ = window_feats(P, d, t0, None)
    ratios = ratio_hist(P, d, t0)
    position = np.arange(len(P['links']))
    history = np.concatenate([ratios[np.clip(position + o, 0, len(position) - 1)] for o in [-2, -1, 0, 1, 2]], axis=1)
    return np.nan_to_num(np.concatenate([base, history], axis=1), nan=0., posinf=10., neginf=-10.).astype('float32')


def window_table(P, d):
    """Vectorised published-rule screen for every t0 of day d. Uses the TRUTH layer only to pick
    windows (as the organisers do), never as model input beyond the test-like history."""
    vf = P['vf'][:, None]
    s = P['s_true'][d]; el = P['elig'][d]
    qobs = (s <= 0.6 * vf) & el                                  # eligible queued cell
    t = np.arange(12, T - H - 1)
    cq = np.concatenate([np.zeros((qobs.shape[0], 1)), np.cumsum(qobs, axis=1)], axis=1)   # [L, T+1]
    ce = np.concatenate([np.zeros((qobs.shape[0], 1)), np.cumsum(el, axis=1)], axis=1)
    hist_q_link = cq[:, t] - cq[:, t - 12]                       # queued obs per link in steps t-12..t-1
    hist_q = hist_q_link.sum(0)
    ongoing = (hist_q >= 2) & (hist_q_link.max(0) >= 2)
    cov = (ce[:, t] - ce[:, t - 12]).mean(0) / 12.0
    fut = np.zeros(len(t), bool); inter = np.zeros(len(t)); union = np.zeros(len(t))
    pred = qobs[:, t - 1]
    for h in range(1, H + 1):
        fq = qobs[:, t + h]
        m = el[:, t + h] & np.isfinite(s[:, t + h])
        fut |= fq.any(0)
        inter += (pred & fq & m).sum(0); union += ((pred | fq) & m).sum(0)
    pers = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    ok = fut & (cov >= MIN_COVERAGE) & (pers <= MAX_PERSISTENCE_IOU)
    return t, ok, ongoing, pers


def sample_windows_official(P, rng, per_day):
    out = []
    for d in range(len(P['days'])):
        t, ok, ongoing, pers = window_table(P, d)
        for cond, flag in (('queue_onset', ~ongoing), ('queue_ongoing', ongoing)):
            idx = np.where(ok & flag)[0]
            if len(idx):
                for i in rng.permutation(idx)[:per_day]:
                    out.append((d, int(t[i]), cond, float(pers[i])))
    return out


def build_cache(P, panel_index, root_seed, per_day):
    """Return dict of arrays for one panel (rows = links of one window)."""
    D = len(P['days'])
    rng = np.random.default_rng(root_seed + 1000 * panel_index)
    xm, xt, ys, ms, gs, sp, dy, cd, ps, pn = ([] for _ in range(10))
    for g, (d, t0, cond, pers) in enumerate(sample_windows_official(P, rng, per_day)):
        a = feature_matrix(view(P, d, t0, False), d, t0)
        b = feature_matrix(view(P, d, t0, True), d, t0)
        y, m = labels(P, d, t0)
        n = len(a)
        split = 0 if d < D - 45 else 1 if d < D - 15 else 2
        pr = ((P['s_true'][d][:, t0 - 1] <= 0.6 * P['vf']) & P['elig'][d][:, t0 - 1]).astype('float32')
        xm.append(a); xt.append(b); ys.append(y.astype('float32')); ms.append(m.astype('float32'))
        gs.append(np.full(n, g, 'int32')); sp.append(np.full(n, split, 'int8')); dy.append(np.full(n, d, 'int32'))
        cd.append(np.full(n, COND[cond], 'int8')); ps.append(pr); pn.append(np.full(n, panel_index, 'int8'))
    assert xm, 'no windows survive the published selection rules for ' + P['panel']
    return dict(x_m=np.concatenate(xm), x_t=np.concatenate(xt), y=np.concatenate(ys), mask=np.concatenate(ms),
                group=np.concatenate(gs), split=np.concatenate(sp), day=np.concatenate(dy), cond=np.concatenate(cd),
                pers=np.concatenate(ps), panel=np.concatenate(pn))


# ----------------------------------------------------------------------------- exact metric
def window_ious(pred, y, mask, group):
    """pred bool [N,H]; official window IoU over ELIGIBLE cells; empty union -> 1.0; no eligible cell -> 0.0."""
    el = mask > 0; tr = y > .5
    inter = ((pred & tr) & el).sum(1); union = ((pred | tr) & el).sum(1)
    G = int(group.max()) + 1
    I = np.bincount(group, weights=inter, minlength=G); U = np.bincount(group, weights=union, minlength=G)
    C = np.bincount(group, weights=el.sum(1), minlength=G)
    return np.where(C == 0, 0.0, np.where(U > 0, I / np.maximum(U, 1), 1.0))


def exact_score(pred, y, mask, group, panel, cond):
    """window -> panel x condition -> panel -> family -> overall, equal means (published aggregation)."""
    iou = window_ious(pred, y, mask, group)
    ug, first = np.unique(group, return_index=True)
    w_panel = panel[first]; w_cond = cond[first]; w_iou = iou[ug]
    per_panel = {}; per_cond = {0: [], 1: []}
    for p in np.unique(w_panel):
        parts = []
        for c in (0, 1):
            sel = (w_panel == p) & (w_cond == c)
            if sel.any():
                parts.append(w_iou[sel].mean()); per_cond[c].append(parts[-1])
        per_panel[PANELS2[p] if p < len(PANELS2) else str(p)] = float(np.mean(parts))
    fam = {}
    for name, v in per_panel.items():
        fam.setdefault(FAMILY.get(name, name), []).append(v)
    fam_mean = {k: float(np.mean(v)) for k, v in fam.items()}
    return dict(overall=float(np.mean(list(fam_mean.values()))), per_family=fam_mean, per_panel=per_panel,
                per_condition={'queue_onset': float(np.mean(per_cond[0])) if per_cond[0] else None,
                               'queue_ongoing': float(np.mean(per_cond[1])) if per_cond[1] else None},
                n_windows=int(len(ug)))


# ----------------------------------------------------------------------------- model / training
def model_for(torch, features, cfg):
    nn = torch.nn
    layers = []; dim = features
    for _ in range(cfg['depth']):
        layers.extend([nn.Linear(dim, cfg['width']), nn.LayerNorm(cfg['width']), nn.SiLU(), nn.Dropout(cfg['dropout'])]); dim = cfg['width']
    layers.append(nn.Linear(dim, H))
    return nn.Sequential(*layers)


def fit_stats(x, train_mask):
    n = int(train_mask.sum()); total = np.zeros(x.shape[1], 'float64'); sq = np.zeros_like(total)
    for a in range(0, len(x), 65536):
        blk = x[a:a + 65536][train_mask[a:a + 65536]].astype('float64'); total += blk.sum(0); sq += (blk * blk).sum(0)
    mean = (total / n).astype('float32')
    std = np.maximum(np.sqrt(np.maximum(sq / n - (total / n) ** 2, 0)), .05).astype('float32')
    return mean, std


def normalised(x, mean, std):
    return np.clip((x - mean) / std, -10, 10).astype('float32')


def probabilities(torch, net, x, bs=16384):
    net.eval()
    with torch.no_grad():
        return torch.cat([net(x[a:a + bs]).sigmoid().float() for a in range(0, len(x), bs)]).cpu().numpy()


def select_threshold(prob, part):
    best = (-1.0, None)
    for th in THRESHOLDS:
        sc = exact_score(prob >= th, part['y'], part['mask'], part['group'], part['panel'], part['cond'])['overall']
        if sc > best[0]:
            best = (sc, th)
    return best


def train_model(torch, d, cfg, device, deadline, xtr, sel_x, sel):
    """xtr: normalised train features (torch, device). sel_x: normalised selection inputs for scoring."""
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'] % (2 ** 31))
    tr = d['split'] == 0
    y = torch.as_tensor(d['y'][tr], device=device); m = torch.as_tensor(d['mask'][tr], device=device)
    g = torch.as_tensor(d['group'][tr], device=device, dtype=torch.long)
    net = model_for(torch, xtr.shape[1], cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    _, inv = torch.unique(g, return_inverse=True)
    denom = torch.zeros(int(inv.max()) + 1, device=device).scatter_add_(0, inv, m.sum(1))
    w = 1 / denom[inv].clamp_min(1); w = w / w.mean()
    posw = torch.tensor(cfg['pos_weight'], device=device)
    best = dict(score=-1.0, threshold=None, epoch=0, state=None); bad = 0; epoch = 0
    for epoch in range(1, cfg.get('max_epochs', 160) + 1):
        if time.monotonic() >= deadline:
            break
        net.train()
        for b in torch.randperm(len(xtr), device=device).split(cfg['batch']):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xtr[b]), y[b], reduction='none', pos_weight=posw)
            loss = (loss * m[b] * w[b, None]).sum() / m[b].sum().clamp_min(1)
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.); opt.step()
        if epoch % 3:
            continue
        score, th = select_threshold(probabilities(torch, net, sel_x), sel)
        if score > best['score'] + 1e-5:
            best = dict(score=score, threshold=th, epoch=epoch, state={k: v.detach().cpu().clone() for k, v in net.state_dict().items()}); bad = 0
        else:
            bad += 1
        if bad >= 7:
            break
    assert best['state'] is not None, 'No evaluated checkpoint before deadline'
    net.load_state_dict(best['state'])
    return net, best, epoch


def partition(d, split, key):
    take = d['split'] == split
    return dict(x=d[key][take], y=d['y'][take], mask=d['mask'][take], group=d['group'][take], panel=d['panel'][take],
                cond=d['cond'][take], pers=d['pers'][take])


def sample_config(rng, seed):
    return dict(width=int(rng.choice([128, 256, 384, 512])), depth=int(rng.choice([2, 3, 4])),
                dropout=float(rng.choice([0., .05, .1, .2])), lr=float(rng.choice([.0003, .001, .002])),
                wd=float(rng.choice([1e-5, .001, .01])), pos_weight=float(rng.choice([1., 1.5, 2.])),
                batch=int(rng.choice([4096, 8192, 16384])), seed=int(seed))


def load_cache(cache_dir, panels):
    parts = []
    off = 0
    for name in panels:
        with np.load(Path(cache_dir) / (name + '.npz')) as z:
            c = {k: z[k] for k in z.files}
        c['group'] = c['group'] + off; off = int(c['group'].max()) + 1
        parts.append(c)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results_replica'); ap.add_argument('--root-seed', type=int, default=20261106)
    ap.add_argument('--per-day', type=int, default=2); ap.add_argument('--panels', default=','.join(PANELS2))
    ap.add_argument('--search-trials', type=int, default=24); ap.add_argument('--trial-seconds', type=int, default=600)
    ap.add_argument('--model-seconds', type=int, default=900); ap.add_argument('--hard-stop', default='2026-11-07T03:00:00Z')
    ap.add_argument('--cache-dir', default=None); ap.add_argument('--window-seed',type=int,default=20261106)
    ap.add_argument('--defer-confirmation',action='store_true')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--prereg', default=str(HERE.parent / 'config' / 'preregistration.json'))
    a = ap.parse_args()
    import torch
    from replica_loader import load_panel_truth
    if a.device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA required for the real run; use tests/test_cpu_smoke.py for CPU'
        assert torch.cuda.device_count() == 1, 'run only on the single GPU lane granted by the scheduler'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); panels = a.panels.split(',')
    prereg = json.loads(Path(a.prereg).read_text())
    write_json(out / 'started.json', dict(started_utc=pd.Timestamp.now(tz='UTC').isoformat(), root_seed=a.root_seed,
               preregistration_sha256=hashlib.sha256(Path(a.prereg).read_bytes()).hexdigest(), panels=panels,
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), submission_written=False))
    hard = pd.Timestamp(a.hard_stop).timestamp()
    cache = Path(a.cache_dir) if a.cache_dir else out / 'cache'; cache.mkdir(exist_ok=True)
    for i, name in enumerate(panels):
        dest = cache / (name + '.npz')
        if not dest.exists():
            assert not a.cache_dir, 'Approved shared cache is incomplete'
            P = load_panel_truth(name, 'train', truth=True)
            assert len(P['days']) > 60
            np.savez(dest, **build_cache(P, PANELS2.index(name), a.window_seed, a.per_day)); del P; gc.collect()
            print(json.dumps(dict(stage='cache', panel=name)), flush=True)
    d = load_cache(cache, panels)
    tr = d['split'] == 0
    stats = {'A': fit_stats(d['x_m'], tr), 'B': fit_stats(d['x_t'], tr)}     # train-partition rows only
    sel = partition(d, 1, 'x_t')
    sel_x = {k: torch.as_tensor(normalised(sel['x'], *stats[k]), device=a.device) for k in 'AB'}
    persistence_sel = exact_score(sel['pers'][:, None].repeat(H, 1) > 0, sel['y'], sel['mask'], sel['group'], sel['panel'], sel['cond'])
    write_json(out / 'selection_persistence.json', persistence_sel)
    write_json(out / 'partitions.json', {k: int((d['split'] == k).sum()) for k in (0, 1, 2)})
    xtr = {'A': torch.as_tensor(normalised(d['x_m'][tr], *stats['A']), device=a.device),
           'B': torch.as_tensor(normalised(d['x_t'][tr], *stats['B']), device=a.device)}
    # ---- stage 1: search, arm A only, selection partition only
    frozen_path = out / 'frozen_config.json'
    if not frozen_path.exists():
        rng = np.random.default_rng(a.root_seed); best = dict(score=-1)
        for trial in range(a.search_trials):
            if time.time() > hard - 4 * 3600:
                break
            cfg = sample_config(rng, int(rng.integers(1, 1_000_000)))
            t0 = time.monotonic()
            net, b, ep = train_model(torch, d, cfg, a.device, t0 + a.trial_seconds, xtr['A'], sel_x['A'], sel)
            rec = dict(trial=trial, config=cfg, selection_exact=b['score'], threshold=b['threshold'], epochs=ep, seconds=time.monotonic() - t0)
            with (out / 'trials.jsonl').open('a') as f:
                f.write(json.dumps(rec) + '\n')
            if b['score'] > best['score']:
                best = dict(score=b['score'], config=cfg, trial=trial)
            del net; gc.collect(); torch.cuda.empty_cache() if a.device == 'cuda' else None
        assert best['score'] > 0, 'search produced no valid candidate'
        body = dict(config=best['config'], search_best_selection_exact=best['score'], trial=best['trial'], frozen_utc=pd.Timestamp.now(tz='UTC').isoformat())
        body['sha256'] = hashlib.sha256(json.dumps(body['config'], sort_keys=True).encode()).hexdigest()
        write_json(frozen_path, body)
    frozen = json.loads(frozen_path.read_text()); cfg0 = frozen['config']
    # ---- stage 2: arms x seeds. Confirmation is NOT touched until all models are stored.
    models = {}
    for arm in 'AB':
        for k, s in enumerate(prereg['seeds_offsets']):
            cfg = dict(cfg0, seed=a.root_seed + s)
            t0 = time.monotonic()
            net, b, ep = train_model(torch, d, cfg, a.device, t0 + a.model_seconds, xtr[arm], sel_x[arm], sel)
            sc_all = exact_score(probabilities(torch, net, sel_x[arm]) >= b['threshold'], sel['y'], sel['mask'], sel['group'], sel['panel'], sel['cond'])
            assert abs(sc_all['overall']-b['score']) < 2e-6, 'Restored checkpoint score mismatch'
            models[(arm, s)] = dict(state=b['state'], threshold=b['threshold'], sel=sc_all, epoch=b['epoch'], cfg=cfg)
            torch.save(dict(state_dict=b['state'], cfg=cfg, mean=stats[arm][0], std=stats[arm][1], threshold=b['threshold'], arm=arm), out / f'model_{arm}_{s}.pt')
            del net; gc.collect()
    write_json(out / 'selection_results.json', {f'{k[0]}_{k[1]}': dict(sel_exact=v['sel'], threshold=v['threshold'], epoch=v['epoch']) for k, v in models.items()})
    if a.defer_confirmation:
        models_manifest={}
        for f in sorted(out.glob('model_*.pt')):
            models_manifest[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
        result=dict(status='SELECTION_COMPLETE_CONFIRMATION_SEALED',root_seed=a.root_seed,window_seed=a.window_seed,
                    cache_dir=str(cache.resolve()),model_sha256=models_manifest,confirmation_evaluated=False,
                    selection_results=json.loads((out/'selection_results.json').read_text()),submission_written=False,
                    finished_utc=pd.Timestamp.now(tz='UTC').isoformat())
        write_json(out/'result.json',result)
        write_json(Path('result.json'),result)
        print(json.dumps(dict(stage='selection_complete',confirmation_evaluated=False,models=len(models_manifest))),flush=True)
        return
    # ---- single read of the reserved confirmation partition
    conf = partition(d, 2, 'x_t'); conf_res = {}
    persistence_conf = exact_score(conf['pers'][:, None].repeat(H, 1) > 0, conf['y'], conf['mask'], conf['group'], conf['panel'], conf['cond'])
    for (arm, s), v in models.items():
        net = model_for(torch, xtr[arm].shape[1], v['cfg']).to(a.device); net.load_state_dict(v['state'])
        cx = torch.as_tensor(normalised(conf['x'], *stats[arm]), device=a.device)
        conf_res[f'{arm}_{s}'] = exact_score(probabilities(torch, net, cx) >= v['threshold'], conf['y'], conf['mask'], conf['group'], conf['panel'], conf['cond'])
    write_json(out / 'confirmation.json', dict(models=conf_res, persistence=persistence_conf))
    verdict = decide(prereg, persistence_sel, models, conf_res)
    write_json(out / 'verdict.json', verdict)
    print(json.dumps(verdict), flush=True)


def decide(prereg, persistence_sel, models, conf_res):
    acc = prereg['acceptance']; offs = prereg['seeds_offsets']
    selA = [models[('A', s)]['sel']['overall'] for s in offs]; selB = [models[('B', s)]['sel']['overall'] for s in offs]
    confA = [conf_res[f'A_{s}']['overall'] for s in offs]; confB = [conf_res[f'B_{s}']['overall'] for s in offs]
    p = persistence_sel['overall']
    validity = all(x - p >= acc['replica_validity_min_gain_over_persistence'] for x in selA)
    dsel = [b - a_ for a_, b in zip(selA, selB)]; dconf = [b - a_ for a_, b in zip(confA, confB)]
    h1 = (np.mean(dsel) >= acc['h1_min_mean_selection_gain'] and sum(x > 0 for x in dsel) >= acc['h1_min_positive_seeds']
          and np.mean(dconf) > acc['h1_min_mean_confirmation_gain'])
    return dict(replica_valid=bool(validity), h1_testlike_history_accepted=bool(validity and h1),
                selection_A=selA, selection_B=selB, selection_delta_B_minus_A=dsel, confirmation_A=confA, confirmation_B=confB,
                confirmation_delta_B_minus_A=dconf, selection_persistence=p, note='Acceptance rule is mechanical; no submission was written.')


if __name__ == '__main__':
    main()
