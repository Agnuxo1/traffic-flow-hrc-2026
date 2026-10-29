"""Task 2: 30-minute queue forecasting from 60 minutes of history (+ the row at T).

Controls: persistence at T-5 (official baseline) and persistence at T.
Model:    LightGBM classifier over link x horizon cells, features from the masked history of the
          link and its corridor neighbours. Optional own-technology block: Holographic Reservoir
          (echo-state reservoir from Agnuxo1/Holographic-Reservoir, SHA-256 seeded) driven by the
          13-step neighbourhood speed-ratio sequence; its final state enters as features (--hrc).
Training windows are drawn from train days with the official selection rules (onset / ongoing,
at least one queued cell in the horizon); labels from the unmasked train layer.
Validation: last HOLD train days; metric = mean window IoU on eligible cells.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from hrc.model import HolographicReservoirModel
from task1 import bfill_idx, ffill_idx, take
from tfb_common import REL, T, load_panel

import os
PANELS2 = os.environ["TFB_P2"].split(",") if os.environ.get("TFB_P2") else ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]
H = 6
NB = [-3, -2, -1, 1, 2, 3]  # neighbour offsets (negative = upstream in corridor order)


class Reservoir:
    def __init__(self, size, input_dim, seed):
        m = HolographicReservoirModel(size=size, input_dim=input_dim, random_seed=seed, leak=0.5)
        self.W_in, self.W, self.leak = m.W_in.astype(np.float32), m.W.astype(np.float32), m.leak

    def final_state(self, U):  # U [N, steps, in]
        x = np.zeros((U.shape[0], self.W.shape[0]), np.float32)
        for k in range(U.shape[1]):
            x = (1 - self.leak) * x + self.leak * np.tanh(U[:, k] @ self.W_in.T + x @ self.W.T)
        return x


def ratio_hist(P, d, t0):
    """Speed/vf for history steps t0-12..t0 at day d, gaps filled in time then space. [L, 13]."""
    s = P["s"][d][:, t0 - 12:t0 + 1] / P["vf"][:, None]
    v = np.isfinite(s)
    s = take(s, ffill_idx(v, 1), 1)
    v = np.isfinite(s)
    s2 = take(s, bfill_idx(v, 1), 1)
    s = np.where(np.isfinite(s), s, s2)
    v = np.isfinite(s)
    up = take(s, ffill_idx(v, 0), 0); dn = take(s, bfill_idx(v, 0), 0)
    s = np.where(np.isfinite(s), s, np.where(np.isfinite(up), up, dn))
    return np.nan_to_num(s, nan=1.0).astype(np.float32)


def window_feats(P, d, t0, res=None):
    L = len(P["links"])
    R = ratio_hist(P, d, t0)  # [L,13]
    qf = R <= 0.6
    Q = P["q"][d][:, t0 - 12:t0 + 1] / P["lanes"][:, None]
    Qn = np.nan_to_num(np.nanmean(np.where(np.isfinite(Q[:, -3:]), Q[:, -3:], np.nan), axis=1), nan=-1)
    pos = np.arange(L)
    f = [R[:, -1], R[:, -2], R[:, -4], R[:, -7], R[:, 0], R.min(1), R[:, -1] - R[:, -4], Qn]
    names = ["r0", "r1", "r3", "r6", "r12", "rmin", "dr3", "qpl"]
    for o in NB:
        j = np.clip(pos + o, 0, L - 1)
        f += [R[j, -1], R[j, -4]]
        names += [f"r0_{o}", f"r3_{o}"]
    q0 = qf[:, -1]; q3 = qf[:, -4]
    qi = np.where(q0)[0]; qi3 = np.where(q3)[0]

    def dist(qidx, sign):
        out = np.full(L, 99.0, np.float32)
        if len(qidx):
            dd = (qidx[None, :] - pos[:, None]) * sign
            dd = np.where(dd >= 0, dd, 999)
            out = np.minimum(dd.min(1), 99).astype(np.float32)
        return out
    f += [dist(qi, 1), dist(qi, -1), dist(qi3, 1), dist(qi3, -1),
          np.full(L, q0.mean(), np.float32), np.full(L, qf.any(1).mean(), np.float32),
          np.full(L, float(qf[:, :-1].any()), np.float32), pos / max(L - 1, 1),
          np.full(L, t0, np.float32), np.full(L, float(P["dow"][d] >= 5), np.float32)]
    names += ["dq_dn0", "dq_up0", "dq_dn3", "dq_up3", "fracq0", "fracq_hist", "anyq_hist", "pos", "tod", "weekend"]
    if res is not None:
        U = np.stack([R[np.clip(pos + o, 0, L - 1)] for o in [-2, -1, 0, 1, 2]], axis=2)  # [L,13,5]
        st = res.final_state(U)
        f += list(st.T)
        names += [f"hrc{i}" for i in range(st.shape[1])]
    return np.stack(f, 1).astype(np.float32), names


def labels(P, d, t0):
    s = P["s_true"][d][:, t0 + 1:t0 + 1 + H]
    lab = s <= 0.6 * P["vf"][:, None]
    elig = P["elig"][d][:, t0 + 1:t0 + 1 + H] & np.isfinite(s)
    return lab, elig


def sample_windows(P, rng, per_day):
    """Train windows following the official selection rules, using the unmasked layer."""
    D = len(P["days"])
    thr = 0.6 * P["vf"][:, None]
    out = []
    for d in range(D):
        s = P["s_true"][d]
        q = (s <= thr) & P["elig"][d]
        cands = rng.permutation(np.arange(12, T - H - 1))
        n_on = n_og = 0
        for t0 in cands:
            if not q[:, t0 + 1:t0 + 1 + H].any():
                continue
            if q[:, t0].any() and n_og < per_day:
                out.append((d, t0, "queue_ongoing")); n_og += 1
            elif not q[:, t0 - 12:t0 + 1].any() and n_on < per_day:
                out.append((d, t0, "queue_onset")); n_on += 1
            if n_on >= per_day and n_og >= per_day:
                break
    return out


def iou(pred, lab, elig):
    p = pred & elig; t = lab & elig
    u = (p | t).sum()
    return 1.0 if u == 0 else (p & t).sum() / u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=45)
    ap.add_argument("--per-day", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hrc", type=int, default=0, help="reservoir size (0 = off)")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    res = Reservoir(args.hrc, 5, args.seed) if args.hrc else None
    t00 = time.time()
    Xtr, Ytr, Wtr, hold = [], [], [], []
    for p in PANELS2:
        P = load_panel(p, "train", truth=True)
        D = len(P["days"])
        for (d, t0, cond) in sample_windows(P, rng, args.per_day):
            F, names = window_feats(P, d, t0, res)
            lab, el = labels(P, d, t0)
            Fh = np.concatenate([np.repeat(F, H, 0), np.tile(np.arange(1, H + 1), len(F))[:, None]], 1)
            if d < D - args.hold:
                m = el.reshape(-1)
                Xtr.append(Fh[m]); Ytr.append(lab.reshape(-1)[m])
                Wtr.append(np.full(m.sum(), 1.0 / max(m.sum(), 1), np.float32))  # equal weight per window
            else:
                hold.append(dict(panel=p, cond=cond, F=Fh, lab=lab, el=el, r0=F[:, 0], r1=F[:, 1]))
        print(f"{p}: train rows so far {sum(len(x) for x in Xtr)} hold windows {len(hold)} t={time.time()-t00:.0f}s", flush=True)
        del P
    names = names + ["h"]
    X = np.concatenate(Xtr); Y = np.concatenate(Ytr); W = np.concatenate(Wtr) * 1000
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=args.seed,
                  num_threads=4, verbose=-1)
    model = lgb.train(params, lgb.Dataset(X, Y, weight=W), num_boost_round=400)
    probs = [model.predict(w["F"]).reshape(-1, H) for w in hold]
    res_json = {"args": vars(args), "n_hold": len(hold)}
    for name, fn in [("persist_T-5", lambda w, pr: np.repeat((w["r1"] <= 0.6)[:, None], H, 1)),
                     ("persist_T", lambda w, pr: np.repeat((w["r0"] <= 0.6)[:, None], H, 1))]:
        res_json[name] = float(np.mean([iou(fn(w, None), w["lab"], w["el"]) for w in hold]))
    best = (0, 0.5)
    for th in np.arange(0.2, 0.71, 0.05):
        sc = float(np.mean([iou(pr >= th, w["lab"], w["el"]) for w, pr in zip(hold, probs)]))
        res_json[f"lgbm_th{th:.2f}"] = sc
        best = max(best, (sc, float(th)))
    res_json["lgbm_best"] = best
    for cond in ("queue_onset", "queue_ongoing"):
        res_json[f"lgbm_best_{cond}"] = float(np.mean([iou(pr >= best[1], w["lab"], w["el"]) for w, pr in zip(hold, probs) if w["cond"] == cond]))
    print(json.dumps(res_json, indent=1), flush=True)
    tag = f"seed{args.seed}_hrc{args.hrc}"
    (out / f"task2_holdout_{tag}.json").write_text(json.dumps(res_json, indent=1))
    if args.predict:
        predict(args, model, best[1], res, out, tag)
    print(f"done {time.time()-t00:.0f}s")


def predict(args, model, th, res, out, tag):
    parts = []
    for p in PANELS2:
        for split in ("validation", "private"):
            P = load_panel(p, split)
            widx = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
            tmpl = pd.read_csv(REL / "task2" / p / split / "sample_submission_queue.csv")
            dmap = {d: i for i, d in enumerate(P["days"])}
            lmap = {l: i for i, l in enumerate(P["links"])}
            for w in widx.itertuples():
                to = pd.Timestamp(w.forecast_origin)
                t0 = to.hour * 12 + to.minute // 5
                d = dmap[str(w.date)]
                F, _ = window_feats(P, d, t0, res)
                Fh = np.concatenate([np.repeat(F, H, 0), np.tile(np.arange(1, H + 1), len(F))[:, None]], 1)
                pr = model.predict(Fh).reshape(-1, H) >= th  # [Lsensored, H]
                tw = tmpl[tmpl.window_id == w.window_id].copy()
                ts = pd.to_datetime(tw.timestamp, utc=True)
                hi = ((ts - to).dt.total_seconds() // 300).astype(int).to_numpy() - 1
                li = tw.link_id.map(lmap)
                # unsensored links: nearest sensored link in corridor order
                topo = pd.read_csv(REL / "corridors" / p / "network" / "lwr_mainline_topology.csv").sort_values("order_index")
                sens_order = [lmap.get(l) for l in topo.link_id]
                last = None; near = {}
                for l, si in zip(topo.link_id, sens_order):
                    if si is not None:
                        last = si
                    near[l] = si if si is not None else last
                first = next(s for s in sens_order if s is not None)
                li = tw.link_id.map(lambda l: near.get(l) if near.get(l) is not None else first).astype(int).to_numpy()
                tw["queue_pred"] = pr[li, np.clip(hi, 0, H - 1)].astype(int)
                parts.append(tw)
            print(f"pred {p} {split} windows={len(widx)}", flush=True)
            del P
    pd.concat(parts).to_csv(out / f"queue_submission_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
