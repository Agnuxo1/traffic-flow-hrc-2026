"""Task 1: masked state reconstruction.

Control A: historical profile (official-style baseline).
Control B: profile + spatio-temporal residual interpolation (no learning).
Model:     LightGBM on residual features, incl. a vehicle-conservation flow estimate
           (upstream flow + net ramp flow), trained on train blanked cells.
Validation: temporal holdout = last HOLD train days (mimics a future month).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb_common import PANELS, REL, T, load_panel

FAMILY = lambda p: p.rsplit("_", 1)[0]


def ffill_idx(valid, axis):
    """Index of last valid position along axis (or -1)."""
    n = valid.shape[axis]
    shape = [1] * valid.ndim
    shape[axis] = n
    ar = np.arange(n).reshape(shape)
    idx = np.where(valid, ar, -1)
    return np.maximum.accumulate(idx, axis=axis)


def bfill_idx(valid, axis):
    n = valid.shape[axis]
    rev = np.flip(valid, axis=axis)
    i = ffill_idx(rev, axis)
    i = np.where(i >= 0, n - 1 - i, -1)
    return np.flip(i, axis=axis)


def take(a, idx, axis):
    safe = np.where(idx >= 0, idx, 0)
    v = np.take_along_axis(a, safe, axis=axis)
    return np.where(idx >= 0, v, np.nan)


def profiles(P, train_days_mask):
    """Median profile per (daytype, link, tod) from masked-observed train values."""
    out = {}
    wk = P["dow"] < 5
    for ch in ("s", "qpl"):
        prof = np.zeros((2,) + P[ch].shape[1:], dtype=np.float32)
        for dt, sel in ((0, wk), (1, ~wk)):
            sel2 = sel & train_days_mask
            if sel2.sum() == 0:
                sel2 = train_days_mask
            with np.errstate(all="ignore"):
                prof[dt] = np.nanmedian(P[ch][sel2], axis=0)
        # fill any remaining NaN by link mean
        lm = np.nanmean(prof, axis=2, keepdims=True)
        prof = np.where(np.isfinite(prof), prof, lm)
        out[ch] = np.nan_to_num(prof)
    return out


def features(P, prof):
    """Feature tensor [D, L, T, F] (float32) and names."""
    D, L, _ = P["s"].shape
    dt = (P["dow"] >= 5).astype(int)
    feats, names = [], []
    tod = np.broadcast_to(np.arange(T)[None, None, :], (D, L, T)).astype(np.float32)
    feats += [tod, np.broadcast_to(dt[:, None, None], (D, L, T)).astype(np.float32),
              np.broadcast_to(P["lanes"][None, :, None], (D, L, T)),
              np.broadcast_to(P["vf"][None, :, None], (D, L, T)),
              np.broadcast_to((np.arange(L) / max(L - 1, 1))[None, :, None], (D, L, T)).astype(np.float32)]
    names += ["tod", "weekend", "lanes", "vf", "pos"]
    for ch in ("s", "qpl"):
        X = P[ch]
        pr = prof[ch][dt]  # [D,L,T]
        R = X - pr
        valid = np.isfinite(R)
        ip = ffill_idx(valid, 2); inx = bfill_idx(valid, 2)
        rp = take(R, ip, 2); rn = take(R, inx, 2)
        t = np.arange(T)[None, None, :]
        gp = np.where(ip >= 0, t - ip, 99).astype(np.float32)
        gn = np.where(inx >= 0, inx - t, 99).astype(np.float32)
        wp = 1.0 / gp; wn = 1.0 / gn
        rl = (np.nan_to_num(rp) * wp * np.isfinite(rp) + np.nan_to_num(rn) * wn * np.isfinite(rn)) / (
            wp * np.isfinite(rp) + wn * np.isfinite(rn) + 1e-9)
        iu = ffill_idx(valid, 1); idn = bfill_idx(valid, 1)
        l = np.arange(L)[None, :, None]
        ru = take(R, iu, 1); rd = take(R, idn, 1)
        xu = take(X, iu, 1); xd = take(X, idn, 1)
        du = np.where(iu >= 0, l - iu, 99).astype(np.float32)
        dd = np.where(idn >= 0, idn - l, 99).astype(np.float32)
        with np.errstate(all="ignore"):
            rtm = np.nanmean(R, axis=1, keepdims=True)
            rday = np.nanmean(R, axis=2, keepdims=True)
        rtm = np.broadcast_to(rtm, (D, L, T)); rday = np.broadcast_to(rday, (D, L, T))
        # neighbours at t-1 / t+1 for up/down links (shock propagation)
        ru_m1 = np.roll(ru, 1, axis=2); rd_p1 = np.roll(rd, -1, axis=2)
        ru_p1 = np.roll(ru, -1, axis=2); rd_m1 = np.roll(rd, 1, axis=2)
        feats += [pr, rp, rn, gp, gn, rl, ru, rd, xu, xd, du, dd, rtm, rday, ru_m1, rd_p1, ru_p1, rd_m1]
        names += [f"{ch}_{n}" for n in ["prof", "rprev", "rnext", "gprev", "gnext", "rlin", "rup", "rdn",
                                         "xup", "xdn", "dup", "ddn", "rtmean", "rday", "rup_m1", "rdn_p1",
                                         "rup_p1", "rdn_m1"]]
    # vehicle conservation: q(link) ~ q(upstream sensored) + net ramp of this link (per lane)
    q = P["q"]
    valid = np.isfinite(q)
    iu = ffill_idx(valid, 1); idn = bfill_idx(valid, 1)
    qu = take(q, iu, 1); qd = take(q, idn, 1)
    nr = P["net_ramp"]
    lanes = P["lanes"][None, :, None]
    cons_up = (qu + nr) / lanes
    nr_d = take(nr, idn, 1)
    cons_dn = (qd - np.nan_to_num(nr_d)) / lanes
    feats += [cons_up, cons_dn, nr / lanes, P["ramp_ok"].astype(np.float32)]
    names += ["cons_up", "cons_dn", "netramp_pl", "ramp_ok"]
    F = np.stack([np.asarray(f, dtype=np.float32) for f in feats], axis=-1)
    return F, names


DAYKEYS = ("s", "q", "qpl", "s_true", "q_true", "qpl_true", "elig", "present", "net_ramp", "ramp_ok", "dow", "regime", "elig_masked")


def sub(P, sl):
    Q = dict(P)
    for k in DAYKEYS:
        if k in P:
            Q[k] = P[k][sl]
    return Q


def chunks(D, n=40):
    for a in range(0, D, n):
        yield slice(a, min(D, a + n))


def prep(P):
    P["qpl"] = P["q"] / P["lanes"][None, :, None]
    if "q_true" in P:
        P["qpl_true"] = P["q_true"] / P["lanes"][None, :, None]
    return P


def targets_mask(P):
    blank = P["present"] & ~np.isfinite(P["s"]) & ~np.isfinite(P["q"])
    return blank


def score(rows):
    """rows: list of dict(panel, regime, se_s, se_q, n). Returns S_state with family aggregation."""
    df = pd.DataFrame(rows)
    g = df.groupby(["panel", "regime"]).sum(numeric_only=True).reset_index()
    g["S"] = 0.54 * np.maximum(0, 1 - np.sqrt(g.se_s / g.n) / 25) + 0.46 * np.maximum(0, 1 - np.sqrt(g.se_q / g.n) / 600)
    pan = g.groupby("panel").S.mean()
    fam = pan.groupby(pan.index.map(FAMILY)).mean()
    return float(fam.mean()), pan.to_dict(), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="all")
    ap.add_argument("--hold", type=int, default=45)
    ap.add_argument("--sample", type=int, default=400000, help="training cells per panel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cons", action="store_true", help="ablation: drop conservation features")
    ap.add_argument("--predict", action="store_true", help="refit on all train and write val/private rows")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    panels = PANELS if args.panels == "all" else args.panels.split(",")
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    Xtr, ys, yq, Xho_info = [], [], [], []
    cache = {}
    for p in panels:
        P = prep(load_panel(p, "train", truth=True))
        D = len(P["days"])
        trm = np.arange(D) < D - args.hold
        prof = profiles(P, trm)
        dtall = (P["dow"] >= 5).astype(int)
        for sl in chunks(D):
            Pc = sub(P, sl)
            F, names = features(Pc, prof)
            tm = targets_mask(Pc) & Pc["elig"] & np.isfinite(Pc["s_true"]) & np.isfinite(Pc["q_true"])
            dt = dtall[sl]
            ps = prof["s"][dt]; pq = prof["qpl"][dt]
            trc = trm[sl]
            idx = np.argwhere(tm & trc[:, None, None])
            frac = args.sample / max(1, int((targets_mask(P) & trm[:, None, None]).sum()))
            idx = idx[rng.random(len(idx)) < frac]
            d, l, t = idx.T
            Xtr.append(F[d, l, t]); ys.append(Pc["s_true"][d, l, t] - ps[d, l, t]); yq.append(Pc["qpl_true"][d, l, t] - pq[d, l, t])
            hidx = np.argwhere(tm & ~trc[:, None, None])
            hidx = hidx[rng.random(len(hidx)) < 0.35]
            if len(hidx):
                hi = tuple(hidx.T)
                Fh = F[hi]
                c = cache.setdefault(p, {k: [] for k in ("F", "ps", "pq", "st", "qt", "reg")})
                c["F"].append(Fh); c["ps"].append(ps[hi]); c["pq"].append(pq[hi])
                c["st"].append(Pc["s_true"][hi]); c["qt"].append(Pc["qpl_true"][hi]); c["reg"].append(Pc["regime"][hidx[:, 0]])
            del F
        c = cache[p]
        for k in list(c):
            c[k] = np.concatenate(c[k])
        c["rl_s"] = c["F"][:, names.index("s_rlin")]; c["rl_q"] = c["F"][:, names.index("qpl_rlin")]
        idx = np.zeros(sum(len(x) for x in Xtr)); hidx = c["F"]
        print(f"{p}: D={D} L={len(P['links'])} train_cells={len(idx)} hold_cells={len(hidx)} t={time.time()-t0:.0f}s", flush=True)
        del P
    keep = [i for i, n in enumerate(names) if not (args.no_cons and n in ("cons_up", "cons_dn", "netramp_pl", "ramp_ok"))]
    X = np.concatenate(Xtr)[:, keep]; YS = np.concatenate(ys); YQ = np.concatenate(yq)
    params = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  seed=args.seed, num_threads=4, verbose=-1)
    ms = lgb.train(params, lgb.Dataset(X, YS), num_boost_round=600)
    mq = lgb.train(params, lgb.Dataset(X, YQ), num_boost_round=600)
    rows = {"profile": [], "interp": [], "lgbm": []}
    for p, c in cache.items():
        Fh = c["F"][:, keep]
        preds = {
            "profile": (c["ps"], c["pq"]),
            "interp": (c["ps"] + np.nan_to_num(c["rl_s"]), c["pq"] + np.nan_to_num(c["rl_q"])),
            "lgbm": (c["ps"] + ms.predict(Fh), c["pq"] + mq.predict(Fh)),
        }
        for k, (sp, qp) in preds.items():
            sp = np.clip(sp, 0, 140); qp = np.clip(qp, 0, 3000)
            for r in ("R1", "R2", "R3"):
                sel = c["reg"] == r
                rows[k].append(dict(panel=p, regime=r, se_s=float(((sp - c["st"])[sel] ** 2).sum()),
                                    se_q=float(((qp - c["qt"])[sel] ** 2).sum()), n=int(sel.sum())))
    res = {}
    for k, rr in rows.items():
        s, pan, g = score(rr)
        res[k] = {"S_state": s, "panels": pan}
        g.to_csv(out / f"holdout_{k}.csv", index=False)
        print(f"HOLDOUT {k}: S_state={s:.5f}", flush=True)
    imp = dict(zip([names[i] for i in keep], ms.feature_importance("gain").round(0).tolist()))
    res["importance_speed"] = imp
    res["args"] = vars(args)
    (out / f"task1_holdout_seed{args.seed}{'_nocons' if args.no_cons else ''}.json").write_text(json.dumps(res, indent=1))
    if args.predict:
        predict_all(panels, keep, names, params, args, out, rng)
    print(f"done {time.time()-t0:.0f}s")


def predict_all(panels, keep, names, params, args, out, rng):
    Xtr, ys, yq = [], [], []
    store = {}
    for p in panels:
        P = prep(load_panel(p, "train", truth=True))
        D = len(P["days"])
        prof = profiles(P, np.ones(D, bool))
        dtall = (P["dow"] >= 5).astype(int)
        frac = args.sample / max(1, int(targets_mask(P).sum()))
        for sl in chunks(D):
            Pc = sub(P, sl)
            F, _ = features(Pc, prof)
            tm = targets_mask(Pc) & Pc["elig"] & np.isfinite(Pc["s_true"]) & np.isfinite(Pc["q_true"])
            dt = dtall[sl]
            idx = np.argwhere(tm); idx = idx[rng.random(len(idx)) < frac]
            d, l, t = idx.T
            Xtr.append(F[d, l, t][:, keep]); ys.append(Pc["s_true"][d, l, t] - prof["s"][dt][d, l, t])
            yq.append(Pc["qpl_true"][d, l, t] - prof["qpl"][dt][d, l, t])
            del F
        store[p] = prof
        del P
    ms = lgb.train(params, lgb.Dataset(np.concatenate(Xtr), np.concatenate(ys)), num_boost_round=600)
    mq = lgb.train(params, lgb.Dataset(np.concatenate(Xtr), np.concatenate(yq)), num_boost_round=600)
    parts = []
    for p in panels:
        for split in ("validation", "private"):
            P = prep(load_panel(p, split))
            prof = store[p]
            F, _ = features(P, prof)
            dt = (P["dow"] >= 5).astype(int)
            tmpl = pd.read_csv(REL / "task1" / p / split / "sample_submission_state.csv")
            ts = pd.to_datetime(tmpl.timestamp, utc=True)
            di = tmpl.timestamp.str[:10].map({d: i for i, d in enumerate(P["days"])}).to_numpy()
            li = tmpl.link_id.map({l: i for i, l in enumerate(P["links"])}).to_numpy()
            ti = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
            ok = ~(pd.isna(di) | pd.isna(li))
            di = np.where(ok, di, 0).astype(int); li = np.where(ok, li, 0).astype(int)
            Fx = F[di, li, ti][:, keep]
            sp = prof["s"][dt][di, li, ti] + ms.predict(Fx)
            qp = (prof["qpl"][dt][di, li, ti] + mq.predict(Fx)) * P["lanes"][li]
            tmpl["speed_kmh"] = np.clip(sp, 1, 140).round(3)
            tmpl["flow_vph"] = np.clip(qp, 0, None).round(2)
            parts.append(tmpl)
            print(f"pred {p} {split} rows={len(tmpl)} unmapped={int((~ok).sum())}", flush=True)
            del F, P
    pd.concat(parts).to_csv(out / f"state_submission_seed{args.seed}.csv", index=False)


if __name__ == "__main__":
    main()
