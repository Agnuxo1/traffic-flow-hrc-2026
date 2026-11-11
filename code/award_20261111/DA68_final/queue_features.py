# Extracted existing queue features; no LightGBM or reservoir dependency.
import numpy as np
from stream_loader import load_panel
T=288
H=6
NB=[-3,-2,-1,1,2,3]
PANELS2=["D12_I5_N","D12_I5_S","D7_I10_E","D7_I10_W","D7_I210_E","D7_I210_W","D7_I405_N","D7_I405_S"]


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
