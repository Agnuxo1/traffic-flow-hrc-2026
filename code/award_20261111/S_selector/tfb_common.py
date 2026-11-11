"""Shared loaders for TrafficFlowBench: per-day link x time arrays for one panel and split."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

REL = Path(os.environ.get("TFB_REL", "/home/ai/proyectos/traffic-flow-bench/data/kaggle_public"))
PANELS = ["D12_I405_N", "D12_I405_S", "D12_I5_N", "D12_I5_S", "D7_I10_E",
          "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]
T = 288  # five-minute steps per day


def network(panel: str) -> dict:
    nd = REL / "corridors" / panel / "network"
    topo = pd.read_csv(nd / "lwr_mainline_topology.csv").sort_values("order_index")
    links = pd.read_csv(nd / "links.csv")
    return {"topo": topo, "links": links.set_index("link_id")}


def _read(paths, cols):
    out = []
    for p in paths:
        out.append(pd.read_parquet(p, columns=cols))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def load_panel(panel: str, split: str, truth: bool = False) -> dict:
    """Return arrays [D, L, T] for masked (and optionally true) speed/flow plus metadata.

    Links are the sensored mainline links in corridor order (order_index).
    """
    base = REL / "corridors" / panel / split
    cols = ["date", "timestamp", "station_id", "link_id", "speed_kmh", "flow_vph",
            "pct_observed", "is_score_eligible"]
    mfiles = sorted(glob.glob(str(base / "mainline_states_masked" / "mask_regime=*" / "*.parquet")))
    regime_of_day = {}
    frames = []
    for p in mfiles:
        reg = Path(p).parent.name.split("=")[1]
        d = pd.read_parquet(p, columns=cols)
        regime_of_day[str(d.date.iloc[0])] = reg
        frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    net = network(panel)
    order = [l for l in net["topo"].link_id if l in set(m.link_id.unique())]
    lidx = {l: i for i, l in enumerate(order)}
    days = sorted(m.date.astype(str).unique())
    didx = {d: i for i, d in enumerate(days)}
    station_of = m.drop_duplicates("link_id").set_index("link_id").station_id.to_dict()
    D, L = len(days), len(order)

    def to_arr(frame, col, dtype=np.float32, fill=np.nan):
        a = np.full((D, L, T), fill, dtype=dtype)
        ts = pd.to_datetime(frame.timestamp, utc=True)
        ti = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
        di = frame.date.astype(str).map(didx).to_numpy()
        li = frame.link_id.map(lidx).to_numpy()
        a[di, li, ti] = frame[col].to_numpy()
        return a

    out = {
        "panel": panel, "split": split, "days": days, "links": order, "station_of": station_of,
        "regime": np.array([regime_of_day[d] for d in days]),
        "s": to_arr(m, "speed_kmh"), "q": to_arr(m, "flow_vph"),
        "elig_masked": to_arr(m, "is_score_eligible", np.float32, 0),
    }
    present = to_arr(m.assign(one=1.0), "one", np.float32, 0)
    out["present"] = present > 0
    if truth:
        tfiles = sorted(glob.glob(str(base / "mainline_states" / "*" / "*.parquet")))
        t = _read(tfiles, cols)
        t = t[t.link_id.isin(lidx)]
        out["s_true"] = to_arr(t, "speed_kmh")
        out["q_true"] = to_arr(t, "flow_vph")
        out["elig"] = to_arr(t, "is_score_eligible", np.float32, 0) > 0
    # ramps: net on-ramp minus off-ramp flow attached to each sensored link (NaN if a ramp is missing)
    rfiles = sorted(glob.glob(str(base / "ramp_states" / "*" / "*.parquet")))
    r = _read(rfiles, ["date", "timestamp", "ramp_link_id", "ramp_type", "flow_vph", "is_score_eligible"])
    topo = net["topo"].set_index("link_id")
    net_ramp = np.zeros((D, L, T), dtype=np.float32)
    ramp_ok = np.ones((D, L, T), dtype=bool)
    if len(r):
        r = r.copy()
        r.loc[r.is_score_eligible.astype(int) == 0, "flow_vph"] = np.nan
        ts = pd.to_datetime(r.timestamp, utc=True)
        r["ti"] = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
        r["di"] = r.date.astype(str).map(didx)
        r = r.dropna(subset=["di"])
        rid = {k: g for k, g in r.groupby("ramp_link_id")}
        for l, li in lidx.items():
            for colname, sign in (("on_ramp_link_ids", 1.0), ("off_ramp_link_ids", -1.0)):
                v = topo.at[l, colname] if l in topo.index else np.nan
                if isinstance(v, str):
                    for rr in v.split(";"):
                        g = rid.get(rr)
                        if g is None:
                            ramp_ok[:, li, :] = False
                            continue
                        a = np.full((D, T), np.nan, dtype=np.float32)
                        a[g.di.astype(int).to_numpy(), g.ti.to_numpy()] = g.flow_vph.to_numpy()
                        ramp_ok[:, li, :] &= np.isfinite(a)
                        net_ramp[:, li, :] += sign * np.nan_to_num(a)
    out["net_ramp"] = net_ramp
    out["ramp_ok"] = ramp_ok
    links = net["links"]
    out["lanes"] = np.array([float(links.at[l, "lanes"]) for l in order], dtype=np.float32)
    out["vf"] = np.array([float(links.at[l, "free_speed_kmh"]) for l in order], dtype=np.float32)
    out["length"] = np.array([float(links.at[l, "length_km"]) for l in order], dtype=np.float32)
    dates = pd.to_datetime(pd.Series(days))
    out["dow"] = dates.dt.dayofweek.to_numpy()
    return out
