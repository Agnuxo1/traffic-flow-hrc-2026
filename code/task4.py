"""Task 4: prior-regularised ODME per split, using each split's own released counts and prior.

Uses the official operator/solver helpers (official/src/task4). --source train reproduces the
official naive baseline (train counts); --source own uses the scored split's released counts.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import lsq_linear
import scipy.sparse as sp
from tfb_common import PANELS, REL
sys.path.insert(0, str(Path(__file__).resolve().parent / "official_src"))
from task4.build_task4_odme_artifacts import load_operator, prior_values, released_counts, released_prior

ap = argparse.ArgumentParser()
ap.add_argument("--lam", type=float, default=0.05)
ap.add_argument("--source", choices=["own", "train"], default="own")
ap.add_argument("--out", default="out")
a = ap.parse_args()
parts = []
for p in PANELS:
    path_ids, link_ids, A, paths = load_operator(REL / "corridors" / p / "network")
    for split in ("validation", "private"):
        base = prior_values(REL, p, split, path_ids)
        cs = released_counts(REL, p, split)
        src = released_counts(REL, p, "train") if a.source == "train" else cs
        meas = np.array([l in set(cs.link_id) for l in link_ids])
        c = src.set_index("link_id").reindex(link_ids).fillna(0.0)["count"].to_numpy(float)
        # same objective as the official nnls solve: ||A f - c||^2 + lam ||f - b||^2, f >= 0 (sparse)
        aa = sp.vstack([sp.csr_matrix(A[meas]), np.sqrt(a.lam) * sp.identity(A.shape[1], format="csr")]).tocsr()
        bb = np.concatenate([c[meas], np.sqrt(a.lam) * base])
        f = lsq_linear(aa, bb, bounds=(0, np.inf), lsmr_tol="auto", tol=1e-10, max_iter=5000).x
        pr = released_prior(REL, p, split)
        x = paths[["path_id", "origin_zone", "destination_zone"]].copy()
        x["panel"] = p; x["departure_time"] = str(pr.departure_time.iloc[0]); x["path_flow"] = f.round(4)
        resid = A[meas] @ f - c[meas]
        print(f"{p} {split} paths={len(f)} link_rmse={np.sqrt((resid**2).mean()):.1f} dev={np.abs(f-base).mean():.2f}", flush=True)
        parts.append(x)
Path(a.out).mkdir(parents=True, exist_ok=True)
pd.concat(parts).to_csv(Path(a.out) / f"odme_{a.source}_lam{a.lam}.csv", index=False)
