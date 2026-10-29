# TrafficFlowBench — Agnuxo1 entry (IEEE Big Data Cup 2026, Track 2)

Single-entry reproduction: `bash run_a.sh` (Task 4 + Task 1) and `bash run_b.sh 0` (Task 2), then
`bash make_submission.sh results/state_submission_seed0.csv results/queue_submission_seed0_hrc<N>.csv results/odme_own_lam0.05.csv submission.csv`.
Set `TFB_REL` to the unpacked `kaggle_public/` folder. Environment: Python 3.13, numpy 2.5.4, pandas 3.1.0,
lightgbm 4.6.0, scipy 1.18.1, pyarrow; CPU only (4 threads). Seeds fixed by `--seed`.

## Method
- **Task 1 (state):** per-link weekday/weekend time-of-day median profile from `train`; residual features from the
  same day (nearest observed value before/after in time, nearest observed upstream/downstream sensor at t, t±1,
  corridor-wide residual at t, day mean), plus a **vehicle-conservation estimate** of flow
  (upstream sensored flow + net on/off-ramp flow of the link, per lane). LightGBM regresses the speed and
  per-lane-flow residuals. Trained on blanked, eligible `train` cells; validated on the last 45 train days.
- **Task 2 (queues):** LightGBM classifier over (link, horizon) with masked-history features of the link and
  ±3 corridor neighbours and distances to the nearest queued link at T and T-15. Optional block: the
  **Holographic Reservoir** (Agnuxo1/Holographic-Reservoir, echo-state reservoir) driven by the 13-step
  neighbourhood speed-ratio sequence; its final state is added as features. Ablation reported with/without.
  Training windows sampled from `train` with the official onset/ongoing rules; threshold tuned on holdout IoU.
- **Task 3:** no file; scored on Task 1 rows (conservation features are the lever).
- **Task 4 (ODME):** non-negative prior-regularised least squares (official operator and solver form) on each
  scored split's own released link counts and weak prior, lambda = 0.05.

No external data. All third-party code: official TrafficFlowBench repo (merge script, Task 4 helpers), license in `official_src`.
