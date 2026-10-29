# Traffic Flow Bench — conservation features and reservoir ablations

Implementation for [IEEE Big Data Traffic Flow Bench](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench), combining masked-state reconstruction, queue prediction and OD estimation with the author's [Holographic Reservoir](https://github.com/Agnuxo1/Holographic-Reservoir).

## Current evidence

The CPU control for queue prediction has a campaign-recorded holdout IoU of approximately0.667, compared with persistence approximately0.41. The32-unit reservoir comparison and state-reconstruction runs are still incomplete. The OD solution uses non-negative prior-regularised least squares with lambda0.05. These are exploratory local records, not independently verified official scores or a demonstrated reservoir gain. Detailed logs remain in [experiments](experiments).

The historical server jobs were cancelled by their owner and work continued on the local CPU; this publication does not restart or cancel any job. Prize eligibility is unresolved because the separate organiser-email registration deadline may have passed. Kaggle membership alone is not proof of that registration or prize eligibility.

## Reproduce

```bash
python -m pip install -r code/requirements.txt
cd code
export TFB_REL=/path/to/kaggle_public
bash run_a.sh
bash run_b.sh 0
```

`run_a.sh` produces Task4 and Task1 results; `run_b.sh` produces Task2 with a specified reservoir size. The final merge command and exact inputs are in [implementation README](code/README.md). The current scripts use four CPU threads and fixed seeds. They require the authorised competition release; **data are not included**. The release is derived from calibrated simulated traffic, not a claim that every cell is a directly observed road measurement.

## Scientific comparison

Freeze chronological validation windows and official masks, match compute, then compare persistence, the feature-based control, and reservoir features with at least three seeds. Conservation features and reservoir states need separate ablations. Keep all subtasks and errors visible; a positive queue-control result does not prove an aggregate competition improvement.

Original contributions and the imported Holographic Reservoir retain Apache-2.0. Official helper code is not relicensed: its provenance and upstream terms are recorded in `code/official_src/`. See [research protocol](docs/RESEARCH_PROTOCOL.md). No official acceptance, rank or prize is claimed.
