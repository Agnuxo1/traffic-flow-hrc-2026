# IEEE Big Data Traffic Flow Bench 2026 — Final-Candidate Evidence

**Team:** Agnuxo1 (Francisco Angulo de Lafuente)  
**Competition:** `2026-ieee-big-data-traffic-flow-bench`  
**Prepared:** 11 November 2026, before the final code/report deadline  
**Licence:** Original project: Apache-2.0; upstream TrafficFlowBench helper: its own MIT licence.

## Why this package exists

The original public implementation and technical report described the 29 October
baseline (`sub_v1.zip`, Kaggle ID **57763508**). Later competition submissions
were accepted by Kaggle; those later frozen candidates have different queue
decisions and cannot be represented as the initial baseline. This branch
preserves the corresponding additional source and the frozen S checkpoints
without changing the original main branch.

| Kaggle ID | File | Public score | Private score | Interpretation |
| --- | --- | ---: | ---: | --- |
| 57763508 | `sub_v1.zip` | 0.72625 | 0.73170 | Initial v1 |
| 58062952 | `sub_S.zip` | **0.73684** | **0.74291** | Best recorded private score; frozen S ensemble |
| 58063013 | `sub_F.zip` | 0.72512 | 0.72194 | Frozen F control |
| 58082051 | `sub_final.zip` | 0.72113 | 0.72513 | Latest submitted DA68-arm candidate |

These results come from the user's authenticated Kaggle submission history,
not from independent recomputation. **Which candidate was selected as the
official final submission has not been independently verified.** Neither
the best private score nor a Kaggle accepted status proves award eligibility.

## Contents

- `S_selector/`: byte-preserved frozen S selector source, configuration,
  test suite and verification programs from the original experiment.
- `weights/S/`: the three frozen S checkpoints. Their SHA-256 hashes have
  been verified against `S_selector/config/final.json`.
- `reproduce_s.py`: one portable entry point for frozen S postprocessing.
  It verifies model and base hashes, reconstructs the S queue predictions
  from the official masked release, preserves non-queue cells of the v1
  submission, creates CSV/ZIP outputs and writes a machine-readable receipt.
- `DA68_final/`: original DA68 finalizer source, preserved for provenance.
  This folder is archival evidence, **not** a fully self-contained replay:
  its separately frozen inputs, checkpoints and historical execution
  conditions are not packaged here.

## Reproduce S from authorized inputs

Requirements: Python 3.13, PyTorch, NumPy, pandas and PyArrow installed
using versions compatible with the frozen model checkpoints. The official
`kaggle_public` release and the **exact frozen v1 submission CSV** must be
available locally under their competition permissions. The v1 base file is
approximately 255 MB and is deliberately not redistributed here.

On the original authorized server, the command is:

```bash
python code/award_20261111/reproduce_s.py \
  --release /home/ai/proyectos/traffic-flow-bench/data/kaggle_public \
  --base /mnt/datos/competition-campaigns/traffic-umud-20261105/inputs/traffic/sub_v1.csv \
  --output output/submission.csv
```

On another machine, use the equivalent authorized local paths with the same
two arguments. The base CSV must have SHA-256
`c3c6eeaabdcc1335480f33fae9cc37a0ec982f6f428f273bc256034bc2caebb0`;
otherwise the entry script refuses to produce a potentially misleading result.

The original archived `sub_S.zip` contains one `submission.csv`
(266,611,504 bytes), SHA-256
`d5c8c9a9d420453c0c117ba7e905ed1f7613acec2c8498896b36636026bcf425`.
The archive itself has SHA-256
`96df91f79825f8648fcd1d3d4c6afe4ba9208ecbd0a620ae3eb1f808b1d3cad9`.
For exact replay, compare the produced CSV checksum against the recorded
reference; output ZIP bytes can differ by metadata or compression settings.

The historical source includes an expired 7 November dispatch-time guard. The
portable replay invokes the unchanged inference and merge functions directly
rather than invoking that time-bound job dispatcher. No training, score tuning,
Kaggle submission, authorization step or final-choice action is performed.

## Reproducibility boundaries and safeguards

1. The models and original source are preserved and their checksums verified.
2. The authorized competition data release and the original v1 base file are
   external prerequisites. Exact, independent end-to-end reproduction and
   selection of the final Kaggle candidate are not yet certified.
3. The original experiment tests cover invariants and causal window alignment,
   but historical time guards and platform-dependent newline handling can
   affect some tests after the dispatch window or on Windows.
4. The original source's operational guards and its notes on historical
   calibration and confirmation populations remain part of the evidence.
   They must not be interpreted as independent blind confirmation.
5. No model weights, data, results or experimental evidence are represented as
   validated beyond the explicitly recorded checks. No claims about operational
   speed, traffic performance, energy or environmental savings are made.

## Author and provenance

Francisco Angulo de Lafuente  
ORCID: https://orcid.org/0009-0001-1634-7063  
Repository: https://github.com/Agnuxo1/traffic-flow-hrc-2026

MIT-licensed upstream official helper code, where used by the initial v1
pipeline, remains available from
https://github.com/jacky850/trafficflowbench-public and retains the
organizers' licence and attribution. The submitted S code uses the official
released data and should be evaluated under the organizers' competition rules.
