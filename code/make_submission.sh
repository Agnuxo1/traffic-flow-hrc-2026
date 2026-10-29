#!/bin/bash
# Merge the three per-task files into the Kaggle upload (official merge script).
# usage: make_submission.sh STATE QUEUE ODME OUT
set -e
REL=${TFB_REL:-/home/ai/proyectos/traffic-flow-bench/data/kaggle_public}
HERE=$(cd "$(dirname "$0")" && pwd)
python "$HERE/official_src/merge_submissions.py" --state "$1" --queue "$2" --odme "$3" --key "$REL/submission_key.csv" --output "$4"
