#!/bin/bash
# Job B: Task 2 with and without the Holographic Reservoir block (seed $1), with prediction
set -e
export PYTHONUNBUFFERED=1
S=${1:-0}
/home/ai/proyectos/traffic-flow-bench/.venv/bin/python task2.py --seed $S --per-day 1 --hrc 0 --predict --out results
/home/ai/proyectos/traffic-flow-bench/.venv/bin/python task2.py --seed $S --per-day 1 --hrc 32 --predict --out results
