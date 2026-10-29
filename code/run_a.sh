#!/bin/bash
# Job A: Task 4 (two variants) + Task 1 holdout + full prediction (seed 0)
set -e
export PYTHONUNBUFFERED=1
/home/ai/proyectos/traffic-flow-bench/.venv/bin/python task4.py --source own --out results
/home/ai/proyectos/traffic-flow-bench/.venv/bin/python task4.py --source train --out results
/home/ai/proyectos/traffic-flow-bench/.venv/bin/python task1.py --seed 0 --sample 400000 --predict --out results
