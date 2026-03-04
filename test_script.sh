#!/bin/bash
#SBATCH --job-name=safe-rs-bench
#SBATCH --partition=cpu
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=outputs/%j/safe_rs_benchmark.log

.venv/bin/python example.py
