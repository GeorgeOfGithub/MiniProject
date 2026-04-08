#!/bin/bash
#BSUB -J python
#BSUB -q hpc
#BSUB -W 15
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -o batch_output/python_%J.out
#BSUB -e batch_output/python_%J.err

# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

# Run Python script
time python dynamic_scheduling.py 10