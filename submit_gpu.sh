#!/bin/bash
#BSUB -J python
#BSUB -q c02613
#BSUB -W 10
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o batch_output/python_%J.out
#BSUB -e batch_output/python_%J.err

# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

# Run Python script
