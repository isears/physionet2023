#!/bin/bash
#SBATCH -n 1
#SBATCH -p batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output ./logs/notebook.log


export PYTHONUNBUFFERED=TRUE

jupyter notebook --no-browser --NotebookApp.allow_origin='*' --ip=0.0.0.0