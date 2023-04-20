#!/bin/bash
#SBATCH -n 1
#SBATCH -p bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output ./logs/tuhcache.log


export PYTHONUNBUFFERED=TRUE

python physionet2023/dataProcessing/buildTuhPsdCache.py