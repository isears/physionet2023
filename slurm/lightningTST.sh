#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output ./logs/lightningTST.log


export PYTHONUNBUFFERED=TRUE

python physionet2023/modeling/lightningTST.py