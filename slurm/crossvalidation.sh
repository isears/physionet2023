#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output ./logs/crossvalidation.log


export PYTHONUNBUFFERED=TRUE

python physionet2023/modeling/evaluation/encoderCV.py pretrained