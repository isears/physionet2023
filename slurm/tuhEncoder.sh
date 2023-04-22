#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output ./logs/tuhEncoder.log


export PYTHONUNBUFFERED=TRUE

python physionet2023/modeling/encoders/tuhSpectrogramAutoencoder.py