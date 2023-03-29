#!/bin/bash

pip freeze \
--exclude black \
--exclude torch \
--exclude torchvision \
--exclude torchmetrics \
--exclude pytorch-lightning \
--exclude nvidia-cublas-cu11 \
--exclude nvidia-cuda-cupti-cu11 \
--exclude nvidia-cuda-nvrtc-cu11 \
--exclude nvidia-cuda-runtime-cu11 \
--exclude nvidia-cudnn-cu11 \
--exclude nvidia-cufft-cu11 \
--exclude nvidia-curand-cu11 \
--exclude nvidia-cusolver-cu11 \
--exclude nvidia-cusparse-cu11 \
--exclude nvidia-nccl-cu11 \
--exclude nvidia-nvtx-cu11 \
--exclude physionet2023 \
--exclude mvtst \