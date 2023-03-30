import os
from dataclasses import dataclass

import torch


@dataclass
class Config:
    cores_available: int
    gpus_available: int


gpu_count = 0

try:
    gpu_count = torch.cuda.device_count()
except:  # I don't care if this fails
    pass

config = Config(
    cores_available=len(os.sched_getaffinity(0)),
    gpus_available=gpu_count,
)
