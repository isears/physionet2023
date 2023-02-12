import os
from dataclasses import dataclass

import torch


@dataclass
class Config:
    cores_available: int
    gpus_available: int


config = Config(
    cores_available=len(os.sched_getaffinity(0)),
    gpus_available=torch.cuda.device_count(),
)
