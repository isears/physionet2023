import os
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch
from mvtst.models import TSTConfig


class LabelType(Enum):
    DUMMY = 0
    RAW = 1
    NORMALIZED = 2
    SINGLECLASS = 3
    MULTICLASS = 4
    AGE = 5


@dataclass
class Config:
    cores_available: int
    gpus_available: int


@dataclass
class PhysionetConfig(TSTConfig):
    label_type: LabelType = LabelType.RAW

    def __post_init__(self) -> None:
        if self.label_type == LabelType.MULTICLASS:
            self.num_classes = 5
        else:
            self.num_classes = 1
            
        return super().__post_init__()


gpu_count = 0

try:
    gpu_count = torch.cuda.device_count()
except:  # I don't care if this fails
    pass

config = Config(
    cores_available=len(os.sched_getaffinity(0)),
    gpus_available=gpu_count,
)
