import os.path
import random
from typing import Literal

import numpy as np
import pandas as pd
import torch
from scipy.signal import decimate
from sklearn.model_selection import train_test_split

from helper_code import load_challenge_data, load_recording_data, load_text_file
from physionet2023 import LabelType, config


class PatientDataset(torch.utils.data.Dataset):
    channels = [
        "Fp1-F7",
        "F7-T3",
        "T3-T5",
        "T5-O1",
        "Fp2-F8",
        "F8-T4",
        "T4-T6",
        "T6-O2",
        "Fp1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "Fp2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
        "Fz-Cz",
        "Cz-Pz",
    ]

    static_features = {
        "Age": lambda x: float(x),
        "Sex": lambda x: 1.0 if x == "Male" else 0.0,
        # TODO: could find a better way to deal with na ROSC
        "ROSC": lambda x: float(x),
        "OHCA": lambda x: 1.0 if x == "True" else 0.0,
        "VFib": lambda x: 1.0 if x == "True" else 0.0,
        "TTM": lambda x: float(x),
    }

    full_record_len = 30000

    sampling_frequency = 100.0

    def __init__(
        self,
        root_folder: str = "./data",
        patient_ids: list = None,
        quality_cutoff: float = 0.5,
        include_static: bool = True,
        label_type: LabelType = LabelType.RAW,
    ):
        random.seed(0)

        if not label_type in LabelType:
            raise ValueError(f"Unsupported label type: {label_type}")

        self.label_type = label_type

        data_folders = list()
        for x in os.listdir(root_folder):
            data_folder = os.path.join(root_folder, x)
            if os.path.isdir(data_folder):
                data_folders.append(x)

        self.patient_ids = sorted(data_folders)

        if patient_ids is not None:
            self.patient_ids = [pid for pid in self.patient_ids if pid in patient_ids]

        self.root_folder = root_folder
        self.quality_cutoff = quality_cutoff

        self.include_static = include_static
        if include_static:
            self.features_dim = len(self.static_features) + len(self.channels)
        else:
            self.features_dim = len(self.channels)

    def _load_patient_metadata(self, patient_id) -> dict:
        # Define file location.
        patient_metadata_file = os.path.join(
            self.root_folder, patient_id, patient_id + ".txt"
        )

        # Load non-recording data.
        patient_metadata_raw = load_text_file(patient_metadata_file)
        patient_metadata = {
            line.split(": ")[0]: line.split(": ")[-1]
            for line in patient_metadata_raw.split("\n")
        }

        return patient_metadata

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index: int):
        patient_id = self.patient_ids[index]

        patient_metadata = self._load_patient_metadata(patient_id)

        with open(f"{self.root_folder}/{patient_id}/RECORDS", "r") as f:
            records = [r.strip() for r in f.readlines() if "ECG" not in r]

        # Load recordings.
        # recordings = [
        #     load_recording_data(f"{self.root_folder}/{patient_id}/{r}") for r in records
        # ]

        return patient_metadata, records

    def _get_label(self, patient_id: str):
        if self.label_type == LabelType.DUMMY:
            return torch.tensor(float("nan")).unsqueeze(-1)

        patient_metadata = self._load_patient_metadata(patient_id)

        try:
            raw_label = int(patient_metadata["CPC"])
        except KeyError:
            print("[*] Warning: label not found, returning made-up label")
            raw_label = 1

        if self.label_type == LabelType.RAW:
            return torch.tensor(float(raw_label)).unsqueeze(-1)
        elif self.label_type == LabelType.NORMALIZED:
            return torch.tensor((raw_label - 1.0) / 4.0).unsqueeze(-1)
        elif self.label_type == LabelType.SINGLECLASS:
            return torch.tensor(float(raw_label > 2)).unsqueeze(-1)
        elif self.label_type == LabelType.MULTICLASS:
            ret = torch.zeros(5)

            ret[raw_label - 1] = 1.0

            return ret
        else:
            raise ValueError(f"Unsupported label type: {self.label_type}")


def demo(dl, n_batches=3):
    for batchnum, batch_data in enumerate(dl):
        print("=" * 15)
        print(f"Batch number: {batchnum}")
        print(batch_data)

        if batchnum == n_batches:
            break


if __name__ == "__main__":
    ds = PatientDataset()

    for patient_metadata, recordings in ds:
        print(patient_metadata)
        print(recordings)
        break
