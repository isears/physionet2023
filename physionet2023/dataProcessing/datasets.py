import os.path
import random
from typing import Literal

import numpy as np
import pandas as pd
import torch
from scipy.signal import decimate
from sklearn.model_selection import train_test_split

from physionet2023 import LabelType, config
from physionet2023.dataProcessing.exampleUtil import *


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

        # Do quality control to ensure all patients have usable data based on quality cutoff
        passed_qc = list()

        for pid in self.patient_ids:
            recording_metadata = self._load_recording_metadata(pid)
            if len(recording_metadata) > 0:
                passed_qc.append(pid)

        self.patient_ids = passed_qc

    def _load_recording_metadata(self, patient_id) -> pd.DataFrame:
        recording_metadata_file = os.path.join(
            self.root_folder, patient_id, patient_id + ".tsv"
        )
        recording_metadata = pd.read_csv(recording_metadata_file, delimiter="\t")
        recording_metadata = recording_metadata.dropna()
        recording_metadata = recording_metadata[
            recording_metadata["Quality"] > self.quality_cutoff
        ]

        return recording_metadata

    def _load_single_recording(self, patient_id, recording_id):
        recording_location = os.path.join(self.root_folder, patient_id, recording_id)

        recording_data, sampling_frequency, channels = load_recording(
            recording_location
        )

        # Making some assumptions about uniformity of data. Will have to rethink this if not true
        # ...seems like it is true; could take this out if performance becomes a concern
        assert sampling_frequency == self.sampling_frequency
        assert len(channels) == len(self.channels)
        assert recording_data.shape[-1] == self.full_record_len

        for expected_channel, actual_channel in zip(self.channels, channels):
            assert expected_channel == actual_channel

        return recording_data

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
        recording_metadata = self._load_recording_metadata(patient_id)

        # Load recordings.
        recordings = [
            self._load_single_recording(patient_id, r)
            for r in recording_metadata["Record"].to_list()
        ]

        # Make sure each recording has associated metadata
        assert len(recording_metadata) == len(recordings)

        return patient_metadata, recording_metadata, recordings

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


class PatientTrainingDataset(PatientDataset):
    def __init__(self, root_folder: str, sample_len: int, normalize=True, **kwargs):
        super().__init__(root_folder, **kwargs)

        if self.include_static:
            raise NotImplementedError

        assert sample_len <= self.full_record_len

        self.sample_len = sample_len
        self.normalize = normalize

    def collate(self, batch):
        X = torch.stack([recording_data for recording_data, _ in batch], dim=0)
        y = torch.stack([label for _, label in batch], dim=0)

        return X, y

    def tst_collate(self, batch):
        """
        TST also needs pad_mask, even though all sequences are the same length
        """
        X, y = self.collate(batch)

        pad_mask = torch.ones_like(X[:, 0, :]).bool()

        return X.permute(0, 2, 1), y, pad_mask, "DummyID"

    def __getitem__(self, index: int):
        patient_id = self.patient_ids[index]
        patient_metadata = self._load_patient_metadata(patient_id)
        recording_metadata = self._load_recording_metadata(patient_id)

        # Only the finest recordings will do
        # TODO: could iterate through all recordings and select the one with highest StD or something
        # but that would be much more computationally expensive
        recording_id = recording_metadata.nlargest(1, "Quality")["Record"].item()

        recording_data = self._load_single_recording(patient_id, recording_id)

        if self.normalize:
            recording_data = (
                recording_data - recording_data.mean()
            ) / recording_data.std()

        # Take sequence from the middle of the recording
        left_margin = int((self.full_record_len - self.sample_len) / 2)
        right_margin = (self.full_record_len - self.sample_len) - left_margin

        recording_sample = recording_data[
            :, left_margin : (left_margin + self.sample_len)
        ]

        return (torch.tensor(recording_sample), self._get_label(patient_id))


def demo(dl, n_batches=3):
    for batchnum, batch_data in enumerate(dl):
        print("=" * 15)
        print(f"Batch number: {batchnum}")
        print(batch_data)

        if batchnum == n_batches:
            break


if __name__ == "__main__":
    pass
