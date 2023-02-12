import os.path
import random

import pandas as pd
import torch

from physionet2023 import config
from physionet2023.dataProcessing.exampleUtil import *

random.seed(0)


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

    def __init__(self, root_folder: str, quality_cutoff: float = 0.5):

        data_folders = list()
        for x in os.listdir(root_folder):
            data_folder = os.path.join(root_folder, x)
            if os.path.isdir(data_folder):
                data_folders.append(x)

        # TODO: implement optional shuffling
        self.patient_ids = sorted(data_folders)
        self.root_folder = root_folder
        self.quality_cutoff = quality_cutoff

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
        assert sampling_frequency == 100.0
        assert len(channels) == len(self.channels)

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


class RecordingDataset(PatientDataset):
    def __init__(self, root_folder: str, quality_cutoff: float = 0.5, shuffle=True):
        super().__init__(root_folder, quality_cutoff)

        # Generate an index of tuples (patient_id, recording_id)
        self.patient_recording_index = list()

        for pid in self.patient_ids:
            recording_metadata = self._load_recording_metadata(pid)

            for recording_id in recording_metadata["Record"].to_list():
                self.patient_recording_index.append((pid, recording_id))

        if shuffle:
            random.shuffle(self.patient_recording_index)

    def __len__(self):
        return len(self.patient_recording_index)

    def __getitem__(self, index: int):
        patient_id, recording_id = self.patient_recording_index[index]
        patient_metadata = self._load_patient_metadata(patient_id)
        recording_data = self._load_single_recording(patient_id, recording_id)

        if patient_metadata["Outcome"] == "Good":
            outcome = 1.0
        elif patient_metadata["Outcome"] == "Poor":
            outcome = 0.0
        else:
            raise ValueError(patient_metadata["Outcome"])

        return recording_data, patient_metadata["Outcome"], patient_metadata["CPC"]


def demo(dl):
    print("Printing first few batches:")
    for batchnum, (X, Y, Z) in enumerate(dl):
        print(f"Batch number: {batchnum}")
        print(X[0].shape)
        print(Y)
        print(Z)

        if batchnum == 5:
            break


if __name__ == "__main__":
    ds = RecordingDataset(root_folder="./data")

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=4,
        pin_memory=True,
    )

    print("Demoing first few batches...")
    demo(dl)
