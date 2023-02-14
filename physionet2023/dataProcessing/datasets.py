import os.path
import random

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from physionet2023 import config
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

    def __init__(self, root_folder: str, quality_cutoff: float = 0.5):
        random.seed(0)
        data_folders = list()
        for x in os.listdir(root_folder):
            data_folder = os.path.join(root_folder, x)
            if os.path.isdir(data_folder):
                data_folders.append(x)

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


class RecordingDataset(PatientDataset):
    def __init__(
        self,
        root_folder: str,
        quality_cutoff: float = 0.5,
        shuffle=True,
    ):
        super().__init__(root_folder, quality_cutoff)

        # Generate an index of tuples (patient_id, recording_id)
        self.patient_recording_index = list()

        for pid in self.patient_ids:
            recording_metadata = self._load_recording_metadata(pid)

            for recording_id in recording_metadata["Record"].to_list():
                self.patient_recording_index.append((pid, recording_id))

        if shuffle:
            random.shuffle(self.patient_recording_index)

    def collate(self, batch):
        X = torch.stack([recording_data for recording_data, _, _ in batch], dim=0)
        y = torch.stack([label for _, _, label in batch], dim=0)
        static_data = torch.stack([s for _, s, _ in batch], dim=0)

        # Repeat static data over timeseries dimension, then concat with X
        static_data_repeat = static_data.unsqueeze(2).repeat(1, 1, X.shape[-1])
        X_with_static = torch.cat((X, static_data_repeat), 1)

        return X_with_static, y

    def tst_collate(self, batch):
        """
        TST also needs pad_mask, even though all sequences are the same length
        """
        X, y = self.collate(batch)

        pad_mask = torch.ones_like(X[:, 0, :]).bool()

        return X.permute(0, 2, 1), y, pad_mask, "DummyID"

    def __len__(self):
        return len(self.patient_recording_index)

    def __getitem__(self, index: int):
        patient_id, recording_id = self.patient_recording_index[index]
        patient_metadata = self._load_patient_metadata(patient_id)
        recording_data = self._load_single_recording(patient_id, recording_id)

        static_data = torch.tensor(
            [
                converter(patient_metadata[f])
                for f, converter in self.static_features.items()
            ]
        )

        return (
            torch.tensor(recording_data),
            static_data,
            torch.tensor(float(patient_metadata["CPC"])),
        )


class SampleDataset(RecordingDataset):
    def __init__(
        self,
        root_folder: str,
        quality_cutoff: float = 0.5,
        shuffle=True,
        sample_len=1000,
    ):
        super().__init__(root_folder, quality_cutoff, shuffle)

        self.sample_len = sample_len
        self.patient_recording_sample_index = list()

        for patient_id, recording_id in self.patient_recording_index:
            for sample_idx in range(0, self.full_record_len - sample_len, sample_len):
                self.patient_recording_sample_index.append(
                    (patient_id, recording_id, sample_idx)
                )

        if shuffle:
            random.shuffle(self.patient_recording_sample_index)

    def noleak_traintest_split(self, test_size=0.1, seed=0):
        train_pids, test_pids = train_test_split(
            self.patient_ids, test_size=test_size, random_state=seed
        )

        train_ds = PidSampleDataset(self.root_folder, patient_ids=train_pids)
        test_ds = PidSampleDataset(self.root_folder, patient_ids=test_pids)

        return train_ds, test_ds

    def __len__(self):
        return len(self.patient_recording_sample_index)

    def __getitem__(self, index: int):
        patient_id, recording_id, sample_idx = self.patient_recording_sample_index[
            index
        ]
        patient_metadata = self._load_patient_metadata(patient_id)
        recording_data = self._load_single_recording(patient_id, recording_id)
        sample_data = recording_data[:, sample_idx : sample_idx + self.sample_len]

        static_data = torch.tensor(
            [
                converter(patient_metadata[f])
                for f, converter in self.static_features.items()
            ]
        )

        return (
            torch.tensor(sample_data),
            torch.nan_to_num(static_data, 0.0),
            torch.tensor(float(patient_metadata["CPC"])),
        )


class PidSampleDataset(SampleDataset):
    def __init__(
        self,
        root_folder: str,
        patient_ids: list[str],
        quality_cutoff: float = 0.5,
        shuffle=True,
        sample_len=1000,
    ):
        super().__init__(root_folder, quality_cutoff, shuffle, sample_len)

        # Restricts the available data to just data associated with patient ids passed in constructor
        self.patient_recording_sample_index = list(
            filter(
                lambda sample_index: sample_index[0] in patient_ids,
                self.patient_recording_sample_index,
            )
        )


def demo(dl, n_batches=3):
    for batchnum, (X, Y) in enumerate(dl):
        print(f"Batch number: {batchnum}")
        print(f"X shape: {X.shape}")
        print(f"Y: {Y}")

        if batchnum == n_batches:
            break


if __name__ == "__main__":
    ds = SampleDataset(root_folder="./data")

    print(f"Initialized dataset with length: {len(ds)}")

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=4,
        collate_fn=ds.collate,
        pin_memory=True,
    )

    print("Demoing first few batches...")
    demo(dl)
