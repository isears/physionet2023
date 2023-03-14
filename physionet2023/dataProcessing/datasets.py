import os.path
import random

import pandas as pd
import torch
from scipy.signal import decimate
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

    def __init__(
        self, root_folder: str, quality_cutoff: float = 0.5, include_static: bool = True
    ):
        random.seed(0)
        data_folders = list()
        for x in os.listdir(root_folder):
            data_folder = os.path.join(root_folder, x)
            if os.path.isdir(data_folder):
                data_folders.append(x)

        self.patient_ids = sorted(data_folders)
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

        return (
            torch.tensor(recording_sample),
            torch.tensor(float(patient_metadata["CPC"])),
        )


class RecordingDataset(PatientDataset):
    def __init__(self, root_folder: str, shuffle=True, **super_kwargs):
        super().__init__(root_folder)

        self.shuffle = shuffle

        # Generate an index of tuples (patient_id, recording_id)
        self.patient_recording_index = list()

        for pid in self.patient_ids:
            recording_metadata = self._load_recording_metadata(pid)

            for recording_id in recording_metadata["Record"].to_list():
                self.patient_recording_index.append((pid, recording_id))

        if self.shuffle:
            random.shuffle(self.patient_recording_index)

    def collate(self, batch):
        X = torch.stack([recording_data for recording_data, _, _ in batch], dim=0)
        y = torch.stack([label for _, _, label in batch], dim=0)

        if self.include_static:
            static_data = torch.stack([s for _, s, _ in batch], dim=0)

            # Repeat static data over timeseries dimension, then concat with X
            static_data_repeat = static_data.unsqueeze(2).repeat(1, 1, X.shape[-1])
            X_with_static = torch.cat((X, static_data_repeat), 1)

            return X_with_static, y
        else:
            return X, y

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
        sample_len=1000,
        resample_factor: int = None,
        normalize=True,
        **super_kwargs,
    ):
        super().__init__(root_folder, **super_kwargs)

        self.sample_len = sample_len
        self.patient_recording_sample_index = list()

        for patient_id, recording_id in self.patient_recording_index:
            for sample_idx in range(0, self.full_record_len - sample_len, sample_len):
                self.patient_recording_sample_index.append(
                    (patient_id, recording_id, sample_idx)
                )

        if self.shuffle:
            random.shuffle(self.patient_recording_sample_index)

        self.resample_factor = resample_factor
        self.normalize = normalize

    def noleak_traintest_split(self, test_size=0.1, seed=0):
        """
        Splits datasets based on pantients, not on samples.
        This ensures samples from a single patient don't appear in both training and validation datasets, which could lead to over-fitting
        """
        train_pids, test_pids = train_test_split(
            self.patient_ids, test_size=test_size, random_state=seed
        )

        train_ds = PidSampleDataset(self.root_folder, patient_ids=train_pids)
        test_ds = PidSampleDataset(self.root_folder, patient_ids=test_pids)

        for pid in test_ds.patient_ids:
            assert pid not in train_ds.patient_ids

        return train_ds, test_ds

    def __len__(self):
        return len(self.patient_recording_sample_index)

    def __getitem__(self, index: int):
        patient_id, recording_id, sample_idx = self.patient_recording_sample_index[
            index
        ]
        patient_metadata = self._load_patient_metadata(patient_id)
        recording_data = self._load_single_recording(patient_id, recording_id)

        if self.resample_factor:
            sample_data = recording_data[
                :, sample_idx : sample_idx + self.resample_factor * self.sample_len
            ]
            sample_data = decimate(sample_data, self.resample_factor)

            assert sample_data.shape[-1] == self.sample_len
        else:
            sample_data = recording_data[:, sample_idx : sample_idx + self.sample_len]

        if self.normalize:
            sample_data = (sample_data - sample_data.mean()) / (sample_data.std())
            sample_data = np.nan_to_num(sample_data)  # Weirdly some data has std of 0

        static_data = torch.tensor(
            [
                converter(patient_metadata[f])
                for f, converter in self.static_features.items()
            ]
        )

        # NOTE: copy was necessary to prevent "negative stride error" after decimation
        # Not sure what the performance implications are
        return (
            torch.tensor(sample_data.copy()),
            torch.nan_to_num(static_data, 0.0),
            torch.tensor(float(patient_metadata["CPC"])),
        )


class PidSampleDataset(SampleDataset):
    def __init__(self, root_folder: str, patient_ids: list[str], **super_kwargs):
        super().__init__(root_folder, **super_kwargs)

        # Restricts the available data to just data associated with patient ids passed in constructor
        self.patient_recording_sample_index = list(
            filter(
                lambda sample_index: sample_index[0] in patient_ids,
                self.patient_recording_sample_index,
            )
        )

        self.patient_ids = patient_ids


def just_give_me_dataloaders(
    data_path="./data",
    batch_size=32,
    test_size=0.1,
    sample_len=1000,
    test_subsample=1.0,
    **ds_kwargs,
):
    ds = SampleDataset(
        data_path,
        sample_len=sample_len,
        **ds_kwargs,
    )
    train_ds, test_ds = ds.noleak_traintest_split(test_size=test_size)

    subsample_length = int(len(test_ds) * test_subsample)

    test_ds_subsampled, _ = torch.utils.data.random_split(
        test_ds,
        [subsample_length, len(test_ds) - subsample_length],
        generator=torch.Generator().manual_seed(42),
    )

    training_dl = torch.utils.data.DataLoader(
        train_ds,
        collate_fn=train_ds.tst_collate,
        num_workers=config.cores_available,
        batch_size=batch_size,
        pin_memory=True,
    )

    testing_dl = torch.utils.data.DataLoader(
        test_ds_subsampled,
        collate_fn=test_ds.tst_collate,
        num_workers=config.cores_available,
        batch_size=batch_size,
        pin_memory=True,
    )

    return training_dl, testing_dl


def just_give_me_numpy(
    data_path="./data",
    num_examples=1000,
    test_size=0.1,
    sample_len=1000,
    resample_factor=None,
):
    ds = SampleDataset(
        data_path, sample_len=sample_len, resample_factor=resample_factor
    )
    train_ds, test_ds = ds.noleak_traintest_split(test_size=test_size)
    training_dl = torch.utils.data.DataLoader(
        train_ds,
        collate_fn=train_ds.collate,
        num_workers=config.cores_available,
        batch_size=int(num_examples - (num_examples * test_size)),
        pin_memory=True,
    )

    testing_dl = torch.utils.data.DataLoader(
        test_ds,
        collate_fn=test_ds.collate,
        num_workers=config.cores_available,
        batch_size=int(num_examples * test_size),
        pin_memory=True,
    )

    X_train, y_train = next(iter(training_dl))
    X_test, y_test = next(iter(testing_dl))

    return X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy()


def demo(dl, n_batches=3):
    for batchnum, (X, Y, pm, id) in enumerate(dl):
        print(f"Batch number: {batchnum}")
        print(f"X shape: {X.shape}")
        print(f"Y: {Y}")

        if batchnum == n_batches:
            break


if __name__ == "__main__":
    ds = PatientTrainingDataset(root_folder="./data", sample_len=2000)

    print(f"Initialized dataset with length: {len(ds)}")

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=4,
        collate_fn=ds.tst_collate,
        pin_memory=True,
    )

    print("Demoing first few batches...")
    demo(dl)
