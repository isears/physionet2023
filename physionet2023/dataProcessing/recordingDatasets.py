import random

import numpy as np
import torch
from scipy.signal import decimate, spectrogram

from physionet2023.dataProcessing.datasets import PatientDataset


class RecordingDataset(PatientDataset):
    def __init__(
        self, shuffle=True, last_only=False, include_static=False, **super_kwargs
    ):
        super().__init__(**super_kwargs)

        self.shuffle = shuffle
        self.include_static = include_static

        # Generate an index of tuples (patient_id, recording_id)
        self.patient_recording_index = list()

        if self.shuffle:
            self.patient_ids = random.sample(self.patient_ids, len(self.patient_ids))
            recordings_dict = {
                pid: sorted(
                    self._load_recording_metadata(pid)["Record"].to_list(),
                    key=lambda k: random.random(),
                )
                for pid in self.patient_ids
            }
        else:
            recordings_dict = {
                pid: self._load_recording_metadata(pid)["Record"].to_list()
                for pid in self.patient_ids
            }

        if last_only:
            recordings_dict = {
                pid: [self._load_recording_metadata(pid)["Record"].to_list()[-1]]
                for pid in self.patient_ids
            }

        while len(recordings_dict) > 0:
            this_iter_patient_ids = list(recordings_dict.copy().keys())

            for patient_id in this_iter_patient_ids:
                self.patient_recording_index.append(
                    (patient_id, recordings_dict[patient_id].pop())
                )

                if len(recordings_dict[patient_id]) == 0:
                    del recordings_dict[patient_id]

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

        if self.include_static:
            return (
                torch.tensor(recording_data),
                static_data,
                self._get_label(patient_id),
            )
        else:
            return torch.tensor(recording_data), self._get_label(patient_id)


class FftDownsamplingDataset(RecordingDataset):
    def __init__(self, pids: list, sample_len=1000, **super_kwargs):
        super().__init__(pids, **super_kwargs)

        self.sample_len = sample_len

    def __getitem__(self, index: int):
        X, static_data, y = super().__getitem__(index)

        X_fft = np.zeros_like(X)
        for channel_idx in range(0, X.shape[0]):
            X_fft[channel_idx, :] = np.abs(np.fft.fft(X[channel_idx, :]))

        fft_resample_factor = int(self.full_record_len / self.sample_len)
        # TODO: this is sad and wrong
        # There are better ways to downsample FFT
        X_fft_downsampled = decimate(X_fft, fft_resample_factor)

        assert X_fft_downsampled.shape[-1] == self.sample_len

        return X_fft_downsampled, static_data, y


class SpectrogramDataset(RecordingDataset):
    def __init__(
        self,
        shuffle=True,
        f_min=0.5,
        f_max=30,
        **super_kwargs,
    ):
        super().__init__(
            shuffle,
            include_static=False,
            **super_kwargs,
        )
        self.f_min = f_min
        self.f_max = f_max

        sample_X, _ = self.__getitem__(0)
        self.dims = (sample_X.shape[1], sample_X.shape[2])
        self.sample_len = self.dims[1]  # Mostly for backwards compatibility

    def __getitem__(self, index: int):
        recording_data, static_data, label = super().__getitem__(index)

        spectrograms = list()

        for channel_idx in range(0, recording_data.shape[0]):
            f, t, s = spectrogram(recording_data[channel_idx, :], 100.0)
            freq_filter = np.logical_and(f > self.f_min, f < self.f_max)
            s = s[freq_filter]
            f = f[freq_filter]

            with np.errstate(divide="ignore"):
                s = np.log10(s)  # TODO: research alternative approaches

            spectrograms.append(s)

        # channels-first
        X = np.stack(spectrograms, axis=0)

        # deal with -inf
        X[X == -np.inf] = X[X != -np.inf].min()

        return torch.tensor(X), label


class SpectrogramAgeDataset(SpectrogramDataset):
    MAX_AGE = 90.0  # Specified in competition data description: "all ages above 89 were aggregated into a single category and encoded as “90”"

    def __getitem__(self, index: int):
        patient_id, recording_id = self.patient_recording_index[index]
        patient_metadata = self._load_patient_metadata(patient_id)

        age = patient_metadata["Age"]

        if self.normalize:
            y = torch.tensor(float(age) / self.MAX_AGE).unsqueeze(-1)

        X, _ = super().__getitem__(index)

        return X, y


if __name__ == "__main__":
    pds = PatientDataset()
    ds = SpectrogramAgeDataset(pds.patient_ids)

    shape = None

    for X, y in ds:
        if shape is None:
            shape = X.shape
        else:
            assert shape == X.shape, f"Shapes didn't match: {shape}, {X.shape}"
