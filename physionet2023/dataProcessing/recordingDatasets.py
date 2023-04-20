import random

import numpy as np
import torch
from scipy.signal import decimate, resample_poly, spectrogram
from sklearn.preprocessing import robust_scale

from physionet2023.dataProcessing.datasets import PatientDataset


def preprocess_signal(
    sig_in: np.ndarray, original_rate: int = 100, common_rate: int = 100
) -> np.ndarray:
    # Robust scaling, low pass filter, resample
    sig_out = sig_in

    # robust scaling
    sig_out = robust_scale(sig_in, axis=1)  # TODO: double check this is the right axis

    # resampling (includes low-pass filter)
    if common_rate > original_rate:
        up = common_rate
        down = original_rate
    else:
        up = common_rate
        down = original_rate

    factor = np.gcd(up, down)

    up = int(up / factor)
    down = int(down / factor)

    sig_out = resample_poly(sig_out, up, down, axis=1)

    return sig_out


class RecordingDataset(PatientDataset):
    def __init__(
        self,
        shuffle=True,
        last_only=False,
        include_static=False,
        preprocess=False,
        **super_kwargs,
    ):
        super().__init__(include_static=include_static, **super_kwargs)

        self.shuffle = shuffle
        self.include_static = include_static
        self.preprocess = preprocess

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

    def __len__(self):
        return len(self.patient_recording_index)

    def __getitem__(self, index: int):
        patient_id, recording_id = self.patient_recording_index[index]
        patient_metadata = self._load_patient_metadata(patient_id)
        recording_data = self._load_single_recording(patient_id, recording_id)

        if self.preprocess:
            recording_data = preprocess_signal(recording_data)

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
        super().__init__(pids, include_static=False, **super_kwargs)

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
    # Cutoff frequencies for spectrogram
    f_min = 0.5
    f_max = 30

    def __init__(
        self,
        shuffle=True,
        **super_kwargs,
    ):
        super().__init__(
            shuffle,
            include_static=False,
            **super_kwargs,
        )

        sample_X, _ = self.__getitem__(0)
        self.dims = (sample_X.shape[1], sample_X.shape[2])
        self.sample_len = self.dims[1]  # Mostly for backwards compatibility

    @classmethod
    def _to_spectrogram(cls, eeg_data):
        spectrograms = list()

        for channel_idx in range(0, eeg_data.shape[0]):
            f, t, s = spectrogram(eeg_data[channel_idx, :], 100.0)
            freq_filter = np.logical_and(f > cls.f_min, f < cls.f_max)
            s = s[freq_filter]
            f = f[freq_filter]

            with np.errstate(divide="ignore"):
                s = np.log10(s)  # TODO: research alternative approaches

            spectrograms.append(s)

        # channels-first
        X = np.stack(spectrograms, axis=0)
        # deal with -inf
        X[X == -np.inf] = X[X != -np.inf].min()
        # (18, 75, 133)
        return torch.tensor(X)

    def __getitem__(self, index: int):
        recording_data, label = super().__getitem__(index)

        X = self.__class__._to_spectrogram(recording_data)

        return X, label


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
    ds = SpectrogramDataset(pds.patient_ids)

    shape = None

    for X, y in ds:
        if shape is None:
            shape = X.shape
        else:
            assert shape == X.shape, f"Shapes didn't match: {shape}, {X.shape}"
