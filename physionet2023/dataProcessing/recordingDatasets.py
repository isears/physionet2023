import numpy as np
import torch
from scipy.signal import spectrogram

from physionet2023.dataProcessing.datasets import PatientDataset, RecordingDataset


class SpectrogramDataset(RecordingDataset):
    def __init__(self, pids: list, shuffle=True, f_min=0.5, f_max=30, **super_kwargs):
        super().__init__(pids, shuffle, **super_kwargs)
        self.f_min = f_min
        self.f_max = f_max

    def __getitem__(self, index: int):
        recording_data, static_data, label = super().__getitem__(index)

        spectrograms = list()

        for channel_idx in range(0, recording_data.shape[0]):
            f, t, s = spectrogram(recording_data[channel_idx, :], 100.0)
            freq_filter = np.logical_and(f > self.f_min, f < self.f_max)
            s = s[freq_filter]
            f = f[freq_filter]

            # TODO: research alternative approaches
            s = np.log10(s)
            spectrograms.append(s)

        # channels-last
        X = np.stack(spectrograms, axis=-1)

        # deal with -inf
        X[X == -np.inf] = X[X != -np.inf].min()

        return torch.tensor(X), label


if __name__ == "__main__":
    pds = PatientDataset()
    ds = SpectrogramDataset(pds.patient_ids)

    shape = None

    for X, y in ds:
        if shape is None:
            shape = X.shape
        else:
            assert shape == X.shape, f"Shapes didn't match: {shape}, {X.shape}"
