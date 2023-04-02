import numpy as np
import torch
from scipy.signal import spectrogram

from physionet2023.dataProcessing.datasets import PatientDataset, RecordingDataset


class SpectrogramDataset(RecordingDataset):
    def __init__(
        self,
        shuffle=True,
        for_classification=False,
        normalize=True,
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
        self.for_classification = for_classification
        self.normalize = normalize

        if self.for_classification and self.normalize:
            print(
                "[WARNING] Incompatible params for_classification and normalize (normalize will have no effect)"
            )

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

        if self.for_classification and not self.for_testing:
            classification_label = (label > 2).float()
            return torch.tensor(X), classification_label.unsqueeze(-1)
        else:
            if self.normalize:
                return torch.tensor(X), ((label - 1.0) / 4.0).unsqueeze(-1)
            else:
                return torch.tensor(X), label.unsqueeze(-1)


if __name__ == "__main__":
    pds = PatientDataset()
    ds = SpectrogramDataset(pds.patient_ids)

    shape = None

    for X, y in ds:
        if shape is None:
            shape = X.shape
        else:
            assert shape == X.shape, f"Shapes didn't match: {shape}, {X.shape}"
