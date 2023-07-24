import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from scipy.signal import spectrogram

from helper_code import (
    get_utility_frequency,
    load_recording_data,
    preprocess_data,
    reduce_channels,
)
from physionet2023.dataProcessing.datasets import PatientDataset


class AvgFFTDataset(PatientDataset):
    def __init__(
        self,
        root_folder: str = "./data",
        sample_len=1000,
        patient_ids: list = None,
        quality_cutoff: float = 0.5,
    ):
        super().__init__(root_folder, patient_ids, quality_cutoff, include_static=False)
        self.sample_len = sample_len

    @staticmethod
    def tst_collate(batch):
        """
        TST also needs pad_mask, even though all sequences are the same length
        """
        X = torch.stack([recording_data for recording_data, _ in batch], dim=0)
        y = torch.stack([label for _, label in batch], dim=0)

        pad_mask = torch.ones_like(X[:, 0, :]).bool()

        return X.permute(0, 2, 1), y, pad_mask, "DummyID"

    def __getitem__(self, index: int):
        patient_metadata, recording_metadata, recordings = super().__getitem__(index)

        sample_count = 0
        aggregated_fft = np.zeros((len(self.channels), self.sample_len))
        for r in recordings:
            for sample_idx in range(
                0, self.full_record_len - self.sample_len, self.sample_len
            ):
                sample_count += 1

                sample_data = r[:, sample_idx : sample_idx + self.sample_len]

                sample_fft = np.zeros_like(sample_data)
                for channel_idx in range(0, sample_data.shape[0]):
                    sample_fft[channel_idx, :] = np.abs(
                        np.fft.fft(sample_data[channel_idx, :])
                    )

                aggregated_fft += sample_fft

        aggregated_fft = aggregated_fft / sample_count

        return torch.tensor(aggregated_fft, dtype=torch.float32), self._get_label(
            patient_metadata["Patient"]
        )


class MetadataOnlyDataset(PatientDataset):
    """
    Save loading recordings if all we're interested in is the metadata (e.g. for stratified test/train split)
    """

    def __getitem__(self, index: int):
        patient_id = self.patient_ids[index]

        patient_metadata = self._load_patient_metadata(patient_id)
        recording_metadata = self._load_recording_metadata(patient_id)

        return patient_id, patient_metadata, recording_metadata

    def get_by_pid(self, patient_id: str):
        patient_metadata = self._load_patient_metadata(patient_id)
        recording_metadata = self._load_recording_metadata(patient_id)

        return patient_id, patient_metadata, recording_metadata


class AvgSpectralDensityDataset(PatientDataset):
    """
    Average spectral densities over all recordings
    """

    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)
        sample_X, _ = self.__getitem__(0)

        assert sample_X.ndim == 2
        self.features_dim = sample_X.shape[0]
        self.max_len = sample_X.shape[-1]

    def __getitem__(self, index: int):
        patient_id = self.patient_ids[index]
        patient_metadata, recording_metadata, recordings = super().__getitem__(index)

        recording_sds = list()

        for r in recordings:
            # Includes delta, theta, alpha, and beta waves
            #
            # delta: 0.5 - 4 Hz
            # theta: 4 - 7 Hz
            # alpha: 8 - 13 Hz
            # beta: 13 - 30 Hz
            #
            # https://en.wikipedia.org/wiki/Electroencephalography
            this_recording_sd, _ = mne.time_frequency.psd_array_welch(
                r,
                sfreq=self.sampling_frequency,
                fmin=0.5,
                fmax=30,
                verbose=False,
                n_fft=int(self.sampling_frequency * 10),
            )

            recording_sds.append(this_recording_sd)

        # avg along recording axis
        # X = np.mean(np.stack(recording_sds, axis=-1), axis=-1)
        # TODO: for now just take last recording, averaging may be causing some weirdness
        X = recording_sds[-1]

        # log10
        X = np.log10(X)

        # Within-spectrum normalization
        X = (X - np.mean(X)) / np.std(X)

        return torch.tensor(X).float(), self._get_label(patient_id)


class SpectrogramDataset(PatientDataset):
    def __init__(
        self,
        shuffle=True,
        f_min=0.5,
        f_max=30,
        **super_kwargs,
    ):
        super().__init__(
            **super_kwargs,
        )
        self.f_min = f_min
        self.f_max = f_max

        self.used_channels = ["F3", "P3", "F4", "P4"]

        sample_X, _ = self.__getitem__(0)
        self.dims = (sample_X.shape[1], sample_X.shape[2])
        self.sample_len = self.dims[1]  # Mostly for backwards compatibility

    def __getitem__(self, index: int):
        patient_metadata, recording_metadata = super().__getitem__(index)
        patient_id = self.patient_ids[index]

        qualified_records = recording_metadata[
            (recording_metadata[self.used_channels].all(axis=1))
            & (recording_metadata["length"] >= self.signal_length)
        ]

        if len(qualified_records) == 0:  # Can't always get what you want
            qualified_records = recording_metadata[
                (recording_metadata[self.used_channels].any(axis=1))
                & (recording_metadata["length"] >= self.signal_length)
            ]

        if len(qualified_records) == 0:
            print(f"[Warning] no qualified records found, returning zero sepctrogram")
            return torch.zeros(self.dims).float(), self._get_label(patient_id)

        selected_record = qualified_records[
            qualified_records.etime == qualified_records.etime.max()
        ]["name"].item()

        recording_location = f"{self.root_folder}/{patient_id}/{selected_record}"
        recording_data, channels, sampling_frequency = load_recording_data(
            recording_location
        )

        utility_frequency = get_utility_frequency(recording_location + ".hea")

        # Fix channels, if necessary
        if all(channel in channels for channel in self.used_channels):
            recording_data, channels = reduce_channels(
                recording_data, channels, self.used_channels
            )
            recording_data, sampling_frequency = preprocess_data(
                recording_data, sampling_frequency, utility_frequency
            )

        else:
            print(f"[Chan Warn] Desired channels not available!")

        if sampling_frequency != self.sampling_frequency:
            print(
                f"[Freq Warn] {selected_record} f of {sampling_frequency} != {self.sampling_frequency}"
            )

        recording_length = recording_data.shape[-1] / sampling_frequency

        # Take 5 min from the middle of the recording
        excess_data = recording_data.shape[-1] - (
            self.signal_length.total_seconds() * sampling_frequency
        )
        left_margin = int(np.ceil(excess_data / 2))
        right_margin = int(recording_data.shape[-1] - np.floor(excess_data / 2))
        trimmed_data = recording_data[:, left_margin:right_margin]

        # assert (trimmed_data.shape[-1] / sampling_frequency) == (60 * 5)

        spectrograms = list()

        for channel_idx in range(0, trimmed_data.shape[0]):
            f, t, s = spectrogram(trimmed_data[channel_idx, :], sampling_frequency)
            freq_filter = np.logical_and(f > self.f_min, f < self.f_max)
            s = s[freq_filter]
            f = f[freq_filter]

            with np.errstate(divide="ignore"):
                s = np.log10(s)  # TODO: research alternative approaches

            spectrograms.append(s)

        # channels-first
        X = np.stack(spectrograms, axis=0)

        # deal with -inf
        if (X == -np.inf).all():
            # print(f"[Allzero Warn] {selected_record} random sample has no data")
            X = np.zeros_like(X)
        elif (X == -np.inf).any():
            X[X == -np.inf] = X[X != -np.inf].min()

        return torch.tensor(X).float(), self._get_label(patient_id)


def draw_sample_spectrogram(idx: int):
    ds = SpectrogramDataset()
    X, y = ds[idx]
    f = np.linspace(ds.f_min, ds.f_max, ds.dims[0])
    t = np.linspace(0, ds.signal_length.total_seconds(), ds.dims[1])

    plt.ylabel("f [Hz]")
    plt.xlabel("t [sec]")
    plt.pcolormesh(t, f, X[0])

    plt.savefig("results/sample_spectrogram.png")


if __name__ == "__main__":

    draw_sample_spectrogram(10)
    # ds = SpectrogramDataset()

    # for X, y in ds:
    #     # print(X.shape)
    #     # print(y)
    #     pass
