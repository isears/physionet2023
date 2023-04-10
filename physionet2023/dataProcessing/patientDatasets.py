import mne
import numpy as np
import torch

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

        return torch.tensor(X), self._get_label(patient_id)


if __name__ == "__main__":
    ds = AvgSpectralDensityDataset()

    for X, y in ds:
        print(X)
        break
