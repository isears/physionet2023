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

        return torch.tensor(aggregated_fft, dtype=torch.float32), self._get_label(patient_metadata["Patient"])


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


if __name__ == "__main__":
    ds = AvgFFTDataset()

    for X, y in ds:
        print(X)
        break
