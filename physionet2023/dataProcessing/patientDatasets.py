import numpy as np

from physionet2023.dataProcessing.datasets import PatientDataset


class AvgFFTDataset(PatientDataset):
    def __init__(
        self,
        root_folder: str = "./data",
        sample_len=1000,
        patient_ids: list = None,
        quality_cutoff: float = 0.5,
        include_static: bool = True,
    ):
        super().__init__(root_folder, patient_ids, quality_cutoff, include_static)
        self.sample_len = sample_len

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

        return aggregated_fft, patient_metadata["CPC"]


if __name__ == "__main__":
    ds = AvgFFTDataset()

    for X, y in ds:
        print(X)
        break
