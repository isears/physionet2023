import random

import numpy as np
import torch
from scipy.signal import decimate

from physionet2023.dataProcessing.recordingDatasets import RecordingDataset


class SampleDataset(RecordingDataset):
    def __init__(
        self,
        sample_len=1000,
        resample_factor: int = None,
        normalize=True,
        **super_kwargs,
    ):
        super().__init__(include_static=False, **super_kwargs)
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
            with np.errstate(divide="ignore"):
                sample_data = (sample_data - sample_data.mean()) / (sample_data.std())
                sample_data = np.nan_to_num(
                    sample_data
                )  # Weirdly some data has std of 0

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
            self._get_label(patient_id),
        )


class FftDataset(SampleDataset):
    def __init__(self, patient_ids: list, **super_kwargs):
        super().__init__(patient_ids=patient_ids, **super_kwargs)

        if self.resample_factor:
            raise NotImplementedError("FFT automatically downsamples to sample_len")

    def __getitem__(self, index: int):
        # TODO: we can do more here by compressing the entire sequence down to an appropriate length by downsampling the FFT
        patient_id, recording_id, sample_idx = self.patient_recording_sample_index[
            index
        ]
        patient_metadata = self._load_patient_metadata(patient_id)
        recording_data = self._load_single_recording(patient_id, recording_id)

        sample_data = recording_data[:, sample_idx : sample_idx + self.sample_len]
        X_fft = np.zeros_like(sample_data)
        for channel_idx in range(0, sample_data.shape[0]):
            X_fft[channel_idx, :] = np.abs(np.fft.fft(sample_data[channel_idx, :]))

        # fft_resample_factor = self.sample_len / self.full_record_len
        # X_fft_downsampled = decimate(X_fft, fft_resample_factor)

        static_data = torch.tensor(
            [
                converter(patient_metadata[f])
                for f, converter in self.static_features.items()
            ]
        )

        # NOTE: copy was necessary to prevent "negative stride error" after decimation
        # Not sure what the performance implications are
        return (
            torch.tensor(X_fft.copy()),
            # torch.nan_to_num(static_data, 0.0),
            self._get_label(patient_id),
        )
