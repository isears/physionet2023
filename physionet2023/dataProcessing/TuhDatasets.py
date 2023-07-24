import glob

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch

from helper_code import preprocess_data
from physionet2023.dataProcessing.patientDatasets import SpectrogramDataset
from physionet2023 import config
from tqdm import tqdm

class TuhPatientDataset(torch.utils.data.Dataset):
    # PIDs with insufficient channels to reconstruct the physionet bipolar channels
    bad_pids = ["aaaaahie", "aaaaaskf", "aaaaaria", "aaaaamqq"]

    le_anodes = [
        "EEG FP1-LE",
        "EEG F7-LE",
        "EEG T3-LE",
        "EEG T5-LE",
        "EEG FP2-LE",
        "EEG F8-LE",
        "EEG T4-LE",
        "EEG T6-LE",
        "EEG FP1-LE",
        "EEG F3-LE",
        "EEG C3-LE",
        "EEG P3-LE",
        "EEG FP2-LE",
        "EEG F4-LE",
        "EEG C4-LE",
        "EEG P4-LE",
        "EEG FZ-LE",
        "EEG CZ-LE",
    ]

    le_cathodes = [
        "EEG F7-LE",
        "EEG T3-LE",
        "EEG T5-LE",
        "EEG O1-LE",
        "EEG F8-LE",
        "EEG T4-LE",
        "EEG T6-LE",
        "EEG O2-LE",
        "EEG F3-LE",
        "EEG C3-LE",
        "EEG P3-LE",
        "EEG O1-LE",
        "EEG F4-LE",
        "EEG C4-LE",
        "EEG P4-LE",
        "EEG O2-LE",
        "EEG CZ-LE",
        "EEG PZ-LE",
    ]

    ref_anodes = [
        "EEG FP1-REF",
        "EEG F7-REF",
        "EEG T3-REF",
        "EEG T5-REF",
        "EEG FP2-REF",
        "EEG F8-REF",
        "EEG T4-REF",
        "EEG T6-REF",
        "EEG FP1-REF",
        "EEG F3-REF",
        "EEG C3-REF",
        "EEG P3-REF",
        "EEG FP2-REF",
        "EEG F4-REF",
        "EEG C4-REF",
        "EEG P4-REF",
        "EEG FZ-REF",
        "EEG CZ-REF",
    ]

    ref_cathodes = [
        "EEG F7-REF",
        "EEG T3-REF",
        "EEG T5-REF",
        "EEG O1-REF",
        "EEG F8-REF",
        "EEG T4-REF",
        "EEG T6-REF",
        "EEG O2-REF",
        "EEG F3-REF",
        "EEG C3-REF",
        "EEG P3-REF",
        "EEG O1-REF",
        "EEG F4-REF",
        "EEG C4-REF",
        "EEG P4-REF",
        "EEG O2-REF",
        "EEG CZ-REF",
        "EEG PZ-REF",
    ]

    channel_names = [
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

    used_channels = [
        "EEG F3-REF",
        "EEG F3-LE",
        "EEG F4-REF",
        "EEG F4-LE",
        "EEG P3-REF",
        "EEG P3-LE",
        "EEG P4-REF",
        "EEG P4-LE",
    ]

    def __init__(self, tuh_dir="./tuh/edf", fft_coeff=10) -> None:
        super().__init__()
        self.root_dir = tuh_dir

        self.patient_paths = list()
        for subdir in glob.glob(f"{self.root_dir}/*"):
            for patient_dir in glob.glob(f"{subdir}/*"):
                if len(glob.glob(f"{patient_dir}/*/*/*.edf")) > 0:

                    if not self.__is_bad_pid(patient_dir):
                        self.patient_paths.append(patient_dir)

        if not len(self.patient_paths) > 0:
            raise ValueError(f"No data found in path: {tuh_dir}")

        self.fft_coeff = fft_coeff

    def __is_bad_pid(self, patient_dir: str):
        for p in self.bad_pids:
            if p in patient_dir:
                return True

        return False

    def __len__(self):
        return len(self.patient_paths)

    def _get_physionet_channels(self, edf_path: str):
        edf_obj = mne.io.read_raw_edf(
            edf_path, verbose=False, preload=True, include=self.used_channels
        )
        eeg_waveform = edf_obj.get_data()
        assert (
            eeg_waveform.shape[0] == 4
        ), f"Actual channels {eeg_waveform.shape[0]} != expected {4}"

        return eeg_waveform, edf_obj.info["sfreq"]

    def _get_edf_obj(self, index: int):
        # TODO: just take first edf for now but later could go for longest and / or highest-quality
        all_possible_edfs = glob.glob(f"{self.patient_paths[index]}/*/*/*.edf")

        assert len(all_possible_edfs) > 0, "0 available edfs"

        edf_obj = mne.io.read_raw_edf(all_possible_edfs[0], verbose=False)
        return edf_obj

    def __getitem__(self, index: int):
        # TODO: just take first edf for now but later could go for longest and / or highest-quality
        all_possible_edfs = glob.glob(f"{self.patient_paths[index]}/*/*/*.edf")
        assert len(all_possible_edfs) > 0, "0 available EDFs"
        eeg_waveform, sfreq = self._get_physionet_channels(all_possible_edfs[0])

        # TODO: utility frequency of 50 Hz is just an educated guess
        processed_signal, f = preprocess_data(
            eeg_waveform, sampling_frequency=sfreq, utility_frequency=50.0
        )

        return torch.tensor(processed_signal)


class TuhBestRecordingDataset(TuhPatientDataset):
    # NOTE: output could be none if no signal satisfies requirements
    def __getitem__(self, index: int):

        try:
            all_possible_edfs = [
                self._get_physionet_channels(p)
                for p in glob.glob(f"{self.patient_paths[index]}/*/*/*.edf")
            ]

            for edf, sfreq in all_possible_edfs:
                # Min 10 minutes data
                if (edf.shape[-1] / sfreq) >= (10 * 60):
                    # TODO: utility frequency is just a guess
                    data, f = preprocess_data(
                        edf, sampling_frequency=sfreq, utility_frequency=50.0
                    )

                    assert f == 128.0

                    return data

            else:
                return float("nan")
        except Exception as e:
            print(f"Warning, caught exception: {e} (index {index})")
            return float("nan")


class TuhPreprocessedDataset(torch.utils.data.Dataset):

    seq_len = 30000

    def __init__(self, path="cache/tuh_cache") -> None:
        super().__init__()

        self.fnames = [f for f in glob.glob(f"{path}/*.pt") if not f.startswith("_")]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index: int):
        X = torch.load(self.fnames[index]).float()

        chan_means = X.mean(dim=1, keepdim=True)
        chan_stds = X.std(dim=1, keepdim=True)
        X_norm = (X - chan_means) / chan_stds

        if torch.isnan(X_norm).any():
            X_norm[torch.isnan(X_norm)] = 0.0

        if torch.isinf(X_norm).any():
            X_norm[torch.isinf(X_norm)] = 0.0

        return X_norm


class TuhPsdCacheDataset(torch.utils.data.Dataset):
    def __init__(self, path="cache/tuh_psd_cache"):
        super().__init__()
        self.fnames = [f for f in glob.glob(f"{path}/*.pt") if not f.startswith("_")]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index: int):
        psd = torch.tensor(torch.load(self.fnames[index]))

        psd_norm = (psd - psd.mean()) / psd.std()

        assert not psd_norm.isnan().any()

        return psd_norm

def load_all_psd():
    ds = TuhPsdCacheDataset()

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=32,
    )

    batches = list()

    for batch in tqdm(dl, total = len(dl)):
        batches.append(batch)

    return torch.concat(batches, dim=0).float()

def draw_sample_spectrogram(idx: int):
    ds = TuhPreprocessedDataset()
    X = ds[idx]
    f = np.linspace(0.5, 30, X.shape[1])
    t = np.linspace(0, 5 * 60, X.shape[2])

    plt.ylabel("f [Hz]")
    plt.xlabel("t [sec]")
    plt.pcolormesh(t, f, X[0])

    plt.savefig("results/sample_tuh_spectrogram.png")


if __name__ == "__main__":
    draw_sample_spectrogram(960)
