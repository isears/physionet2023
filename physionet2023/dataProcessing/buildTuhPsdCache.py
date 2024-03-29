import mne
import numpy as np
import torch
from tqdm import tqdm

from physionet2023 import config
from physionet2023.dataProcessing.recordingDatasets import SpectrogramDataset
from physionet2023.dataProcessing.TuhDatasets import TuhBestRecordingDataset

if __name__ == "__main__":
    ds = TuhBestRecordingDataset()

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=1,
    )

    total_psds = 0
    total_patients = len(ds.patient_paths)

    expected_shape = None

    for idx, edf in enumerate(tqdm(dl, total=len(ds.patient_paths))):
        this_edf = edf[0]  # Unwrap batch stack

        # TODO: ds will return float('nan') if no suitable data
        # could think of a better way to handle this
        if this_edf.shape == torch.Size([]):
            continue

        psd, _ = mne.time_frequency.psd_array_welch(
            this_edf.numpy(), sfreq=128.0, fmin=0.5, fmax=30.0, verbose=False
        )

        if expected_shape == None:
            expected_shape = psd.shape
        else:
            assert psd.shape == expected_shape

        total_psds += 1
        torch.save(psd, f"./cache/tuh_psd_cache/{idx:05d}.pt")

    print(f"Saved {total_psds} out of {total_patients} possible")
    print(f"{total_psds / total_patients * 100} % success rate")
