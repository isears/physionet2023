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

    total_spectrograms = 0
    total_patients = len(ds.patient_paths)

    for idx, edf in enumerate(tqdm(dl, total=len(ds.patient_paths))):
        this_edf = edf[0]  # Unwrap batch stack

        # TODO: ds will return float('nan') if no suitable data
        # could think of a better way to handle this
        if this_edf.shape == torch.Size([]):
            continue

        assert this_edf.shape[-1] >= (5 * 60 * 128)

        overflow = this_edf.shape[-1] - (5 * 60 * 128)
        left_margin = int(overflow / 2)
        right_margin = this_edf.shape[-1] - left_margin

        this_edf = this_edf[:, left_margin:right_margin]

        assert this_edf.shape[-1] == (5 * 60 * 128)

        spectrogram = SpectrogramDataset._to_spectrogram(this_edf)

        assert spectrogram.shape == torch.Size([4, 58, 171])

        total_spectrograms += 1
        torch.save(spectrogram, f"./cache/tuh_cache/{idx:05d}.pt")

    print(f"Saved {total_spectrograms} out of {total_patients} possible")
    print(f"{total_spectrograms / total_patients * 100} % success rate")
