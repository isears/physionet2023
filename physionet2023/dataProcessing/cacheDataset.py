import torch
from tqdm import tqdm

from physionet2023 import config
from physionet2023.dataProcessing.patientDatasets import SpectrogramDataset


class CacheableSpectrogramDataset(SpectrogramDataset):
    first_item_gotten = False

    def __getitem__(self, index: int):
        X, y = super().__getitem__(index)
        pid = self.patient_ids[index]

        if self.first_item_gotten:
            return pid, X, y
        else:  # For compatibility with constructor of SpectrogramDataset
            self.first_item_gotten = True
            return X, y


if __name__ == "__main__":
    ds = CacheableSpectrogramDataset()

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=32,
        # Only pin memory if we have GPUs
        pin_memory=(config.gpus_available > 0),
    )

    for batch_pid, batch_X, batch_y in tqdm(dl, total=len(dl)):
        for batch_idx in range(0, len(batch_pid)):
            torch.save(
                batch_X[batch_idx, :],
                f"cache/physionet_cache/{batch_pid[batch_idx]}_X.pt",
            )
            torch.save(
                batch_y[batch_idx, :],
                f"cache/physionet_cache/{batch_pid[batch_idx]}_y.pt",
            )
