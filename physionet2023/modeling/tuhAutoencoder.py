import pytorch_lightning as pl
import torch

from physionet2023 import config
from physionet2023.dataProcessing.patientDatasets import AvgSpectralDensityDataset
from physionet2023.dataProcessing.TuhDatasets import TuhPreprocessedDataset


# https://www.tutorialspoint.com/how-to-implementing-an-autoencoder-in-pytorch
class Autoenc(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_dim),
            # torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def training_step(self, batch, batch_idx):
        # TODO: for now just using first EEG channel
        x = batch[:, 0, :]
        x_hat = self.autoencoder(x)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: for now just using first EEG channel
        x = batch[:, 0, :]
        x_hat = self.autoencoder(x)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    tuh_ds = TuhPreprocessedDataset()
    physionet_ds = AvgSpectralDensityDataset()

    tuh_dl = torch.utils.data.DataLoader(
        tuh_ds,
        num_workers=config.cores_available,
        batch_size=16,
        # Only pin memory if we have GPUs
        pin_memory=(config.gpus_available > 0),
    )

    physionet_dl = torch.utils.data.DataLoader(
        physionet_ds,
        num_workers=config.cores_available,
        batch_size=16,
        # Only pin memory if we have GPUs
        pin_memory=(config.gpus_available > 0),
    )

    # TODO: actually get input dim from dataset
    autoencoder = LitAutoEncoder(autoencoder=Autoenc(input_dim=296))

    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, logger=False)
    trainer.fit(
        model=autoencoder, train_dataloaders=tuh_dl, val_dataloaders=physionet_dl
    )
