import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping

from physionet2023 import config
from physionet2023.dataProcessing.patientDatasets import AvgSpectralDensityDataset
from physionet2023.dataProcessing.TuhDatasets import TuhPreprocessedDataset

# https://www.tutorialspoint.com/how-to-implementing-an-autoencoder-in-pytorch


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(296, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 296),
            # torch.nn.Sigmoid(),
        )

        self.train_mse = torchmetrics.MeanSquaredError()
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.train_losses = list()
        self.valid_losses = list()

    def training_step(self, batch, batch_idx):
        # TODO: for now just using first EEG channel
        x = batch[:, 0, :]
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)

        self.train_mse.update(preds=x_hat, target=x)

        self.train_losses.append(loss)

        return loss

    def on_train_epoch_end(self) -> None:
        print(f"\n\ntrain_loss: {sum(self.train_losses) / len(self.train_losses)}\n\n")
        del self.train_losses
        self.train_losses = list()
        self.train_mse.reset()

    def validation_step(self, batch, batch_idx):
        # TODO: for now just using first EEG channel
        x, _ = batch
        x = x[:, 0, :]
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss)

        self.valid_mse.update(preds=x_hat, target=x)

        self.valid_losses.append(loss)

        return loss

    def on_validation_epoch_end(self) -> None:
        print(f"\n\nval_loss: {sum(self.valid_losses) / len(self.valid_losses)}\n\n")
        del self.valid_losses
        self.valid_losses = list()
        self.valid_mse.reset()

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
    autoencoder = LitAutoEncoder()

    trainer = pl.Trainer(
        # limit_train_batches=100,
        max_epochs=15,
        logger=False,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=True,
                patience=3,
                check_finite=False,
            ),
        ],
        default_root_dir="cache/encoder_models",
    )
    trainer.fit(
        model=autoencoder, train_dataloaders=tuh_dl, val_dataloaders=physionet_dl
    )
