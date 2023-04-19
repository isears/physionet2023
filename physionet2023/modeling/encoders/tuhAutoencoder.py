import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping

from physionet2023 import config
from physionet2023.dataProcessing.recordingDatasets import RecordingDataset
from physionet2023.dataProcessing.TuhDatasets import TuhPreprocessedDataset


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, channels=18):
        super().__init__()

        """
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
        """

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=channels,
                out_channels=channels * 2,
                kernel_size=3,
                padding=1,
                # dilation=9,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(
                in_channels=channels * 2,
                out_channels=channels * 4,
                kernel_size=3,
                padding=1,
                # dilation=9,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            # torch.nn.Conv1d(
            #     in_channels=channels * 4,
            #     out_channels=channels * 8,
            #     kernel_size=5,
            #     stride=2,
            #     padding=1,
            #     # dilation=9,
            # ),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool1d(kernel_size=4),
        )

        self.decoder = torch.nn.Sequential(
            # torch.nn.ConvTranspose1d(
            #     in_channels=channels * 8,
            #     out_channels=channels * 4,
            #     kernel_size=5,
            #     stride=2,
            #     padding=1,
            #     # dilation=9,
            # ),
            # torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(
                in_channels=channels * 4,
                out_channels=channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                # dilation=9,
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(
                in_channels=channels * 2,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
                # dilation=9,
            ),
        )

        self.train_mse = torchmetrics.MeanSquaredError()
        self.valid_mse = torchmetrics.MeanSquaredError()

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)

        self.train_mse.update(preds=x_hat, target=x)

        return loss

    def on_train_epoch_end(self) -> None:
        print(f"\n\ntrain_loss: {self.train_mse.compute()}\n\n")
        self.train_mse.reset()

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss)

        self.valid_mse.update(preds=x_hat, target=x)

        return loss

    def on_validation_epoch_end(self) -> None:
        print(f"\n\nval_loss: {self.valid_mse.compute()}\n\n")
        self.valid_mse.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    tuh_ds = TuhPreprocessedDataset()
    physionet_ds = RecordingDataset()

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
