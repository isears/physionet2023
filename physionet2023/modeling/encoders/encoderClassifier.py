import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from torchmetrics.classification import BinaryAUROC

from physionet2023 import LabelType, config
from physionet2023.dataProcessing.patientDatasets import (
    MetadataOnlyDataset,
    SpectrogramDataset,
)
from physionet2023.dataProcessing.cachedDataset import PhysionetPreprocessedDataset

from physionet2023.modeling.encoders.tuhSpectrogramAutoencoder import LitAutoEncoder
from physionet2023.modeling.scoringUtil import CompetitionScore


class encoderClassifier(pl.LightningModule):
    def __init__(self, n_classes=1) -> None:
        super().__init__()

        self.autoencoder = LitAutoEncoder.load_from_checkpoint(
            "./cache/tuh_encoder.ckpt"
        )
        # self.autoencoder.freeze()

        # self.autoencoder = LitAutoEncoder()

        # TODO: need to reduce reliance on magic numbers here
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(9408, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_classes),
            torch.nn.Sigmoid(),
        )

        self.scorers = [CompetitionScore(), BinaryAUROC().to('cuda')]

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = torch.nn.functional.binary_cross_entropy(preds, y)
        self.log("train_loss", loss)

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = torch.nn.functional.binary_cross_entropy(preds, y)
        for s in self.scorers:
            s.update(preds, y)
        self.log("val_loss", loss)

        return loss

    def on_validation_epoch_end(self) -> None:
        print("\n\n")

        for s in self.scorers:
            print(f"{s.__class__.__name__}: {s.compute()}")
            s.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer

    def forward(self, X):
        encoded = self.autoencoder.encoder(X)
        preds = self.fc(encoded.flatten(1))
        return preds


def single_dl_factory(pids: list, data_path: str = "./data", **ds_args):
    ds = PhysionetPreprocessedDataset(
        patient_ids=pids,
    )

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=32,
        # Only pin memory if we have GPUs
        pin_memory=(config.gpus_available > 0),
    )

    return dl


def dataloader_factory(
    data_path: str = "./data",
    deterministic_split=False,
    test_size=0.1,
):
    pids = PhysionetPreprocessedDataset().patient_ids

    if deterministic_split:
        train_pids, valid_pids = train_test_split(
            pids, random_state=42, test_size=test_size
        )
    else:
        train_pids, valid_pids = train_test_split(pids, test_size=test_size)

    train_dl = single_dl_factory(train_pids, data_path)
    valid_dl = single_dl_factory(valid_pids, data_path)

    for pid in train_dl.dataset.patient_ids:
        assert pid not in valid_dl.dataset.patient_ids

    return train_dl, valid_dl


if __name__ == "__main__":
    train_dl, valid_dl = dataloader_factory(deterministic_split=True)

    model = encoderClassifier()

    trainer = pl.Trainer(
        max_epochs=50,
        logger=False,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=True,
                patience=10,
                check_finite=False,
            ),
        ],
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
