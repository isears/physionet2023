import pytorch_lightning as pl
import torch
from mvtst.models.ts_transformer import TSTransformerEncoderClassiregressor
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

from physionet2023 import *
from physionet2023.dataProcessing.patientDatasets import MetadataOnlyDataset
from physionet2023.dataProcessing.recordingDatasets import SpectrogramDataset
from physionet2023.modeling.encoders.tuhSpectrogramAutoencoder import LitAutoEncoder
from physionet2023.modeling.scoringUtil import CompetitionScore


class ConvEncoderTST(pl.LightningModule):
    def __init__(self, tst_config: PhysionetConfig, pretrained=True) -> None:
        super().__init__()

        if pretrained:

            self.autoencoder = LitAutoEncoder.load_from_checkpoint(
                "./cache/tuh_encoder.ckpt"
            )

            self.autoencoder.freeze()
        else:
            self.autoencoder = LitAutoEncoder()

        self.tst = TSTransformerEncoderClassiregressor(
            **tst_config.generate_model_params(), feat_dim=128, max_len=364
        )

        self.scorers = [
            BinaryAUROC().to("cuda"),
            BinaryF1Score().to("cuda"),
            CompetitionScore(),
        ]

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

    def test_step(self, batch, batch_idx):
        X, y = batch
        preds = self.forward(X)
        loss = torch.nn.functional.binary_cross_entropy(preds, y)

        for s in self.scorers:
            s.update(preds, y)

        return loss

    def on_test_epoch_end(self):
        test_competition_score = 0.0
        for s in self.scorers:
            final_score = s.compute()
            self.log(f"Test {s.__class__.__name__}", final_score)

        return test_competition_score

    def on_validation_epoch_end(self) -> None:
        print("\n\n")

        for s in self.scorers:
            print(f"{s.__class__.__name__}: {s.compute()}")
            s.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def forward(self, X):
        encoded = self.autoencoder.encoder(X)
        encoded = encoded.flatten(2)
        padding_masks = torch.ones_like(encoded[:, 0, :]).bool()
        logits = self.tst(encoded.permute(0, 2, 1), padding_masks)
        return torch.sigmoid(logits)


def config_factory():
    problem_params = {
        "lr": 1e-3,
        "dropout": 0.1,
        "d_model_multiplier": 8,
        "num_layers": 3,
        "n_heads": 16,
        "dim_feedforward": 256,
        "pos_encoding": "learnable",
        "activation": "gelu",
        "norm": "LayerNorm",
        "optimizer_name": "AdamW",
        "batch_size": 32,
    }

    tst_config = PhysionetConfig(
        save_path="ConvTst", label_type=LabelType.SINGLECLASS, **problem_params
    )

    return tst_config


def single_dl_factory(
    tst_config: PhysionetConfig, pids: list, data_path: str = "./data", **ds_args
) -> torch.utils.data.DataLoader:
    ds = SpectrogramDataset(
        root_folder=data_path,
        patient_ids=pids,
        label_type=tst_config.label_type,
        preprocess=True,
        # last_only=True,
        # include_static=False,
        quality_cutoff=0.0,
        **ds_args,
    )

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=tst_config.batch_size,
        # Only pin memory if we have GPUs
        pin_memory=(config.gpus_available > 0),
    )

    return dl


def dataloader_factory(
    tst_config: PhysionetConfig,
    data_path: str = "./data",
    deterministic_split=False,
    test_size=0.2,
):
    pids = MetadataOnlyDataset(root_folder=data_path).patient_ids

    if deterministic_split:
        train_pids, valid_pids = train_test_split(
            pids, random_state=1, test_size=test_size
        )
    else:
        train_pids, valid_pids = train_test_split(pids, test_size=test_size)

    train_dl = single_dl_factory(tst_config, train_pids, data_path)
    valid_dl = single_dl_factory(tst_config, valid_pids, data_path)

    for pid in train_dl.dataset.patient_ids:
        assert pid not in valid_dl.dataset.patient_ids

    return train_dl, valid_dl


if __name__ == "__main__":
    train_dl, valid_dl = dataloader_factory(config_factory())

    model = ConvEncoderTST(config_factory(), pretrained=True)
    checkpoint_saver = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="cache/convEncoderTstCheckpoints",
    )

    trainer = pl.Trainer(
        max_epochs=10,
        logger=False,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=True,
                patience=10,
                check_finite=False,
            ),
            checkpoint_saver,
        ],
        enable_checkpointing=True,
        val_check_interval=0.1,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
    trainer.test(
        model=model, dataloaders=valid_dl, ckpt_path=checkpoint_saver.best_model_path
    )
