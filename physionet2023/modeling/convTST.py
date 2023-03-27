import pytorch_lightning as pl
import torch
from mvtst.models import TSTConfig
from mvtst.models.loss import NoFussCrossEntropyLoss
from mvtst.models.ts_transformer import ConvTST, TSTransformerEncoderClassiregressor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from physionet2023 import config
from physionet2023.dataProcessing.datasets import PatientDataset
from physionet2023.dataProcessing.recordingDatasets import SpectrogramDataset
from physionet2023.modeling.scoringUtil import (
    ClassifierAUROC,
    ClassifierCompetitionScore,
    CompetitionScore,
    RegressorAUROC,
)


class plConvTst(pl.LightningModule):
    def __init__(self, tst: ConvTST, tst_config: TSTConfig) -> None:
        super().__init__()
        # TODO: damned if you do damned if you don't
        # self.save_hyperparameters(ignore=["tst"])
        self.tst = tst

        self.tst_config = tst_config
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.auroc_scorer = ClassifierAUROC()
        self.competition_scorer = ClassifierCompetitionScore()

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = torch.squeeze(self.tst(X))

        loss = self.loss_fn(preds, y)

        self.log("train_loss", loss, batch_size=self.tst_config.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = torch.squeeze(self.tst(X))

        loss = self.loss_fn(preds, y)
        self.auroc_scorer.update(preds, y)
        self.competition_scorer.update(preds, y)

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = torch.tensor(validation_step_outputs).mean()
        val_auroc = self.auroc_scorer.compute()
        val_competition_score = self.competition_scorer.compute()

        self.log("Validation AUC", val_auroc)
        self.log("Validation Competition Score", val_competition_score)
        self.log("val_loss", val_loss)

        self.auroc_scorer.reset()
        self.competition_scorer.reset()

        print(
            f"\nValidation loss: {val_loss:.4}, auroc: {val_auroc:.4}, competition: {val_competition_score}"
        )

    def test_step(self, batch, batch_idx):
        X, y = batch
        preds = torch.squeeze(self.tst(X))

        loss = self.loss_fn(preds, y)
        self.auroc_scorer.update(preds, y)
        self.competition_scorer.update(preds, y)

        return loss

    def test_epoch_end(self, test_step_outputs):
        test_loss = torch.tensor(test_step_outputs).mean()
        test_auroc = self.auroc_scorer.compute()
        test_competition_score = self.competition_scorer.compute()

        self.log("Test AUC", test_auroc)
        self.log("Test Competition Score", test_competition_score)
        self.log("test_loss", test_loss)

        self.auroc_scorer.reset()
        self.competition_scorer.reset()

        print(
            f"\Test loss: {test_loss:.4}, auroc: {test_auroc:.4}, competition: {test_competition_score}"
        )
        return test_competition_score

    def configure_optimizers(self):
        return self.tst_config.generate_optimizer(self.parameters())


def lightning_tst_factory(tst_config: TSTConfig, ds):
    tst = ConvTST(
        **tst_config.generate_model_params(),
        spectrogram_dims=ds.dims,
        feat_dim=ds.features_dim,
    )

    lightning_wrapper = plConvTst(tst, tst_config)

    return lightning_wrapper


def dataloader_factory(tst_config: TSTConfig, deterministic_split=False):
    pids = PatientDataset().patient_ids

    if deterministic_split:
        train_pids, valid_pids = train_test_split(pids, random_state=42)
    else:
        train_pids, valid_pids = train_test_split(pids)

    train_ds = SpectrogramDataset(patient_ids=train_pids, for_classification=True)
    valid_ds = SpectrogramDataset(patient_ids=valid_pids, for_classification=True)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        num_workers=config.cores_available,
        batch_size=tst_config.batch_size,
        pin_memory=True,
    )

    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        num_workers=config.cores_available,
        batch_size=tst_config.batch_size,
        pin_memory=True,
    )

    return train_dl, valid_dl


if __name__ == "__main__":
    problem_params = {
        "lr": 1e-4,
        "dropout": 0.1,
        "d_model_multiplier": 8,
        "num_layers": 1,
        "n_heads": 8,
        "dim_feedforward": 256,
        "pos_encoding": "learnable",
        "activation": "gelu",
        "norm": "LayerNorm",
        "optimizer_name": "AdamW",
        "batch_size": 16,
    }

    tst_config = TSTConfig(save_path="ConvTst", num_classes=5, **problem_params)

    wandb_logger = WandbLogger(
        project="physionet2023wandb",
        config=tst_config,
        group="ConvTST_classifier",
        job_type="train",
    )

    training_dl, valid_dl = dataloader_factory(tst_config)

    model = lightning_tst_factory(tst_config, training_dl.dataset)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        max_epochs=10,
        gradient_clip_val=4.0,
        gradient_clip_algorithm="norm",
        accelerator="gpu",
        devices=config.gpus_available,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=5),
            checkpoint_callback,
        ],
        enable_checkpointing=True,
        # For when doing sample-based datasets
        val_check_interval=0.1,
        # log_every_n_steps=7,
        logger=wandb_logger,
    )

    trainer.fit(
        model=model,
        train_dataloaders=training_dl,
        val_dataloaders=valid_dl,
    )

    # best_model = LitTst.load_from_checkpoint(checkpoint_callback.best_model_path)
    # results = trainer.test(model=best_model, dataloaders=valid_dl)

    # print(type(results))
    # print(results)
