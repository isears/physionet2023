import pandas as pd
import pytorch_lightning as pl
import torch
from mvtst.models import TSTConfig
from mvtst.models.loss import NoFussCrossEntropyLoss
from mvtst.models.ts_transformer import TSTransformerEncoderClassiregressor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from physionet2023 import config
from physionet2023.dataProcessing.datasets import PatientTrainingDataset
from physionet2023.modeling.scoringUtil import (
    compute_auroc_regressor,
    compute_challenge_score_regressor,
)


class LitTst(pl.LightningModule):
    def __init__(
        self, tst: TSTransformerEncoderClassiregressor, tst_config: TSTConfig
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tst"])
        self.tst = tst

        self.tst_config = tst_config
        # self.loss_fn = NoFussCrossEntropyLoss()

        self.auroc_metric = compute_auroc_regressor
        self.competition_metric = compute_challenge_score_regressor

    def _do_scoring(
        self, batch, batch_idx
    ):  # TODO: use this instead of re-writing every time we want a full eval
        X, y, pm, IDs = batch
        preds = torch.squeeze(self.tst(X, pm))

        val_loss = torch.nn.functional.mse_loss(preds, y)
        self.log(
            "val_loss",
            val_loss,
            # on_step=False,
            # on_epoch=True,
            # prog_bar=True,
            # sync_dist=True,
        )
        return {"loss": val_loss, "preds": preds, "target": y}

    def training_step(self, batch, batch_idx):
        X, y, pm, IDs = batch
        preds = torch.squeeze(self.tst(X, pm))

        # loss = self.loss_fn(preds, y)

        loss = torch.nn.functional.mse_loss(preds, y)

        self.log(
            "train_loss",
            loss,
            batch_size=self.tst_config.batch_size,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, pm, IDs = batch
        preds = torch.squeeze(self.tst(X, pm))

        val_loss = torch.nn.functional.mse_loss(preds, y)
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.tst_config.batch_size,
            # prog_bar=True,
            # sync_dist=True,
        )
        return {"loss": val_loss, "preds": preds, "target": y}

    def validation_step_end(self, outputs):
        auc = self.auroc_metric(
            outputs["target"].cpu().numpy(), outputs["preds"].cpu().numpy()
        )
        self.log("Validation AUC", auc)

        comp = self.competition_metric(
            outputs["target"].cpu().numpy(), outputs["preds"].cpu().numpy()
        )
        self.log("Validation Competition Score", comp)

    def test_step(self, batch, batch_idx):
        X, y, pm, IDx = batch
        preds = torch.squeeze(self.tst(X, pm))
        test_loss = torch.nn.functional.mse_loss(preds, y)
        self.log("test_loss", test_loss)

        return {"loss": test_loss, "preds": preds, "target": y}

    def test_step_end(self, outputs):
        auc = self.auroc_metric(
            outputs["target"].cpu().numpy(), outputs["preds"].cpu().numpy()
        )
        self.log("Test AUC", auc)

        comp = self.competition_metric(
            outputs["target"].cpu().numpy(), outputs["preds"].cpu().numpy()
        )
        self.log("Test Competiton Score", comp)

    def configure_optimizers(self):
        return self.tst_config.generate_optimizer(self.parameters())


def dataloader_factory(batch_size=16):
    ds = PatientTrainingDataset(
        "./data",
        sample_len=4000,
        include_static=False,
    )

    valid_ds_length = int(len(ds) * 0.1)

    train_ds, valid_ds = torch.utils.data.random_split(
        ds,
        [valid_ds_length, len(ds) - valid_ds_length],
        generator=torch.Generator().manual_seed(42),
    )

    training_dl = torch.utils.data.DataLoader(
        train_ds,
        collate_fn=ds.tst_collate,
        num_workers=config.cores_available,
        batch_size=batch_size,
        pin_memory=True,
    )

    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        collate_fn=ds.tst_collate,
        num_workers=config.cores_available,
        batch_size=batch_size,
        pin_memory=True,
    )

    return training_dl, valid_dl


def lightning_tst_factory(tst_config: TSTConfig, dl):
    tst = TSTransformerEncoderClassiregressor(
        **tst_config.generate_model_params(),
        feat_dim=dl.dataset.dataset.features_dim,
        max_len=dl.dataset.dataset.sample_len
    )

    lightning_wrapper = LitTst(tst, tst_config)

    return lightning_wrapper


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

    tst_config = TSTConfig(save_path="lightningTst", **problem_params)

    wandb_logger = WandbLogger(
        project="physionet2023wandb",
        config=tst_config,
        group="TST",
        job_type="train",
    )

    training_dl, valid_dl = dataloader_factory(
        batch_size=tst_config.batch_size,
    )

    model = lightning_tst_factory(tst_config, training_dl)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        max_epochs=100,
        gradient_clip_val=4.0,
        gradient_clip_algorithm="norm",
        accelerator="gpu",
        devices=config.gpus_available,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=5),
            checkpoint_callback,
        ],
        enable_checkpointing=True,
        # val_check_interval=0.005,
        logger=wandb_logger,
        log_every_n_steps=1,
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
