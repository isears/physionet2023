import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from mvtst.models import TSTConfig
from mvtst.models.loss import NoFussCrossEntropyLoss
from mvtst.models.ts_transformer import TSTransformerEncoderClassiregressor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import train_test_split

from physionet2023 import config
from physionet2023.dataProcessing.datasets import just_give_me_dataloaders
from physionet2023.modeling.scoringUtil import (
    compute_auroc_regressor,
    compute_challenge_score_regressor,
)


class LitTst(pl.LightningModule):
    def __init__(
        self, tst: TSTransformerEncoderClassiregressor, tst_config: TSTConfig
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.tst = tst

        self.tst_config = tst_config
        # self.loss_fn = NoFussCrossEntropyLoss()

        self.auroc_metric = compute_auroc_regressor
        self.competition_metric = compute_challenge_score_regressor

    def training_step(self, batch, batch_idx):
        X, y, pm, IDs = batch
        preds = torch.squeeze(self.tst(X, pm))

        # loss = self.loss_fn(preds, y)

        loss = torch.nn.functional.mse_loss(preds, y)

        self.log("train_loss", loss)
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
            prog_bar=True,
            sync_dist=True,
        )
        return {"loss": val_loss, "preds": preds, "target": y}

    def validation_step_end(self, outputs):
        auc = self.auroc_metric(
            outputs["target"].cpu().numpy(), outputs["preds"].cpu().numpy()
        )
        self.log("AUC", auc)

        comp = self.competition_metric(
            outputs["target"].cpu().numpy(), outputs["preds"].cpu().numpy()
        )
        self.log("Competition Score", comp)

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
        self.log("AUC", auc)

        comp = self.competition_metric(
            outputs["target"].cpu().numpy(), outputs["preds"].cpu().numpy()
        )
        self.log("Competiton Metric", comp)

    def configure_optimizers(self):
        return self.tst_config.generate_optimizer(self.parameters())


def lightning_tst_factory(tst_config: TSTConfig, ds):
    tst = TSTransformerEncoderClassiregressor(
        **tst_config.generate_model_params(),
        feat_dim=ds.dataset.features_dim,
        max_len=ds.dataset.sample_len
    )

    lightning_wrapper = LitTst(tst, tst_config)

    return lightning_wrapper


if __name__ == "__main__":
    problem_params = {
        "lr": 0.001,
        "dropout": 0.1,
        "d_model_multiplier": 8,
        "num_layers": 3,
        "n_heads": 16,
        "dim_feedforward": 256,
        "batch_size": 86,
        "pos_encoding": "learnable",
        "activation": "gelu",
        "norm": "LayerNorm",
        "optimizer_name": "AdamW",
        "batch_size": 16,
    }
    tst_config = TSTConfig(save_path="lightningTst", **problem_params)
    training_dl, valid_dl = just_give_me_dataloaders(
        batch_size=tst_config.batch_size, sample_len=500
    )

    model = lightning_tst_factory(tst_config, training_dl)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        max_epochs=10,
        gradient_clip_val=4.0,
        gradient_clip_algorithm="norm",
        accelerator="gpu",
        devices=config.gpus_available,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", verbose=True),
            checkpoint_callback,
        ],
        enable_checkpointing=True,
    )

    trainer.fit(
        model=model,
        train_dataloaders=training_dl,
        val_dataloaders=valid_dl,
    )

    best_model = LitTst.load_from_checkpoint(checkpoint_callback.best_model_path)
    results = trainer.test(model=best_model, dataloaders=valid_dl)

    print(type(results))
    print(results)
