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
from physionet2023.dataProcessing.datasets import FftDataset, just_give_me_dataloaders
from physionet2023.modeling.scoringUtil import (
    RegressionAUROC,
    RegressionCompetitionScore,
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

        self.scorers = [RegressionAUROC(), RegressionCompetitionScore()]

    def training_step(self, batch, batch_idx):
        X, y, pm, IDs = batch
        preds = torch.squeeze(self.tst(X, pm))

        # loss = self.loss_fn(preds, y)

        loss = torch.nn.functional.mse_loss(preds, y)

        self.log("train_loss", loss, batch_size=self.tst_config.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, pm, IDs = batch
        preds = torch.squeeze(self.tst(X, pm))

        for s in self.scorers:
            s.update(preds, y)

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

    def on_validation_epoch_end(self):
        print("\n\nValidation scores:")

        for s in self.scorers:
            final_score = s.compute()
            print(f"\t{s.__class__.__name__}: {final_score}")
            self.log(f"Validation {s.__class__.__name__}", final_score)
            s.reset()

        print()

    def test_step(self, batch, batch_idx):
        X, y, pm, IDx = batch
        preds = torch.squeeze(self.tst(X, pm))

        for s in self.scorers:
            s.update(preds, y)

        test_loss = torch.nn.functional.mse_loss(preds, y)
        self.log("test_loss", test_loss)

        return {"loss": test_loss, "preds": preds, "target": y}

    def on_test_epoch_end(self):
        for s in self.scorers:
            final_score = s.compute()

            if s.__class__.__name__ == "CompetitionScore":
                test_competition_score = final_score

            self.log(f"Test {s.__class__.__name__}", final_score)

        return test_competition_score

    def configure_optimizers(self):
        return self.tst_config.generate_optimizer(self.parameters())


def lightning_tst_factory(tst_config: TSTConfig, ds):
    tst = TSTransformerEncoderClassiregressor(
        **tst_config.generate_model_params(),
        feat_dim=ds.dataset.features_dim,
        max_len=ds.dataset.sample_len,
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

    training_dl, valid_dl = just_give_me_dataloaders(
        batch_size=tst_config.batch_size,
        sample_len=1000,
        test_subsample=0.25,
        include_static=False,
        ds_cls=FftDataset,
        normalize=False,
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
            EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=15),
            checkpoint_callback,
        ],
        enable_checkpointing=True,
        val_check_interval=0.005,
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
