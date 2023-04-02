import pytorch_lightning as pl
import torch
from mvtst.models import TSTConfig
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryPrecision

from physionet2023 import config
from physionet2023.modeling.scoringUtil import (
    CompetitionScore,
    PrintableBinaryConfusionMatrix,
    RegressionAUROC,
    RegressionCompetitionScore,
)


class GenericPlTst(pl.LightningModule):
    def __init__(self, tst, tst_config: TSTConfig) -> None:
        super().__init__()
        self.tst = tst

        self.tst_config = tst_config
        # self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1]))
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        if config.gpus_available > 0:
            device = "cuda"  # TODO: this will break if using ROCm (AMD)
        else:
            device = "cpu"

        self.scorers = [
            BinaryAUROC(),
            BinaryPrecision().to(device),
            BinaryAccuracy().to(device),
            # PrintableBinaryConfusionMatrix().to(device),
            CompetitionScore(),
        ]

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self.forward(X)

        loss = self.loss_fn(preds, y)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=self.tst_config.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self.forward(X)

        loss = self.loss_fn(preds, y)

        for s in self.scorers:
            s.update(preds, y)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.tst_config.batch_size,
        )

        return loss

    def on_validation_epoch_end(self):
        print("\n\nValidation scores:")

        for s in self.scorers:
            final_score = s.compute()
            print(f"\t{s.__class__.__name__}: {final_score}")
            self.log(f"Validation {s.__class__.__name__}", final_score)
            s.reset()

        print()

    def test_step(self, batch, batch_idx):
        X, y = batch
        preds = self.forward(X)
        loss = self.loss_fn(preds, y)

        for s in self.scorers:
            s.update(preds, y)

        return loss

    def on_test_epoch_end(self):
        test_competition_score = 0.0
        for s in self.scorers:
            final_score = s.compute()

            # TODO: this part needs to go into the Regressor child class
            if s.__class__.__name__ == "RegressionCompetitionScore":
                test_competition_score = final_score
                # For the tuner
                self.log(f"Test CompetitionScore", test_competition_score)

            self.log(f"Test {s.__class__.__name__}", final_score)

        return test_competition_score

    def forward(self, X):
        logits = self.tst(X)
        return torch.sigmoid(logits)

    def configure_optimizers(self):
        return self.tst_config.generate_optimizer(self.parameters())


class GenericPlRegressor(GenericPlTst):
    def __init__(self, tst, tst_config: TSTConfig) -> None:
        super().__init__(tst, tst_config)

        self.scorers = [RegressionAUROC(), RegressionCompetitionScore()]

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, X):
        return self.tst(X)


class GenericPlTrainer(pl.Trainer):
    def __init__(
        self,
        logger=None,
        enable_progress_bar=False,
        val_check_interval=0.1,
        es_patience=7,
        **extra_args,
    ):
        self.my_checkpoint_callback = ModelCheckpoint(
            save_top_k=1, monitor="val_loss", mode="min"
        )

        if config.gpus_available > 0:
            accelerator = "gpu"
            devices = config.gpus_available
        else:
            accelerator = "cpu"
            devices = 1

        super().__init__(
            max_epochs=10,
            gradient_clip_val=4.0,
            gradient_clip_algorithm="norm",
            accelerator=accelerator,
            devices=devices,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    verbose=True,
                    patience=es_patience,
                    check_finite=False,
                ),
                self.my_checkpoint_callback,
            ],
            enable_checkpointing=True,
            enable_progress_bar=enable_progress_bar,
            # For when doing sample-based datasets
            val_check_interval=val_check_interval,
            # log_every_n_steps=7,
            logger=logger,
            **extra_args,
        )

    def get_best_params(self):
        return torch.load(self.my_checkpoint_callback.best_model_path)["state_dict"]
