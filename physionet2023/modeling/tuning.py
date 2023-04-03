import optuna
import pytorch_lightning as pl
import torch
from mvtst.models import TSTConfig

# from optuna.integration.wandb import WeightsAndBiasesCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold, train_test_split

import wandb
from physionet2023 import config
from physionet2023.dataProcessing.datasets import PatientDataset
from physionet2023.modeling.convTST import (
    dataloader_factory,
    lightning_tst_factory,
    single_dl_factory,
)


def objective(trial: optuna.Trial) -> float:
    # wandb.init(reinit=True, settings=wandb.Settings(start_method="fork"))
    # Adjust based on GPU capabilities
    max_batch_size = 16

    # Parameters to tune:
    trial.suggest_float("lr", 1e-7, 0.1, log=True)
    trial.suggest_float("dropout", 0.4, 0.9)
    trial.suggest_categorical("d_model_multiplier", [1, 2, 4, 8, 16, 32, 64])
    trial.suggest_int("num_layers", 1, 15)
    trial.suggest_categorical("n_heads", [4, 8, 16, 32, 64])
    trial.suggest_int("dim_feedforward", 64, 1024)
    trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128, 512, 1024])
    trial.suggest_categorical("pos_encoding", ["fixed", "learnable"])
    trial.suggest_categorical("activation", ["gelu", "relu"])
    trial.suggest_categorical("norm", ["BatchNorm", "LayerNorm"])
    trial.suggest_categorical("optimizer_name", ["Adam", "AdamW", "RAdam"])
    trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3])

    print("Starting trial with params:")
    print(trial.params)

    tst_config = TSTConfig(save_path="cache/models/lightningTuning", **trial.params)

    # Maintain an effective batch size of 16 to prevent OOM
    if trial.params["batch_size"] > max_batch_size:
        # NOTE: batch_size should always be divisible by 16 if it's > 16
        assert trial.params["batch_size"] % max_batch_size == 0
        accumulation_coeff = int(trial.params["batch_size"] / max_batch_size)
        tst_config.batch_size = max_batch_size
    else:
        accumulation_coeff = 1

    train_dl, valid_dl = dataloader_factory(
        tst_config, deterministic_split=True, data_path="./data", test_size=0.25
    )

    model = lightning_tst_factory(
        tst_config,
        train_dl.dataset,
    )

    # logger = WandbLogger(
    #     project="physionet2023wandb",
    #     config=tst_config,
    #     group="FftTstTuner",
    #     job_type="train",
    # )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="val_loss", mode="min", dirpath="cache/tuning_models"
    )
    trainer = pl.Trainer(
        max_epochs=5,
        gradient_clip_val=4.0,
        gradient_clip_algorithm="norm",
        accelerator="gpu",
        devices=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=7),
            checkpoint_callback,
        ],
        val_check_interval=0.1,
        enable_checkpointing=True,
        accumulate_grad_batches=accumulation_coeff,
        enable_progress_bar=False,
        logger=False,
    )

    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_dl,
            val_dataloaders=valid_dl,
        )

    except RuntimeError as e:
        del trainer
        del model
        del train_dl
        del valid_dl
        torch.cuda.empty_cache()
        # logger.experiment.finish()

        if "PYTORCH_CUDA_ALLOC_CONF" in str(e):
            print(f"[WARNING] OOM for trial with params {trial.params}")
            return 0.0
        else:
            print(f"[WARNING] Trial failed with params: {trial.params}")
            return 0.0

    # logger.experiment.finish()
    best_state = torch.load(checkpoint_callback.best_model_path)["state_dict"]
    model.load_state_dict(best_state)

    test_results = trainer.test(model=model, dataloaders=valid_dl)

    return test_results[0]["Test CompetitionScore"]


if __name__ == "__main__":
    pruner = None
    # pruner = optuna.pruners.PercentilePruner(25.0)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, timeout=(96.0 * 60.0 * 60.0))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
