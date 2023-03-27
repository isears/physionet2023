import optuna
import pytorch_lightning as pl
import torch
from mvtst.models import TSTConfig
from optuna.integration.wandb import WeightsAndBiasesCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from physionet2023 import config
from physionet2023.modeling.convTST import dataloader_factory, lightning_tst_factory

wandbc = WeightsAndBiasesCallback(
    wandb_kwargs={
        "project": "physionet2023wandb",
        "group": "Optuna",
        "job_type": "tune",
    },
    as_multirun=True,
)


@wandbc.track_in_wandb()
def objective(trial: optuna.Trial) -> float:
    # Parameters to tune:
    trial.suggest_float("lr", 1e-10, 0.1, log=True)
    trial.suggest_float("dropout", 0.01, 0.7)
    trial.suggest_categorical("d_model_multiplier", [1, 2, 4, 8, 16, 32, 64])
    trial.suggest_int("num_layers", 1, 15)
    trial.suggest_categorical("n_heads", [4, 8, 16, 32, 64])
    trial.suggest_int("dim_feedforward", 128, 512)
    trial.suggest_categorical("batch_size", list(range(16, 1040, 16)) + [2, 4, 8])
    trial.suggest_categorical("pos_encoding", ["fixed", "learnable"])
    trial.suggest_categorical("activation", ["gelu", "relu"])
    trial.suggest_categorical("norm", ["BatchNorm", "LayerNorm"])
    trial.suggest_categorical("optimizer_name", ["AdamW", "PlainRAdam", "RAdam"])
    trial.suggest_categorical("weight_decay", [1e-3, 1e-2, 1e-1, 0])

    tst_config = TSTConfig(
        save_path="cache/models/lightningTuning", num_classes=5, **trial.params
    )

    # Maintain an effective batch size of 16 to prevent OOM
    if trial.params["batch_size"] > 16:
        # NOTE: batch_size should always be divisible by 16 if it's > 16
        assert trial.params["batch_size"] % 16 == 0
        accumulation_coeff = int(trial.params["batch_size"] / 16)
        tst_config.batch_size = 16
    else:
        accumulation_coeff = 1

    # TODO: is determinism best here or no?
    train_dl, valid_dl = dataloader_factory(tst_config, deterministic_split=False)

    model = lightning_tst_factory(
        tst_config,
        train_dl.dataset,
    )

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
    trainer = pl.Trainer(
        max_epochs=100,
        gradient_clip_val=4.0,
        gradient_clip_algorithm="norm",
        accelerator="gpu",
        devices=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=10),
            checkpoint_callback,
        ],
        val_check_interval=0.1,
        enable_checkpointing=True,
        accumulate_grad_batches=accumulation_coeff,
        progress_bar_refresh_rate=0,
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
        if "PYTORCH_CUDA_ALLOC_CONF" in str(e):
            print(f"[WARNING] Caught OOM, skipping this trial")
            return None
        else:
            raise e

    best_state = torch.load(checkpoint_callback.best_model_path)["state_dict"]
    model.load_state_dict(best_state)

    test_results = trainer.test(model=model, dataloaders=valid_dl)

    return test_results[0]["Test Competition Score"]


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    pruner = None
    # pruner = optuna.pruners.PercentilePruner(25.0)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, callbacks=[wandbc])

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
