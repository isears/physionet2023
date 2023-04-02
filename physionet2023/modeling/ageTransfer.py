import torch

import physionet2023.modeling.ageRegressor
import physionet2023.modeling.convTST
from physionet2023.dataProcessing.recordingDatasets import (
    SpectrogramAgeDataset,
    SpectrogramDataset,
)
from physionet2023.modeling import GenericPlTrainer

if __name__ == "__main__":
    print("[*] Loading trained age model")
    params = torch.load("cache/age_models/convTST.pt")
    age_config = physionet2023.modeling.ageRegressor.config_factory()
    _, age_test_dl = physionet2023.modeling.ageRegressor.dataloader_factory(
        age_config, deterministic_split=True
    )
    age_model = physionet2023.modeling.ageRegressor.lightning_tst_factory(
        age_config, age_test_dl.dataset
    )
    age_model.load_state_dict(params)
    age_trainer = GenericPlTrainer(logger=False, enable_progress_bar=True)

    print("[*] Testing trained age model")
    age_trainer.test(age_model, age_test_dl)

    print("[*] Loading weights into CPC model")
    cpc_config = physionet2023.modeling.convTST.config_factory()
    cpc_config.lr = 1e-5
    cpc_train_dl, cpc_test_dl = physionet2023.modeling.convTST.dataloader_factory(
        cpc_config, deterministic_split=True
    )
    cpc_model = physionet2023.modeling.convTST.lightning_tst_factory(
        cpc_config, cpc_train_dl.dataset
    )
    cpc_model.load_state_dict(params)
    cpc_trainer = GenericPlTrainer(logger=False, enable_progress_bar=True)
    cpc_trainer.fit(
        model=cpc_model, train_dataloaders=cpc_train_dl, val_dataloaders=cpc_test_dl
    )
