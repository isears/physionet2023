import torch
import glob
import sklearn

class PhysionetPreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, path="cache/physionet_cache", shuffle=True, patient_ids: list = None) -> None:
        super().__init__()
        self.path = path
        self.patient_ids = [f.split("/")[-1].split("_")[0] for f in glob.glob(f"{path}/*_X.pt")]

        if patient_ids != None:
            self.patient_ids = [p for p in self.patient_ids if p in patient_ids]

        if shuffle:
            self.patient_ids = sklearn.utils.shuffle(self.patient_ids, random_state=42)

    def __len__(self):
        return len(self.patient_ids)
    

    def __getitem__(self, index):
        pid = self.patient_ids[index]

        X = torch.load(f"{self.path}/{pid}_X.pt")
        y = torch.load(f"{self.path}/{pid}_y.pt")

        y = (y > 2).float()

        chan_means = X.mean(dim=1, keepdim=True)
        chan_stds = X.std(dim=1, keepdim=True)
        X_norm = (X - chan_means) / chan_stds

        if torch.isnan(X_norm).any():
            X_norm[torch.isnan(X_norm)] = 0.0

        if torch.isinf(X_norm).any():
            X_norm[torch.isinf(X_norm)] = 0.0

        return X_norm, y
    

if __name__ == "__main__":
    ds = PhysionetPreprocessedDataset()
    print(len(ds))

    X, y = ds[0]