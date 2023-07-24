import torch
import glob
from sklearn.utils import shuffle

class PhysionetPreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, path="cache/physionet_cache", shuffle=True) -> None:
        super().__init__()
        self.path = path
        self.pids = [f.split("/")[-1].split("_")[0] for f in glob.glob(f"{path}/*_X.pt")]

        if shuffle:
            self.pids = shuffle(self.pids, random_state=42)

    def __len__(self):
        return len(self.pids)
    

    def __getitem__(self, index):
        pid = self.pids[index]

        X = torch.load(f"{self.path}/{pid}_X.pt")
        y = torch.load(f"{self.path}/{pid}_y.pt")

        return X, y
    

if __name__ == "__main__":
    ds = PhysionetPreprocessedDataset()
    print(len(ds))