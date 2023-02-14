from sklearn.metrics import roc_auc_score
from tsai.models.MINIROCKET import *

from physionet2023.dataProcessing.datasets import just_give_me_numpy

if __name__ == "__main__":
    print("[*] Loading data...")
    X_train, y_train, X_test, y_test = just_give_me_numpy(num_examples=10000)

    print("[*] Fitting model...")
    model = MiniRocketRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    binary_preds = (preds - 1) / 4
    binary_labels = y_test > 2

    print(f"[+] Done: AUROC (test set): {roc_auc_score(binary_labels, binary_preds)}")
