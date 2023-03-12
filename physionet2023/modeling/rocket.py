from tsai.models.MINIROCKET import *

from physionet2023.dataProcessing.datasets import just_give_me_numpy
from physionet2023.modeling.scoringUtil import \
    compute_challenge_score_regressor

if __name__ == "__main__":
    print("[*] Loading data...")
    X_train, y_train, X_test, y_test = just_give_me_numpy(
        num_examples=10000, resample_factor=5
    )

    print("[*] Fitting model...")
    model = MiniRocketRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    test_score = compute_challenge_score_regressor(y_test, preds)

    print("[+] Done training.")
    print(f"Competition Score (test set): {test_score}")
