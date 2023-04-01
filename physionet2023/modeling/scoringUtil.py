import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC, BinaryConfusionMatrix


# The original compute challenge score function from the physionet repository
def compute_challenge_score(labels, outputs):
    assert len(labels) == len(outputs)
    num_instances = len(labels)

    # Collect the unique output values as the thresholds for the positive and negative classes.
    thresholds = np.unique(outputs)
    thresholds = np.append(thresholds, thresholds[-1] + 1)
    thresholds = thresholds[::-1]
    num_thresholds = len(thresholds)

    idx = np.argsort(outputs)[::-1]

    # Initialize the TPs, FPs, FNs, and TNs with no positive outputs.
    tp = np.zeros(num_thresholds)
    fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds)
    tn = np.zeros(num_thresholds)

    tp[0] = 0
    fp[0] = 0
    fn[0] = np.sum(labels == 1)
    tn[0] = np.sum(labels == 0)

    # Update the TPs, FPs, FNs, and TNs using the values at the previous threshold.
    i = 0
    for j in range(1, num_thresholds):
        tp[j] = tp[j - 1]
        fp[j] = fp[j - 1]
        fn[j] = fn[j - 1]
        tn[j] = tn[j - 1]

        while i < num_instances and outputs[idx[i]] >= thresholds[j]:
            if labels[idx[i]]:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Compute the TPRs and FPRs.
    tpr = np.zeros(num_thresholds)
    fpr = np.zeros(num_thresholds)
    for j in range(num_thresholds):
        if tp[j] + fn[j] > 0:
            tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            fpr[j] = float(fp[j]) / float(tp[j] + fn[j])
        else:
            tpr[j] = float("nan")
            fpr[j] = float("nan")

    # Find the largest TPR such that FPR < 0.05.
    max_fpr = 0.05
    max_tpr = float("nan")
    if np.any(fpr <= max_fpr):
        indices = np.where(fpr <= max_fpr)
        max_tpr = np.max(tpr[indices])

    return max_tpr


# Coerce regression-style outputs into binary for scoring
# NOTE: This may not actually be the optimal way to post-process regression results
def compute_challenge_score_regressor(labels, outputs):
    binary_outputs = np.clip((outputs - 1) / 4, 0.0, 1.0)
    binary_labels = (labels > 2).astype("float")

    return compute_challenge_score(binary_labels, binary_outputs)


def compute_auroc_regressor(labels, outputs):
    binary_outputs = np.clip((outputs - 1) / 4, 0.0, 1.0)
    binary_labels = (labels > 2).astype("float")

    try:
        return roc_auc_score(binary_labels, binary_outputs)
    except (ValueError, TypeError):
        return 0.5


def regression_to_probability(regression_pred, clip=True):
    if clip:
        regression_pred = torch.clip(regression_pred, 1.0, 5.0)

    outcome_probability = torch.where(
        regression_pred >= 2.5,
        ((regression_pred - 2.5) * 0.2) + 0.5,
        (regression_pred - 1) * (1 / 3),
    )

    return outcome_probability


class RegressionAUROC(BinaryAUROC):
    def __init__(self) -> None:
        super().__init__()

    def update(self, preds, target):
        probability_outputs = regression_to_probability(preds)
        binary_labels = (target > 2).float()

        return super().update(probability_outputs, binary_labels)


class ClassifierAUROC(BinaryAUROC):
    def __init__(self):
        super().__init__()

    def update(self, preds, target):
        # Predicting probability of "poor" outcome (CPC 3, 4, or 5)
        poor_outcome_probability = torch.sigmoid(preds)

        return super().update(poor_outcome_probability, target)


class CompetitionScore(Metric):
    is_differentiable = False
    higher_is_better = True

    def __init__(self) -> None:
        super().__init__()

        self.all_preds = list()
        self.all_labels = list()

    def update(self, preds, target):
        self.all_preds.append(preds)
        self.all_labels.append(target)

    def compute(self):
        # TODO: implementing competition score in pytorch would allow computation on GPU instead of having to move to CPU
        preds_np = torch.cat(self.all_preds).cpu().numpy()
        labels_np = torch.cat(self.all_labels).cpu().numpy()

        if labels_np.ndim > 1:
            labels_np = np.squeeze(labels_np)
        if preds_np.ndim > 1:
            preds_np = np.squeeze(preds_np)

        return compute_challenge_score(labels_np, preds_np)

    def reset(self):
        del self.all_preds
        del self.all_labels

        self.all_preds = list()
        self.all_labels = list()

        return super().reset()


class RegressionCompetitionScore(CompetitionScore):
    def update(self, preds, target):
        probability_outputs = regression_to_probability(preds)
        binary_labels = (target > 2).float()
        return super().update(probability_outputs, binary_labels)


class PrintableBinaryConfusionMatrix(BinaryConfusionMatrix):
    def __init__(self) -> None:
        super().__init__()

    def compute(self):
        cm = super().compute()

        return f"TP {cm[0,0]}; FP {cm[0,1]}; TN {cm[1,0]}; FN {cm[1,1]}"
