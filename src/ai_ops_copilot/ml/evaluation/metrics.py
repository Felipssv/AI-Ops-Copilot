import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from ai_ops_copilot.domain.ml import ClassificationMetrics, ConfusionMatrixSummary

def compute_classification_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> ClassificationMetrics:

    tn,fp,fn,tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    matrix = ConfusionMatrixSummary(true_negatives=int(tn),false_positives=int(fp),false_negatives=int(fn),true_positives=int(tp))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division = 0.0)
    recall = recall_score(y_true, y_pred, zero_division = 0.0)
    score = f1_score(y_true, y_pred, zero_division = 0.0)

    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5

    pr_auc = average_precision_score(y_true, y_prob)

    return ClassificationMetrics(accuracy=round(float(accuracy), 4), 
                                precision=round(float(precision),4), 
                                recall=round(float(recall),4), 
                                f1_score=round(float(score),4), 
                                roc_auc=round(float(roc_auc),4), 
                                pr_auc=round(float(pr_auc),4),
                                confusion_matrix=matrix
                                )
