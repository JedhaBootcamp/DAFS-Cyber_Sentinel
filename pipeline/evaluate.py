from __future__ import annotations
from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def _predict_proba_best(model, X):
    clf = model.named_steps["model"]
    X = model.named_steps["preproc"].transform(X)
    if hasattr(clf, "best_iteration") and clf.best_iteration is not None:
        return clf.predict_proba(X, iteration_range=(0, clf.best_iteration + 1))[:, 1]
    booster = clf.get_booster()
    if hasattr(booster, "best_ntree_limit") and booster.best_ntree_limit:
        return clf.predict_proba(X, ntree_limit=booster.best_ntree_limit)[:, 1]
    return clf.predict_proba(X)[:, 1]

def evaluate(model, X_test, y_test) -> Dict:
    thr = getattr(model, "threshold_", 0.5)

    proba_test = _predict_proba_best(model, X_test)
    y_pred_test = (proba_test >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test, labels=[0, 1]).ravel()

    test_metrics = {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_test, y_pred_test)),
        "precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba_test)) if len(np.unique(y_test)) == 2 else None,
        "pr_auc": float(average_precision_score(y_test, proba_test)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "positives": int(np.sum(y_test == 1)),
        "negatives": int(np.sum(y_test == 0)),
    }

    return {
        "train": getattr(model, "train_metrics_", None),
        "valid": getattr(model, "valid_metrics_", None),
        "test":  test_metrics,
        "threshold": float(thr),
        "fit_report": getattr(model, "fit_report_", None),
    }
