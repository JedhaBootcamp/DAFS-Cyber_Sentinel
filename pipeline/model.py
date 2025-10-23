from __future__ import annotations
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier

def _scale_pos_weight(y):
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    return neg / pos

def _tune_threshold(y_true, y_scores, precision_floor: float = 0.90) -> Tuple[float, dict]:
    p, r, t = precision_recall_curve(y_true, y_scores)
    thresholds = np.concatenate(([0.0], t))
    mask = p >= precision_floor
    if np.any(mask):
        i = np.where(mask)[0][np.argmax(r[mask])]
        return float(thresholds[i]), {"mode": f"prec≥{precision_floor}", "precision": float(p[i]), "recall": float(r[i])}
    f1 = 2 * p * r / (p + r + 1e-12)
    i = int(np.nanargmax(f1))
    return float(thresholds[i]), {"mode": "best_f1", "precision": float(p[i]), "recall": float(r[i])}

def _metrics_from_proba(y_true, y_proba, thr: float) -> Dict:
    y_pred = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "positives": int(np.sum(y_true == 1)),
        "negatives": int(np.sum(y_true == 0)),
    }

def _predict_proba_best(pipe: Pipeline, X):
    """Version-robust proba using best iteration if available."""
    clf = pipe.named_steps["model"]
    X = pipe.named_steps["preproc"].transform(X)
    if hasattr(clf, "best_iteration") and clf.best_iteration is not None:
        return clf.predict_proba(X, iteration_range=(0, clf.best_iteration + 1))[:, 1]
    booster = clf.get_booster()
    if hasattr(booster, "best_ntree_limit") and booster.best_ntree_limit:
        return clf.predict_proba(X, ntree_limit=booster.best_ntree_limit)[:, 1]
    return clf.predict_proba(X)[:, 1]

def train_model(X_train, y_train, preproc, seed: int = 42):
    """
    Uses XGBoost callback-based early stopping (xgboost>=2).
    Stores:
      - threshold_ (tuned on internal valid)
      - train_metrics_ / valid_metrics_ (to compare vs test in evaluate())
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed
    )

    clf = XGBClassifier(
        n_estimators=2000,            # capped by early stopping
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        enable_categorical=True,
        scale_pos_weight=_scale_pos_weight(y_tr),
        verbosity=0,
    )

    pipe = Pipeline([("preproc", preproc), ("model", clf)])

    # Fit the full pipeline — preprocessing is applied inside
    pipe.fit(X_tr, y_tr)

    # Tune a probability threshold on the VALID split (precision floor)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    thr, report = _tune_threshold(y_val, val_proba, precision_floor=0.90)
    
    # Tune threshold on validation using the best iteration
    val_proba = _predict_proba_best(pipe, X_val)
    thr, report = _tune_threshold(y_val, val_proba, precision_floor=0.90)

    # Store attributes for evaluate()
    setattr(pipe, "threshold_", float(thr))
    setattr(pipe, "fit_report_", report)

    tr_proba = _predict_proba_best(pipe, X_tr)
    setattr(pipe, "train_metrics_", _metrics_from_proba(y_tr, tr_proba, thr))
    setattr(pipe, "valid_metrics_", _metrics_from_proba(y_val, val_proba, thr))

    return pipe
