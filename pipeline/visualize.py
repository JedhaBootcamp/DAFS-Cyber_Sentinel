from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)

CAT_COLS = ["protocol_type", "service", "flag"]

def _ensure(p: str):
    os.makedirs(p, exist_ok=True)

def make_plots(model, X_test, y_test, df_full, outdir: str):
    """
    Generates BOTH:
      - EDA PNGs in  outputs/eda/
      - Model PNGs in outputs/model/
    Uses tuned threshold if present.
    """
    eda_dir = os.path.join(outdir, "eda")
    mdl_dir = os.path.join(outdir, "model")
    _ensure(eda_dir); _ensure(mdl_dir)

    # --------------------
    # EDA: class balance
    # --------------------
    if "class" in df_full.columns:
        plt.figure()
        sns.countplot(x="class", data=df_full.replace({1: "anomaly", 0: "normal"}))
        plt.title("Class Balance (dataset)")
        plt.tight_layout()
        plt.savefig(os.path.join(eda_dir, "class_balance.png"), dpi=150)
        plt.close()

    # EDA: categorical volumes + anomaly rates (top 20)
    if "class" in df_full.columns:
        df_plot = df_full.copy()
        if df_plot["class"].dtype != int:
            df_plot["class"] = (
                df_plot["class"].astype(str).str.lower().map({"normal": 0, "anomaly": 1, "attack": 1}).fillna(0).astype(int)
            )
        for col in [c for c in CAT_COLS if c in df_plot.columns]:
            top = df_plot[col].value_counts().head(20).index
            sub = df_plot[df_plot[col].isin(top)]
            # volume
            plt.figure(figsize=(8, 4))
            sns.countplot(y=col, data=sub, order=top)
            plt.title(f"{col}: Top 20 by Volume")
            plt.tight_layout()
            plt.savefig(os.path.join(eda_dir, f"{col}_top20_volume.png"), dpi=150)
            plt.close()
            # anomaly rate
            rate = sub.groupby(col)["class"].mean().loc[top]
            plt.figure(figsize=(8, 4))
            rate.plot(kind="barh")
            plt.xlabel("Anomaly rate")
            plt.title(f"{col}: Anomaly Rate (Top 20)")
            plt.tight_layout()
            plt.savefig(os.path.join(eda_dir, f"{col}_top20_anomaly_rate.png"), dpi=150)
            plt.close()

    # EDA: numeric distributions (log-aware) + correlation
    num_cols = df_full.select_dtypes(include=[np.number]).columns.drop("class", errors="ignore")
    for c in num_cols[:20]:
        series = df_full[c].dropna()
        plt.figure(figsize=(6, 3))
        # log if super-skewed
        if (series > 0).sum() > 0 and (series.max() / (series[series > 0].min() + 1e-9) > 100):
            sns.histplot(np.log1p(series), bins=50); plt.title(f"{c} (log1p)")
        else:
            sns.histplot(series, bins=50); plt.title(c)
        plt.tight_layout()
        plt.savefig(os.path.join(eda_dir, f"num_{c}.png"), dpi=150)
        plt.close()

    if len(num_cols) > 1:
        plt.figure(figsize=(8, 6))
        corr = df_full[num_cols].corr(numeric_only=True)
        sns.heatmap(corr, cmap="vlag", center=0)
        plt.title("Correlation Heatmap (numeric)")
        plt.tight_layout()
        plt.savefig(os.path.join(eda_dir, "corr_heatmap_numeric.png"), dpi=150)
        plt.close()

    # --------------------
    # Model plots (TEST)
    # --------------------
    thr = getattr(model, "threshold_", 0.5)
    clf = model.named_steps["model"]
    X_test = model.named_steps["preproc"].transform(X_test)
    if hasattr(clf, "best_iteration") and clf.best_iteration is not None:
        proba = clf.predict_proba(X_test, iteration_range=(0, clf.best_iteration + 1))[:, 1]
    else:
        booster = clf.get_booster()
        proba = clf.predict_proba(X_test)[:, 1]

    y_pred = (proba >= thr).astype(int)

    # Confusion matrix (test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["normal (0)", "anomaly (1)"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — Test (thr={thr:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(mdl_dir, "cm_test.png"), dpi=150)
    plt.close()

    # ROC (test)
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Test")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(mdl_dir, "roc_curve_test.png"), dpi=150)
    plt.close()

    # PR (test)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve — Test")
    plt.tight_layout()
    plt.savefig(os.path.join(mdl_dir, "pr_curve_test.png"), dpi=150)
    plt.close()

    # Error analysis by protocol/service/flag (align by X_test indices)
    if hasattr(X_test, "index"):
        idx = X_test.index
        base = df_full.loc[idx, [c for c in CAT_COLS if c in df_full.columns]].copy()
        base["y_true"] = y_test.values if hasattr(y_test, "values") else y_test
        base["y_pred"] = y_pred
        base["error"] = (base["y_true"] != base["y_pred"]).astype(int)
        for col in [c for c in CAT_COLS if c in base.columns]:
            plt.figure(figsize=(7, 4))
            top = base[col].value_counts().head(12).index
            sns.barplot(data=base[base[col].isin(top)], x=col, y="error", order=top)
            plt.title(f"Error Rate by {col} (top 12) — Test")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(mdl_dir, f"errors_by_{col}.png"), dpi=150)
            plt.close()

    # XGBoost split importances (indices)
    try:
        booster = model.named_steps["model"].get_booster()
        scores = booster.get_fscore()
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:25]
        if items:
            names = [k for k, _ in items]
            vals = [v for _, v in items]
            plt.figure(figsize=(6, max(3, 0.3 * len(items))))
            sns.barplot(x=vals, y=names, orient="h")
            plt.title("Top XGBoost Split Importances (feature indices)")
            plt.xlabel("Split Count")
            plt.tight_layout()
            plt.savefig(os.path.join(mdl_dir, "xgb_feature_importance.png"), dpi=150)
            plt.close()
    except Exception:
        pass
