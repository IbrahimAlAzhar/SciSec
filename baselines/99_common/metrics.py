import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: List[float] = None
) -> Dict[str, float]:
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate FPR and FNR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Calculate AUC if scores are provided and have variance
    # Note: AUC requires continuous scores, not just 0/1 predictions
    auc = None
    if y_scores is not None:
        try:
            y_scores = np.array(y_scores)
            # Check if scores have variance (not all same value) and are not just 0/1
            unique_scores = np.unique(y_scores)
            if len(unique_scores) > 1 and not (len(unique_scores) == 2 and set(unique_scores).issubset({0, 1})):
                # Only calculate AUC if scores are continuous (not binary)
                auc = roc_auc_score(y_true, y_scores)
            else:
                # If scores are binary (0/1) or all same, AUC = 0.5 (cannot distinguish)
                auc = 0.5
        except ValueError as e:
            # AUC calculation may fail if only one class is present
            auc = 0.5
    else:
        auc = 0.5  # No scores provided, cannot calculate AUC
    
    return {
        "accuracy": float(accuracy),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc) if auc is not None else 0.5,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": len(y_true)
    }


def calculate_metrics_from_defense_results(
    results: List[Dict[str, Any]]
) -> Dict[str, float]:
    
    y_true = [r["ground_truth"] for r in results]
    y_pred = [r["prediction"] for r in results]
    y_scores = [r.get("score", r["prediction"]) for r in results]
    
    return calculate_metrics(y_true, y_pred, y_scores)


def print_metrics(metrics: Dict[str, float]):
    
    print("\n" + "=" * 80)
    print("DEFENSE EVALUATION METRICS")
    print("=" * 80)

    print(f"\nClassification Metrics:")
    print('Acc\tPrec\tRecall\tF1\tFPR\tFNR\tAUC')
    auc_str = f"{metrics['auc']:.4f}" if metrics.get('auc') is not None else "N/A"
    print(f"{metrics['accuracy']:.4f}\t{metrics['precision']:.4f}\t{metrics['recall']:.4f}\t{metrics.get('f1', 0.0):.4f}\t{metrics['fpr']:.4f}\t{metrics['fnr']:.4f}\t{auc_str}")
    if metrics.get('auc') == 0.5:
        print("\nNote: AUC = 0.5 indicates scores are binary (0/1) or constant. AUC requires continuous confidence scores.")
    print("=" * 80 + "\n")
