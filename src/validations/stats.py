import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Tuple, Callable

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn: Callable, 
                 n_bootstraps: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Бутстрап доверительный интервал для метрики."""
    scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return np.mean(scores), np.percentile(scores, alpha * 100), np.percentile(scores, (1 - alpha) * 100)

def compare_models(y_true: np.ndarray, y_pred_lr: np.ndarray, y_pred_mlp: np.ndarray) -> Dict:
    """Сравнение моделей: метрики + CI + McNemar test."""
    acc_lr, acc_lr_l, acc_lr_u = bootstrap_ci(y_true, y_pred_lr, accuracy_score)
    acc_mlp, acc_mlp_l, acc_mlp_u = bootstrap_ci(y_true, y_pred_mlp, accuracy_score)
    f1_lr, _, _ = bootstrap_ci(y_true, y_pred_lr, f1_score)
    f1_mlp, _, _ = bootstrap_ci(y_true, y_pred_mlp, f1_score)

    n01 = np.sum((y_pred_lr != y_true) & (y_pred_mlp == y_true))
    n10 = np.sum((y_pred_lr == y_true) & (y_pred_mlp != y_true))
    if n01 + n10 == 0:
        mc_pval = 1.0
    else:
        mc_pval = 1 - stats.chi2.cdf((abs(n01 - n10) - 1)**2 / (n01 + n10), 1)

    return {
        "lr_accuracy": {"mean": acc_lr, "ci_95": [acc_lr_l, acc_lr_u]},
        "mlp_accuracy": {"mean": acc_mlp, "ci_95": [acc_mlp_l, acc_mlp_u]},
        "lr_f1": f1_lr, "mlp_f1": f1_mlp,
        "mcnemar_p_value": mc_pval,
        "statistically_significant_diff": mc_pval < 0.05
    }