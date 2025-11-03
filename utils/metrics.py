import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


class SafetyMetrics:
    """安全检测指标计算"""

    @staticmethod
    def calculate_metrics(
            y_true: List[int],
            y_pred: List[int],
            y_scores: List[float] = None
    ) -> Dict[str, float]:
        """
        计算各种指标
        Args:
            y_true: 真实标签 (0: 安全, 1: 不安全)
            y_pred: 预测标签
            y_scores: 预测得分
        """
        metrics = {}

        # 基本指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)

        # False Positive Rate 和 False Negative Rate
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # AUC-ROC
        if y_scores is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
            except:
                metrics['auc_roc'] = 0.0

        return metrics

    @staticmethod
    def calculate_latency_metrics(latencies: List[float]) -> Dict[str, float]:
        """计算延迟指标"""
        if not latencies:
            return {}

        latencies_array = np.array(latencies)
        return {
            'mean_latency_ms': float(np.mean(latencies_array)),
            'median_latency_ms': float(np.median(latencies_array)),
            'p95_latency_ms': float(np.percentile(latencies_array, 95)),
            'p99_latency_ms': float(np.percentile(latencies_array, 99)),
            'max_latency_ms': float(np.max(latencies_array)),
            'min_latency_ms': float(np.min(latencies_array))
        }