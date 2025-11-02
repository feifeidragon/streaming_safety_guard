#!/usr/bin/env python3
import torch
import yaml
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

from core.streaming_guard import StreamingGuard
from utils.metrics import SafetyMetrics


def plot_confusion_matrix(y_true, y_pred, output_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"混淆矩阵已保存到: {output_path}")


def plot_roc_curve(y_true, y_scores, output_path):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC曲线已保存到: {output_path}")


def plot_score_distribution(y_true, y_scores, output_path):
    """绘制得分分布"""
    safe_scores = [score for label, score in zip(y_true, y_scores) if label == 0]
    unsafe_scores = [score for label, score in zip(y_true, y_scores) if label == 1]

    plt.figure(figsize=(10, 6))
    plt.hist(safe_scores, bins=50, alpha=0.5, label='Safe', color='green')
    plt.hist(unsafe_scores, bins=50, alpha=0.5, label='Unsafe', color='red')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.title('Risk Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"得分分布图已保存到: {output_path}")


def analyze_false_cases(results, output_path):
    """分析误判案例"""
    false_positives = []
    false_negatives = []

    for item in results:
        if 'error' in item:
            continue

        true_label = item['true_label']
        pred_label = item['pred_label']

        if true_label == 0 and pred_label == 1:
            false_positives.append(item)
        elif true_label == 1 and pred_label == 0:
            false_negatives.append(item)

    analysis = {
        'false_positives': false_positives[:10],  # 保存前10个
        'false_negatives': false_negatives[:10],
        'fp_count': len(false_positives),
        'fn_count': len(false_negatives)
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    logger.info(f"误判分析已保存到: {output_path}")
    logger.info(f"False Positives: {len(false_positives)}")
    logger.info(f"False Negatives: {len(false_negatives)}")


def evaluate_latency_performance(guard, test_prompts, output_dir):
    """评估延迟性能"""
    latencies = []
    token_counts = []

    logger.info("评估延迟性能...")
    for prompt in tqdm(test_prompts[:50]):  # 测试50个样本
        result = guard.generate_with_guard(
            prompt=prompt,
            max_new_tokens=256
        )

        if result['generation_time'] > 0:
            latencies.append(result['generation_time'] * 1000)  # 转换为毫秒
            token_counts.append(result['num_tokens'])

    # 绘制延迟分布
    plt.figure(figsize=(10, 6))
    plt.scatter(token_counts, latencies, alpha=0.6)
    plt.xlabel('Number of Tokens')
    plt.ylabel('Latency (ms)')
    plt.title('Generation Latency vs Token Count')
    plt.grid(True, alpha=0.3)

    # 添加趋势线
    z = np.polyfit(token_counts, latencies, 1)
    p = np.poly1d(z)
    plt.plot(token_counts, p(token_counts), "r--", alpha=0.8, label='Trend')
    plt.legend()

    output_path = Path(output_dir) / 'latency_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"延迟分析图已保存到: {output_path}")

    # 计算延迟统计
    latency_stats = {
        'mean_latency_ms': np.mean(latencies),
        'median_latency_ms': np.median(latencies),
    'p95_latency_ms': np.percentile(latencies, 95),
    'p99_latency_ms': np.percentile(latencies, 99),
    'max_latency_ms': np.max(latencies),
    'min_latency_ms': np.min(latencies),
    'avg_tokens_per_second': np.mean([t / (l / 1000) for t, l in zip(token_counts, latencies)])
    }

    stats_path = Path(output_dir) / 'latency_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(latency_stats, f, indent=2)

    logger.info(f"延迟统计已保存到: {stats_path}")
    return latency_stats


def comprehensive_evaluation(config_path='config/config.yaml', output_dir='./evaluation_results'):
    """综合评估"""
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 加载数据集
    dataset_path = config['dataset']['train_path']
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    logger.info(f"加载了 {len(dataset)} 条数据")

    # 初始化守卫系统
    logger.info("初始化安全守卫系统...")
    guard = StreamingGuard(config)

    # 1. 基本性能评估
    logger.info("=" * 60)
    logger.info("1. 基本性能评估")
    logger.info("=" * 60)

    results = []
    y_true = []
    y_pred = []
    y_scores = []

    for idx, item in enumerate(tqdm(dataset[:100], desc="评估中")):  # 评估前100个
        prompt = item['prompt']
        true_label = item.get('label', 0)

        try:
            result = guard.generate_with_guard(
                prompt=prompt,
                max_new_tokens=256
            )

            pred_label = 1 if result['is_blocked'] else 0
            risk_score = 0.0
            if result['block_reason']:
                risk_score = result['block_reason']['final_score']

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_scores.append(risk_score)

            results.append({
                'index': idx,
                'prompt': prompt,
                'true_label': true_label,
                'pred_label': pred_label,
                'risk_score': risk_score,
                'is_blocked': result['is_blocked']
            })
        except Exception as e:
            logger.error(f"处理第 {idx} 条数据时出错: {e}")

    # 计算指标
    metrics = SafetyMetrics.calculate_metrics(y_true, y_pred, y_scores)

    # 保存指标
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # 打印指标
    print("\n评估指标:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 2. 可视化分析
    logger.info("\n" + "=" * 60)
    logger.info("2. 生成可视化分析")
    logger.info("=" * 60)

    plot_confusion_matrix(y_true, y_pred, output_dir / 'confusion_matrix.png')
    plot_roc_curve(y_true, y_scores, output_dir / 'roc_curve.png')
    plot_score_distribution(y_true, y_scores, output_dir / 'score_distribution.png')

    # 3. 误判分析
    logger.info("\n" + "=" * 60)
    logger.info("3. 误判案例分析")
    logger.info("=" * 60)
    analyze_false_cases(results, output_dir / 'false_cases_analysis.json')

    # 4. 延迟性能评估
    logger.info("\n" + "=" * 60)
    logger.info("4. 延迟性能评估")
    logger.info("=" * 60)
    test_prompts = [item['prompt'] for item in dataset[:50]]
    latency_stats = evaluate_latency_performance(guard, test_prompts, output_dir)

    print("\n延迟统计:")
    for key, value in latency_stats.items():
        print(f"  {key}: {value:.2f}")

    # 5. 系统统计
    logger.info("\n" + "=" * 60)
    logger.info("5. 系统统计")
    logger.info("=" * 60)

    stats = guard.get_statistics()
    stats_path = output_dir / 'system_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print("\n系统统计:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # 生成综合报告
    report = {
        'metrics': metrics,
        'latency_stats': latency_stats,
        'system_stats': stats,
        'dataset_size': len(dataset),
        'evaluated_samples': len(results)
    }

    report_path = output_dir / 'comprehensive_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n综合报告已保存到: {report_path}")
    logger.info(f"所有评估结果已保存到: {output_dir}")


def main():
    """主函数"""
    comprehensive_evaluation()


if __name__ == "__main__":
    main()