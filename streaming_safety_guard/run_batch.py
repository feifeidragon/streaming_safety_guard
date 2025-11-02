#!/usr/bin/env python3
import json
import yaml
import torch
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from typing import List, Dict
import pandas as pd
from datetime import datetime

from core.streaming_guard import StreamingGuard
from utils.metrics import SafetyMetrics


def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(file_path: str) -> List[Dict]:
    """加载数据集"""
    logger.info(f"加载数据集: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"加载了 {len(data)} 条数据")
    return data


def batch_evaluate(guard: StreamingGuard, dataset: List[Dict], output_dir: str = "./results"):
    """批量评估"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    y_true = []
    y_pred = []
    y_scores = []
    latencies = []

    logger.info("开始批量评估...")

    for idx, item in enumerate(tqdm(dataset, desc="评估进度")):
        prompt = item['prompt']
        true_label = item.get('label', 0)  # 0: 安全, 1: 不安全

        try:
            # 生成并检测
            result = guard.generate_with_guard(
                prompt=prompt,
                max_new_tokens=256
            )

            # 预测标签（如果被拦截则为1，否则为0）
            pred_label = 1 if result['is_blocked'] else 0

            # 获取风险得分
            risk_score = 0.0
            if result['block_reason']:
                risk_score = result['block_reason']['final_score']

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_scores.append(risk_score)

            # 记录结果
            result_item = {
                'index': idx,
                'prompt': prompt,
                'generated_text': result['generated_text'],
                'true_label': true_label,
                'pred_label': pred_label,
                'risk_score': risk_score,
                'is_blocked': result['is_blocked'],
                'num_tokens': result['num_tokens'],
                'generation_time': result['generation_time']
            }

            if result['block_reason']:
                result_item['block_reason'] = {
                    k: v for k, v in result['block_reason'].items()
                    if k not in ['triggered_text', 'full_text']
                }

            results.append(result_item)

        except Exception as e:
            logger.error(f"处理第 {idx} 条数据时出错: {e}")
            results.append({
                'index': idx,
                'prompt': prompt,
                'error': str(e)
            })

    # 计算指标
    logger.info("计算评估指标...")
    metrics = SafetyMetrics.calculate_metrics(y_true, y_pred, y_scores)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存详细结果
    results_file = Path(output_dir) / f"results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果已保存到: {results_file}")

    # 保存指标
    metrics_file = Path(output_dir) / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"评估指标已保存到: {metrics_file}")

    # 保存CSV格式
    df = pd.DataFrame(results)
    csv_file = Path(output_dir) / f"results_{timestamp}.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    logger.info(f"CSV结果已保存到: {csv_file}")

    # 打印指标
    print("\n" + "=" * 60)
    print("评估指标")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:.4f}")
        else:
            print(f"{key:.<40} {value}")
    print("=" * 60)

    # 系统统计
    stats = guard.get_statistics()
    print("\n" + "=" * 60)
    print("系统统计")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:.4f}")
        else:
            print(f"{key:.<40} {value}")
    print("=" * 60)

    return results, metrics


def main():
    """主函数"""
    logger.info("启动批量评估模式")

    # 加载配置
    config = load_config()

    # 加载数据集
    dataset_path = config['dataset']['train_path']
    dataset = load_dataset(dataset_path)

    # 限制样本数量（如果配置了）
    max_samples = config['dataset'].get('max_samples')
    if max_samples and max_samples < len(dataset):
        dataset = dataset[:max_samples]
        logger.info(f"限制评估样本数量为: {max_samples}")

    # 初始化守卫系统
    logger.info("初始化安全守卫系统...")
    guard = StreamingGuard(config)

    # 批量评估
    results, metrics = batch_evaluate(guard, dataset)

    logger.info("批量评估完成！")


if __name__ == "__main__": main()