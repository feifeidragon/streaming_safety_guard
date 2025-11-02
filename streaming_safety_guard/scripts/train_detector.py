#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np

from models.safety_detector import SafetyDetector
from utils.metrics import SafetyMetrics


class SafetyDataset(Dataset):
    """安全检测数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        label = item.get('label', 0)

        # 编码
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_safety_detector(config: dict):
    """训练安全检测器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载tokenizer和base model
    model_path = config['model']['local_model_path']
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)
    base_model.eval()

    # 创建数据集
    train_dataset = SafetyDataset(
        config['dataset']['train_path'],
        tokenizer,
        max_length=config['model']['max_length']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )

    # 初始化安全检测器
    safety_detector = SafetyDetector(config['safety_detector']).to(device)

    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        safety_detector.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    num_training_steps = len(train_loader) * config['training']['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=num_training_steps
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    logger.info("开始训练...")
    global_step = 0
    best_f1 = 0.0

    for epoch in range(config['training']['epochs']):
        safety_detector.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 获取base model的隐藏状态
            with torch.no_grad():
                base_outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden_states = base_outputs.hidden_states[-1]

            # 前向传播
            outputs = safety_detector(hidden_states, attention_mask)

            # 计算损失（多任务）
            loss = 0
            for risk_type, logits in outputs['risk_logits'].items():
                loss += criterion(logits, labels)

            # 添加总体损失
            overall_logits = torch.cat([
                1 - outputs['overall_score'],
                outputs['overall_score']
            ], dim=1)
            loss += criterion(overall_logits, labels)

            loss = loss / (len(outputs['risk_logits']) + 1)

            # 反向传播
            loss.backward()

            if (batch_idx + 1) % config['training']['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(safety_detector.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

            # 保存检查点
            if global_step % config['training']['save_steps'] == 0:
                save_checkpoint(
                    safety_detector,
                    optimizer,
                    scheduler,
                    global_step,
                    config['training']['output_dir']
                )

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

        # 评估
        if (epoch + 1) % 2 == 0:
            metrics = evaluate_detector(
                safety_detector,
                base_model,
                train_loader,
                device
            )
            logger.info(f"Epoch {epoch + 1} 评估指标: F1={metrics['f1']:.4f}")

            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                save_checkpoint(
                    safety_detector,
                    optimizer,
                    scheduler,
                    global_step,
                    config['training']['output_dir'],
                    is_best=True
                )

    logger.info("训练完成！")


def evaluate_detector(detector, base_model, data_loader, device):
    """评估检测器"""
    detector.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 获取隐藏状态
            base_outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = base_outputs.hidden_states[-1]

            # 预测
            outputs = detector(hidden_states, attention_mask)
            scores = outputs['overall_score'].cpu().numpy()
            preds = (scores > 0.5).astype(int)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_scores.extend(scores)

        metrics = SafetyMetrics.calculate_metrics(y_true, y_pred, y_scores)
        return metrics

    def save_checkpoint(model, optimizer, scheduler, global_step, output_dir, is_best=False):
        """保存检查点"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step
        }

        if is_best:
            checkpoint_path = output_dir / 'best_model.pt'
            logger.info(f"保存最佳模型到: {checkpoint_path}")
        else:
            checkpoint_path = output_dir / f'checkpoint_step_{global_step}.pt'
            logger.info(f"保存检查点到: {checkpoint_path}")

        torch.save(checkpoint, checkpoint_path)

    def main():
        """主函数"""
        # 加载配置
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 训练
        train_safety_detector(config)

    if __name__ == "__main__":
        main()