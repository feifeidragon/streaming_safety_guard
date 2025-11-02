# models/latent_analyzer.py

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger


class LatentDynamicsAnalyzer:
    """
    隐藏状态动态分析器
    基于Kelp的思想，分析生成过程中隐藏状态的动态变化
    """

    def __init__(self, config: Dict):
        self.config = config
        self.layer_indices = config.get('layer_indices', [-1, -2, -3])
        self.aggregation = config.get('aggregation', 'weighted')
        self.risk_threshold = config.get('risk_threshold', 0.6)

        # 历史状态缓存
        self.history_states = []
        self.max_history = config.get('max_history', 20)

        # 安全基线（需要从安全样本中学习）
        self.safe_baseline_mean = None
        self.safe_baseline_std = None

        # 记录目标hidden_size（使用最后一层的）
        self.target_hidden_size = None

        logger.info(f"LatentDynamicsAnalyzer initialized with layers: {self.layer_indices}")

    def reset(self):
        """重置历史状态"""
        self.history_states = []
        self.target_hidden_size = None

    def analyze_hidden_states(
            self,
            hidden_states: List[torch.Tensor],
            attention_weights: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        分析隐藏状态的动态特征

        Args:
            hidden_states: 各层的隐藏状态
            attention_weights: 注意力权重（可选）

        Returns:
            metrics: 各种动态指标
        """
        metrics = {}

        try:
            # 首先确定目标hidden_size（使用最后一层的）
            if self.target_hidden_size is None and len(hidden_states) > 0:
                last_state = hidden_states[-1]
                while last_state.dim() > 2:
                    if last_state.size(1) == 1:
                        last_state = last_state.squeeze(1)
                    else:
                        last_state = last_state[:, -1, :]
                if last_state.dim() == 1:
                    last_state = last_state.unsqueeze(0)
                self.target_hidden_size = last_state.size(-1)

            # 提取指定层的隐藏状态
            selected_states = []
            for idx in self.layer_indices:
                if idx < len(hidden_states):
                    state = hidden_states[idx]

                    # 统一处理维度: 目标是 [batch_size, hidden_size]
                    while state.dim() > 2:
                        if state.size(1) == 1:
                            state = state.squeeze(1)
                        else:
                            state = state[:, -1, :]

                    # 确保是2维
                    if state.dim() == 1:
                        state = state.unsqueeze(0)

                    # 统一hidden_size到target_hidden_size
                    current_size = state.size(-1)
                    if current_size != self.target_hidden_size:
                        if current_size > self.target_hidden_size:
                            # 截断
                            state = state[..., :self.target_hidden_size]
                        else:
                            # 填充
                            padding = torch.zeros(
                                state.size(0),
                                self.target_hidden_size - current_size,
                                dtype=state.dtype,
                                device=state.device
                            )
                            state = torch.cat([state, padding], dim=-1)

                    selected_states.append(state)

            if not selected_states:
                return self._get_default_metrics()

            # 现在所有状态的shape都是 [batch_size, target_hidden_size]
            # 可以安全地stack
            stacked_states = torch.stack(selected_states, dim=0)  # [num_layers, batch_size, hidden_size]
            current_state = stacked_states.mean(dim=0)  # [batch_size, hidden_size]

            # 添加到历史
            self.history_states.append(current_state.detach().cpu())
            if len(self.history_states) > self.max_history:
                self.history_states.pop(0)

            # 计算动态指标
            if len(self.history_states) >= 2:
                metrics['velocity'] = self._calculate_velocity()

                if len(self.history_states) >= 3:
                    metrics['acceleration'] = self._calculate_acceleration()
                else:
                    metrics['acceleration'] = 0.0

                metrics['entropy'] = self._calculate_entropy(current_state)

                if self.safe_baseline_mean is not None:
                    metrics['deviation'] = self._calculate_deviation(current_state)
                else:
                    metrics['deviation'] = 0.0
            else:
                metrics = self._get_default_metrics()

            # 注意力异常
            if attention_weights is not None and len(attention_weights) > 0:
                metrics['attention_anomaly'] = self._calculate_attention_anomaly(attention_weights)
            else:
                metrics['attention_anomaly'] = 0.0

        except Exception as e:
            logger.error(f"Error in analyze_hidden_states: {e}")
            import traceback
            logger.error(traceback.format_exc())
            metrics = self._get_default_metrics()

        return metrics

    def _get_default_metrics(self) -> Dict[str, float]:
        """返回默认指标"""
        return {
            'velocity': 0.0,
            'acceleration': 0.0,
            'entropy': 0.0,
            'deviation': 0.0,
            'attention_anomaly': 0.0
        }

    def _calculate_velocity(self) -> float:
        """计算状态变化速度"""
        if len(self.history_states) < 2:
            return 0.0

        current = self.history_states[-1]
        previous = self.history_states[-2]

        velocity = torch.norm(current - previous, p=2).item()
        return float(velocity)

    def _calculate_acceleration(self) -> float:
        """计算状态变化加速度"""
        if len(self.history_states) < 3:
            return 0.0

        current = self.history_states[-1]
        previous = self.history_states[-2]
        pre_previous = self.history_states[-3]

        v1 = torch.norm(current - previous, p=2)
        v2 = torch.norm(previous - pre_previous, p=2)

        acceleration = torch.abs(v1 - v2).item()
        return float(acceleration)

    def _calculate_entropy(self, state: torch.Tensor) -> float:
        """计算状态熵"""
        probs = torch.softmax(state.flatten(), dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        return float(entropy)

    def _calculate_deviation(self, state: torch.Tensor) -> float:
        """计算与安全基线的偏离度"""
        if self.safe_baseline_mean is None:
            return 0.0

        mean = self.safe_baseline_mean.to(state.device)
        std = self.safe_baseline_std.to(state.device)

        deviation = torch.norm((state - mean) / (std + 1e-8), p=2).item()
        return float(deviation)

    def _calculate_attention_anomaly(self, attention_weights: List[torch.Tensor]) -> float:
        """计算注意力异常度"""
        try:
            last_attn = attention_weights[-1]

            while last_attn.dim() > 2:
                if last_attn.size(0) == 1:
                    last_attn = last_attn.squeeze(0)
                elif last_attn.size(1) == 1:
                    last_attn = last_attn.squeeze(1)
                else:
                    last_attn = last_attn.mean(dim=0)

            max_attn = torch.max(last_attn.flatten()).item()
            anomaly = min(max_attn * 10, 1.0)
            return float(anomaly)
        except Exception as e:
            logger.error(f"Error calculating attention anomaly: {e}")
            return 0.0

    def update_safe_baseline(self, safe_states: List[torch.Tensor]):
        """从安全样本中学习基线"""
        if not safe_states:
            return

        states_tensor = torch.stack(safe_states)
        self.safe_baseline_mean = states_tensor.mean(dim=0)
        self.safe_baseline_std = states_tensor.std(dim=0)

        logger.info("Safe baseline updated")

    def get_risk_score(self, metrics: Dict[str, float]) -> float:
        """根据动态指标计算风险分数"""
        weights = {
            'velocity': 0.25,
            'acceleration': 0.20,
            'entropy': 0.20,
            'deviation': 0.25,
            'attention_anomaly': 0.10
        }

        risk_score = 0.0
        for key, weight in weights.items():
            if key in metrics:
                normalized_value = min(metrics[key] / 10.0, 1.0)
                risk_score += weight * normalized_value

        return risk_score