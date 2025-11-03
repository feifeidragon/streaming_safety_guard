# models/safety_detector.py - 完整修复版本

import torch
import torch.nn as nn
from typing import Tuple, Optional
from loguru import logger


class SafetyDetector(nn.Module):
    """
    安全检测器 - 基于隐藏状态的安全性判断
    """

    def __init__(
            self,
            input_dim: int,  # 改为 input_dim 而不是 hidden_size
            num_classes: int = 2,
            dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM用于序列建模 - 修复dropout警告
        self.lstm = nn.LSTM(
            input_dim // 4,
            input_dim // 4,
            num_layers=1,
            batch_first=True,
            dropout=0.0  # 单层LSTM不需要dropout
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(input_dim // 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        logger.info(f"SafetyDetector initialized with input_dim={input_dim}")

    def forward(
            self,
            hidden_states: torch.Tensor,
            return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: [batch_size, seq_len, input_dim] 或 [seq_len, input_dim]
            return_features: 是否返回特征

        Returns:
            logits: [batch_size, num_classes]
            features: [batch_size, input_dim // 4]
        """
        # 统一处理维度
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)  # [1, seq_len, input_dim]

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 检查input_dim是否匹配，如果不匹配则调整
        if hidden_dim != self.input_dim:
            if hidden_dim > self.input_dim:
                hidden_states = hidden_states[..., :self.input_dim]
            else:
                padding = torch.zeros(
                    batch_size, seq_len, self.input_dim - hidden_dim,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                hidden_states = torch.cat([hidden_states, padding], dim=-1)

        # 特征提取
        features = self.feature_extractor(hidden_states)  # [batch_size, seq_len, input_dim // 4]

        # LSTM处理
        lstm_out, _ = self.lstm(features)  # [batch_size, seq_len, input_dim // 4]

        # 使用最后一个时间步
        last_hidden = lstm_out[:, -1, :]  # [batch_size, input_dim // 4]

        # 分类
        logits = self.classifier(last_hidden)  # [batch_size, num_classes]

        if return_features:
            return logits, last_hidden
        return logits, last_hidden