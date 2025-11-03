# models/router.py - 确认 SpecializedRouter 的初始化参数

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from loguru import logger


class SpecializedRouter(nn.Module):
    """
    专家路由系统 - 根据内容类型路由到不同的专家检测器
    """

    def __init__(
            self,
            input_dim: int,  # 确保使用 input_dim
            num_experts: int = 3,
            expert_types: List[str] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.expert_types = expert_types or ['toxicity', 'violence', 'sexual']

        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_experts)
        )

        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim // 2, 2)  # 二分类
            )
            for _ in range(num_experts)
        ])

        logger.info(f"SpecializedRouter initialized with {num_experts} experts: {self.expert_types}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播

        Args:
            x: [batch_size, input_dim]

        Returns:
            route_logits: [batch_size, num_experts] 路由决策
            expert_outputs: List of [batch_size, 2] 每个专家的输出
        """
        # 路由决策
        route_logits = self.router(x)  # [batch_size, num_experts]

        # 获取每个专家的输出
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)  # [batch_size, 2]
            expert_outputs.append(output)

        return route_logits, expert_outputs