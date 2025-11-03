# core/buffer_manager.py

import torch
from typing import List, Dict, Optional
from collections import deque
import time
from loguru import logger


class TokenBuffer:
    """
    Token缓冲区管理器
    用于管理生成过程中的token和隐藏状态
    """

    def __init__(self, buffer_size: int = 20, window_size: int = 10):
        self.buffer_size = buffer_size
        self.window_size = window_size

        # 缓冲区
        self.tokens = deque(maxlen=buffer_size)
        self.token_texts = deque(maxlen=buffer_size)
        self.hidden_states = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)

        # 统计
        self.total_tokens = 0
        self.last_check_position = 0

        # 记录hidden_size
        self.hidden_size = None

    def add_token(
            self,
            token_id: int,
            token_text: str,
            hidden_states: torch.Tensor,
            timestamp: float
    ):
        """添加新token"""
        # 确保hidden_states是2维 [batch_size, hidden_size]
        if hidden_states.dim() > 2:
            while hidden_states.dim() > 2:
                if hidden_states.size(0) == 1:
                    hidden_states = hidden_states.squeeze(0)
                elif hidden_states.size(1) == 1:
                    hidden_states = hidden_states.squeeze(1)
                else:
                    hidden_states = hidden_states[:, -1, :]

        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        # 记录第一次的hidden_size
        if self.hidden_size is None:
            self.hidden_size = hidden_states.size(-1)

        # 确保hidden_size一致
        if hidden_states.size(-1) != self.hidden_size:
            current_size = hidden_states.size(-1)
            if current_size > self.hidden_size:
                hidden_states = hidden_states[..., :self.hidden_size]
            else:
                padding = torch.zeros(
                    hidden_states.size(0),
                    self.hidden_size - current_size,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                hidden_states = torch.cat([hidden_states, padding], dim=-1)

        self.tokens.append(token_id)
        self.token_texts.append(token_text)
        self.hidden_states.append(hidden_states.detach().cpu())
        self.timestamps.append(timestamp)
        self.total_tokens += 1

    def should_check(self, check_interval: int) -> bool:
        """判断是否应该进行检查"""
        return (self.total_tokens - self.last_check_position) >= check_interval

    def mark_checked(self):
        """标记已检查"""
        self.last_check_position = self.total_tokens

    def get_current_text(self) -> str:
        """获取当前窗口的文本"""
        return ''.join(list(self.token_texts)[-self.window_size:])

    def get_recent_hidden_states(self, n: int = 10) -> Optional[torch.Tensor]:
        """获取最近n个隐藏状态"""
        if len(self.hidden_states) == 0:
            return None

        recent_states = list(self.hidden_states)[-n:]
        if not recent_states:
            return None

        # 确定目标hidden_size（使用第一个状态的）
        target_hidden_size = self.hidden_size
        if target_hidden_size is None:
            target_hidden_size = recent_states[0].size(-1)

        # 确保所有状态维度一致
        processed_states = []

        for state in recent_states:
            # 确保是2维
            if state.dim() == 1:
                state = state.unsqueeze(0)
            elif state.dim() > 2:
                state = state.view(-1, state.size(-1))

            # 确保hidden_size一致
            current_size = state.size(-1)
            if current_size != target_hidden_size:
                if current_size > target_hidden_size:
                    state = state[..., :target_hidden_size]
                else:
                    padding = torch.zeros(
                        state.size(0),
                        target_hidden_size - current_size,
                        dtype=state.dtype,
                        device=state.device
                    )
                    state = torch.cat([state, padding], dim=-1)

            # 确保batch_size=1
            if state.size(0) != 1:
                state = state[0:1, :]

            processed_states.append(state)

        try:
            # 在dim=0上拼接 [n, hidden_size]
            result = torch.cat(processed_states, dim=0)
            return result
        except Exception as e:
            logger.error(f"Error in get_recent_hidden_states: {e}")
            logger.error(f"Shapes: {[s.shape for s in processed_states]}")
            return None

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_tokens': self.total_tokens,
            'buffer_size': len(self.tokens),
            'last_check_position': self.last_check_position
        }

    def clear(self):
        """清空缓冲区"""
        self.tokens.clear()
        self.token_texts.clear()
        self.hidden_states.clear()
        self.timestamps.clear()
        self.total_tokens = 0
        self.last_check_position = 0
        self.hidden_size = None