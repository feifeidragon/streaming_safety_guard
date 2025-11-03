# core/streaming_guard.py

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple, Callable
import time
from loguru import logger

from models.safety_detector import SafetyDetector
from models.router import SpecializedRouter  # 或 SafetyRouter
from models.latent_analyzer import LatentDynamicsAnalyzer
from core.buffer_manager import TokenBuffer


class StreamingGuard:
    """
    流式生成安全守卫系统
    实现实时的内容安全检测和拦截
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.dtype = torch.float16 if config.get('use_fp16', True) else torch.float32

        # 加载模型和分词器
        self._load_model_and_tokenizer()

        # 初始化安全检测器 - 修复参数名
        self.safety_detector = SafetyDetector(
            input_dim=self.hidden_size,  # 使用 input_dim 而不是 hidden_size
            num_classes=2,
            dropout=0.1
        ).to(self.device)

        # 转换为fp16
        if self.dtype == torch.float16:
            self.safety_detector = self.safety_detector.half()
            logger.info("Safety detector converted to float16")

        # 加载安全检测器权重（如果存在）
        detector_path = config.get('detector_path')
        if detector_path and os.path.exists(detector_path):
            try:
                self.safety_detector.load_state_dict(torch.load(detector_path, map_location=self.device))
                logger.info(f"Loaded safety detector from {detector_path}")
            except Exception as e:
                logger.warning(f"Failed to load detector weights: {e}")

        # 初始化隐藏状态分析器
        self.latent_analyzer = LatentDynamicsAnalyzer(
            config.get('latent_analyzer', {})
        )

        # 初始化路由器
        router_config = config.get('router', {})
        num_experts = router_config.get('num_experts', 3)
        expert_types = router_config.get('expert_types', ['toxicity', 'violence', 'sexual'])

        # 计算特征维度 (hidden_size // 4 来自SafetyDetector的输出)
        feature_dim = self.hidden_size // 4

        self.router = SpecializedRouter(
            input_dim=feature_dim,
            num_experts=num_experts,
            expert_types=expert_types
        ).to(self.device)

        if self.dtype == torch.float16:
            self.router = self.router.half()
            logger.info("Router converted to float16")

        # 加载路由器权重（如果存在）
        router_path = config.get('router_path')
        if router_path and os.path.exists(router_path):
            try:
                self.router.load_state_dict(torch.load(router_path, map_location=self.device))
                logger.info(f"Loaded router from {router_path}")
            except Exception as e:
                logger.warning(f"Failed to load router weights: {e}")

        # Token缓冲区
        self.token_buffer = TokenBuffer(
            buffer_size=config.get('buffer_size', 20),
            window_size=config.get('window_size', 10)
        )

        # 检测参数
        self.check_interval = config.get('check_interval', 5)
        self.risk_threshold = config.get('risk_threshold', 0.6)
        self.max_tokens = config.get('max_tokens', 512)

        self.stats = {
            'total_checks': 0,
            'blocked_count': 0,
            'total_tokens': 0,
            'avg_risk_score': 0.0
        }

        logger.info("StreamingGuard initialized successfully")

    def _load_model_and_tokenizer(self):
        """加载模型和分词器"""
        model_path = self.config.get('model_path')

        if not model_path:
            raise ValueError("model_path is not set in config!")

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        logger.info(f"Loading model from {model_path}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True
        )

        self.model.eval()
        logger.info("Model loaded successfully")

        # 获取hidden_size
        if hasattr(self.model.config, 'hidden_size'):
            self.hidden_size = self.model.config.hidden_size
        elif hasattr(self.model.config, 'd_model'):
            self.hidden_size = self.model.config.d_model
        else:
            # 尝试从配置中获取
            self.hidden_size = self.config.get('hidden_size', 2560)
            logger.warning(f"Could not detect hidden_size from model, using {self.hidden_size}")

        logger.info(f"Model dtype: {self.dtype}")
        logger.info(f"Model hidden size: {self.hidden_size}")

    def _extract_hidden_states(self, outputs) -> torch.Tensor:
        """
        从模型输出中提取隐藏状态（仅最后一层最后一个token）

        Args:
            outputs: 模型输出对象

        Returns:
            hidden_state: 提取的隐藏状态张量 [1, hidden_size]
        """
        try:
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # 获取最后一层的隐藏状态
                last_layer_hidden = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

                # 只取最后一个token的隐藏状态
                if last_layer_hidden.dim() == 3:
                    last_token_hidden = last_layer_hidden[:, -1, :]  # [batch_size, hidden_size]
                elif last_layer_hidden.dim() == 2:
                    last_token_hidden = last_layer_hidden
                else:
                    last_token_hidden = last_layer_hidden.view(-1, last_layer_hidden.size(-1))

                # 确保是 [1, hidden_size]
                if last_token_hidden.size(0) != 1:
                    last_token_hidden = last_token_hidden[0:1, :]

                return last_token_hidden

            elif hasattr(outputs, 'last_hidden_state'):
                last_hidden = outputs.last_hidden_state
                if last_hidden.dim() == 3:
                    last_token_hidden = last_hidden[:, -1, :]
                else:
                    last_token_hidden = last_hidden

                if last_token_hidden.size(0) != 1:
                    last_token_hidden = last_token_hidden[0:1, :]

                return last_token_hidden

            else:
                logger.warning("No hidden states found in model output")
                return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

        except Exception as e:
            logger.error(f"Error extracting hidden states: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    def _check_safety(self, current_text: str) -> Tuple[bool, float, Dict]:
        """
        执行安全检查

        Args:
            current_text: 当前生成的文本

        Returns:
            is_safe: 是否安全
            risk_score: 风险分数
            details: 详细信息
        """
        self.stats['total_checks'] += 1

        try:
            # 获取最近的隐藏状态
            recent_states = self.token_buffer.get_recent_hidden_states(n=10)

            if recent_states is None or recent_states.size(0) < 3:
                return True, 0.0, {'reason': 'insufficient_data'}

            # 确保维度正确 [1, seq_len, hidden_size]
            if recent_states.dim() == 2:
                recent_states = recent_states.unsqueeze(0)  # [1, seq_len, hidden_size]

            recent_states = recent_states.to(self.device)

            # 安全检测器判断
            with torch.no_grad():
                logits, features = self.safety_detector(recent_states)
                probs = torch.softmax(logits, dim=-1)
                risk_score = probs[0, 1].item()  # 不安全的概率

            # 路由器判断 - features是 [batch_size, hidden_size // 4]
            route_output = self.router(features)  # 返回 (route_logits, expert_outputs)

            if isinstance(route_output, tuple):
                route_logits, expert_outputs = route_output
            else:
                route_logits = route_output
                expert_outputs = None

            route = torch.argmax(route_logits, dim=-1).item()

            # 更新平均风险分数
            self.stats['avg_risk_score'] = (
                (self.stats['avg_risk_score'] * (self.stats['total_checks'] - 1) + risk_score)
                / self.stats['total_checks']
            )

            # 综合判断
            is_safe = risk_score < self.risk_threshold and route != 2

            details = {
                'risk_score': risk_score,
                'route': route,
                'route_name': ['safe', 'warning', 'block'][route] if route < 3 else 'unknown',
                'text_sample': current_text[-50:] if len(current_text) > 50 else current_text
            }

            if not is_safe:
                self.stats['blocked_count'] += 1
                logger.warning(f"Unsafe content detected: risk={risk_score:.3f}, route={route}")

            return is_safe, risk_score, details

        except Exception as e:
            logger.error(f"Error in safety check: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 出错时保守处理，认为不安全
            return False, 1.0, {'reason': 'check_error', 'error': str(e)}

    def generate_safe(
            self,
            prompt: str,
            max_new_tokens: int = 100,
            temperature: float = 0.7,
            top_p: float = 0.9,
            top_k: int = 50,
            callback: Optional[Callable[[str], None]] = None
    ) -> Dict:
        """
        安全生成文本

        Args:
            prompt: 输入提示词
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数
            callback: 回调函数，用于实时输出 callback(token_text)

        Returns:
            result: 生成结果字典
        """
        # 重置状态
        self.token_buffer.clear()
        self.latent_analyzer.reset()

        # 编码输入
        if not isinstance(prompt, str):
            if prompt is None:
                raise ValueError("prompt 不能为 None")
            elif isinstance(prompt, (list, dict)):
                raise ValueError(f"prompt 必须是字符串，但收到了 {type(prompt)}")
            else:
                prompt = str(prompt)

        # 额外检查：确保 prompt 不是空字符串
        if not prompt.strip():
            raise ValueError("prompt 不能为空字符串")

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']

        generated_tokens = []
        generated_text = ""
        blocked = False
        block_reason = None
        block_step = -1

        start_time = time.time()

        with torch.no_grad():
            for step in range(max_new_tokens):
                # 生成下一个token
                outputs = self.model(
                    input_ids,
                    output_hidden_states=True,
                    return_dict=True
                )

                # 获取logits
                logits = outputs.logits[:, -1, :] / temperature

                # Top-k过滤
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(1, top_k_indices, top_k_values)
                    logits = logits_filtered

                # Top-p采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                # 采样
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # 解码token
                next_token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)

                # 提取隐藏状态（仅最后一层最后一个token）
                hidden_state = self._extract_hidden_states(outputs)

                # 添加到缓冲区
                self.token_buffer.add_token(
                    token_id=next_token.item(),
                    token_text=next_token_text,
                    hidden_states=hidden_state,
                    timestamp=time.time()
                )

                generated_tokens.append(next_token.item())
                generated_text += next_token_text

                # 回调输出
                if callback:
                    try:
                        callback(next_token_text)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                # 定期安全检查
                if self.token_buffer.should_check(self.check_interval):
                    current_text = self.token_buffer.get_current_text()
                    is_safe, risk_score, details = self._check_safety(current_text)

                    if not is_safe:
                        blocked = True
                        block_reason = details
                        block_step = step
                        logger.warning(f"Content blocked at step {step}: {details}")
                        break

                    self.token_buffer.mark_checked()

                # 检查结束条件
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # 更新input_ids
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                self.stats['total_tokens'] += 1

        generation_time = time.time() - start_time

        result = {
            'prompt': prompt,
            'generated_text': generated_text,
            'blocked': blocked,
            'block_reason': block_reason,
            'block_step': block_step,
            'num_tokens': len(generated_tokens),
            'generation_time': generation_time,
            'tokens_per_second': len(generated_tokens) / generation_time if generation_time > 0 else 0,
            'stats': self.stats.copy()
        }

        return result

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_checks': 0,
            'blocked_count': 0,
            'total_tokens': 0,
            'avg_risk_score': 0.0
        }

    def save_models(self, save_dir: str):
        """保存模型权重"""
        os.makedirs(save_dir, exist_ok=True)

        detector_path = os.path.join(save_dir, 'safety_detector.pt')
        router_path = os.path.join(save_dir, 'router.pt')

        torch.save(self.safety_detector.state_dict(), detector_path)
        torch.save(self.router.state_dict(), router_path)

        logger.info(f"Models saved to {save_dir}")
