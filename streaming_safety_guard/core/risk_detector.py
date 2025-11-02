import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import re
import time


class RiskDetector:
    """
    风险检测器
    整合多种检测策略
    """

    def __init__(self, config: Dict):
        self.config = config
        self.risk_categories = config.get('risk_categories', [])

        # 关键词匹配
        self.keyword_patterns = self._build_keyword_patterns()

        # 检测历史
        self.detection_history = []
        self.max_history = 100

    def _build_keyword_patterns(self) -> Dict[str, List[re.Pattern]]:
        """构建关键词模式"""
        patterns = {}
        for category in self.risk_categories:
            category_name = category['name']
            keywords = category['keywords']
            patterns[category_name] = [
                re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
                for kw in keywords
            ]
        return patterns

    def quick_keyword_check(self, text: str) -> Tuple[bool, Dict[str, float]]:
        """
        快速关键词检查（低延迟）
        Returns:
            is_risky: 是否检测到风险关键词
            scores: 各类别的匹配得分
        """
        scores = {}
        is_risky = False

        for category_name, patterns in self.keyword_patterns.items():
            match_count = sum(
                1 for pattern in patterns if pattern.search(text)
            )
            score = min(match_count / len(patterns), 1.0) if patterns else 0.0
            scores[category_name] = score

            if score > 0.3:  # 阈值
                is_risky = True

        return is_risky, scores

    def statistical_check(self, text: str) -> Dict[str, float]:
        """
        统计特征检查
        """
        features = {}

        # 文本长度特征
        features['length'] = len(text)
        features['word_count'] = len(text.split())

        # 特殊字符比例
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        features['special_char_ratio'] = special_chars / max(len(text), 1)

        # 大写字母比例
        upper_count = sum(1 for c in text if c.isupper())
        features['upper_ratio'] = upper_count / max(len(text), 1)

        # 重复字符检测
        features['repetition_score'] = self._detect_repetition(text)

        return features

    def _detect_repetition(self, text: str) -> float:
        """检测重复模式"""
        if len(text) < 3:
            return 0.0

        # 检测连续重复字符
        max_repeat = 0
        current_repeat = 1
        for i in range(1, len(text)):
            if text[i] == text[i - 1]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1

        return min(max_repeat / 10.0, 1.0)

    def combine_scores(
            self,
            keyword_scores: Dict[str, float],
            model_scores: Dict[str, float],
            statistical_features: Dict[str, float],
            latent_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        组合多个检测源的得分
        Returns:
            final_score: 最终风险得分
            detailed_scores: 详细得分
        """
        detailed_scores = {}

        # 1. 关键词得分（权重0.2）
        keyword_avg = sum(keyword_scores.values()) / max(len(keyword_scores), 1)
        detailed_scores['keyword'] = keyword_avg

        # 2. 模型得分（权重0.5）
        model_avg = sum(model_scores.values()) / max(len(model_scores), 1)
        detailed_scores['model'] = model_avg

        # 3. 统计特征得分（权重0.1）
        stat_score = (
                statistical_features.get('special_char_ratio', 0) * 0.3 +
                statistical_features.get('upper_ratio', 0) * 0.3 +
                statistical_features.get('repetition_score', 0) * 0.4
        )
        detailed_scores['statistical'] = stat_score

        # 4. 隐藏状态分析得分（权重0.2）
        latent_score = 0.0
        if latent_metrics:
            latent_score = latent_metrics.get('risk_score', 0)
        detailed_scores['latent'] = latent_score

        # 加权组合
        final_score = (
                0.2 * keyword_avg +
                0.5 * model_avg +
                0.1 * stat_score +
                0.2 * latent_score
        )

        return final_score, detailed_scores

    def record_detection(self, result: Dict):
        """记录检测结果"""
        self.detection_history.append({
            'timestamp': time.time(),
            'result': result
        })

        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)