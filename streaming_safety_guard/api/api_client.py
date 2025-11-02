import requests
from typing import Dict, Optional, Iterator
import json
from loguru import logger


class LLMAPIClient:
    """
    大模型API客户端
    支持OpenAI、Anthropic等API
    """

    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config.get('openai_key') or config.get('anthropic_key')
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        self.model_name = config.get('model_name', 'gpt-4')
        self.provider = self._detect_provider()

    def _detect_provider(self) -> str:
        """检测API提供商"""
        if 'anthropic' in self.base_url.lower():
            return 'anthropic'
        else:
            return 'openai'

    def generate_stream(
            self,
            prompt: str,
            max_tokens: int = 512,
            temperature: float = 0.7
    ) -> Iterator[str]:
        """
        流式生成
        Yields:
            生成的文本片段
        """
        if self.provider == 'openai':
            yield from self._openai_stream(prompt, max_tokens, temperature)
        elif self.provider == 'anthropic':
            yield from self._anthropic_stream(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _openai_stream(
            self,
            prompt: str,
            max_tokens: int,
            temperature: float
    ) -> Iterator[str]:
        """OpenAI流式API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
            'temperature': temperature,
            'stream': True
        }

        try:
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=data,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]
                        if line.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(line)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _anthropic_stream(
            self,
            prompt: str,
            max_tokens: int,
            temperature: float
    ) -> Iterator[str]:
        """Anthropic流式API"""
        headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }

        data = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
            'temperature': temperature,
            'stream': True
        }

        try:
            response = requests.post(
                f'{self.base_url}/messages',
                headers=headers,
                json=data,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]
                        try:
                            chunk = json.loads(line)
                            if chunk.get('type') == 'content_block_delta':
                                delta = chunk.get('delta', {})
                                content = delta.get('text', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise