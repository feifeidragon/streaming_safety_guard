from typing import List, Tuple
import torch

def merge_tokens_to_text(tokens: List[str]) -> str:
    """合并token列表为文本"""
    return ''.join(tokens).replace('▁', ' ').strip()

def calculate_perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算困惑度"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    perplexity = torch.exp(loss)
    return perplexity.item()

def truncate_to_max_length(text: str, tokenizer, max_length: int) -> str:
    """截断文本到最大长度"""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)