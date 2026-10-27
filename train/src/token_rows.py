"""历史benchmark adapter推理兼容入口；不提供LoRA训练。"""
from train.src.infrastructure.legacy_token_rows import load_adapter
__all__ = ['load_adapter']
