"""全参模型身份兼容入口；不再提供LoRA训练或adapter保存加载。"""
from train.src.infrastructure.checkpoints import full_model_identity, tokenizer_metadata
__all__ = ['full_model_identity', 'tokenizer_metadata']
