"""对外兼容入口；唯一实现位于 domain.agent，训练内部请直接使用新分层路径。"""

from train.src.domain.agent import (
    PROTOCOL,
    RECORD_FORMAT,
    SEPARATOR,
    training_sample,
    training_record,
    require_protocol,
    validate_prepared,
    instruction,
    target_text,
    plain_prompt,
    system_prompt,
)

__all__ = ['PROTOCOL', 'RECORD_FORMAT', 'SEPARATOR', 'training_sample', 'training_record', 'require_protocol', 'validate_prepared', 'instruction', 'target_text', 'plain_prompt', 'system_prompt']
