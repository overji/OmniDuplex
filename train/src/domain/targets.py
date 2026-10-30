"""将单条回复编译为原生speak/chunk标签。损失组1/2/3分别表示决策、内容和委托instruction。"""

from train.src.domain.agent import PROTOCOL, require_protocol, training_record


def target_chunks(record, tokenizer, budget, *, protocol=PROTOCOL):
    """Teacher-forced text with native speak/turn/chunk markers, split across seconds.
    
    返回[(token_ids, loss_groups)]。0为mask，1为控制/分隔符，2为回复，3为instruction；跨chunk不重复分隔符。"""
    require_protocol({'agent_output_protocol': protocol})
    from train.src.domain.agent import validate_text_separator
    validate_text_separator(tokenizer)
    record = training_record(record)
    reference, sample = record['reference'], record['sample']
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    if sample['mode'] == 'agent':
        calls = reference.get('tool_calls') or []
        kind = sample.get('sample_type', sample.get('generation_context', {}).get('sample_type'))
        if kind in ('missing_tool', 'no_context') and calls:
            raise ValueError('Negative Agent reference must not contain a tool call')
        from train.src.domain.agent import instruction, SEPARATOR, target_text
        target_text(reference)  # Validate the same two-field target used by JEV.
        task = instruction(reference)
        parts = [(encode(reference['response']), 2)]
        if task is not None:
            parts.extend([(encode(SEPARATOR), 1), (encode(task), 3)])
    else:
        parts = [(encode(reference['response']), 2)]
    body = [(token, group) for tokens, group in parts for token in tokens]
    chunks = []
    while body:
        take, body = body[:budget - 3], body[budget - 3:]
        ids = [tokenizer.convert_tokens_to_ids('<|speak|>')] + [x[0] for x in take]
        groups = [1] + [x[1] for x in take]
        if not body:
            ids.append(tokenizer.convert_tokens_to_ids('<|turn_eos|>')); groups.append(1)
        # Native runtime inserts chunk_eos on budget exhaustion. Keep it as context, not a target.
        ids.append(tokenizer.convert_tokens_to_ids('<|chunk_eos|>')); groups.append(0)
        chunks.append((ids, groups))
    if not chunks:
        raise ValueError('Empty training target')
    return chunks
