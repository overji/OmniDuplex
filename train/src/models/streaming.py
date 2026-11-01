"""将逐秒媒体与文本输出接到 MiniCPM 原生流式接口上。

SFT：trajectory.py 排定参考回复时间 → replay() 收集可微 embedding 和标签
     → graph.py 执行 LLM 前向 → sft_loss.py 计算分组交叉熵。
GRPO：rollout() 无梯度自主采样 → 外部奖励函数评分并计算优势
      → replay() 可微重建相同轨迹 → graph.py 计算动作概率和策略 loss。

两条路径都先输入当前秒媒体，再处理该秒输出。本文件不计算奖励或最终 loss。
"""

import math
import torch
from benchmark.data.prompts import user_request
from benchmark.data.task_text import TaskTextStream
from train.src.domain.trajectory import compile_trajectory, second_target, schedule_stats, supervision_metadata, TrajectoryError
from train.src.domain.grpo import action_log_probs, new_response_started


class SequenceTooLong(TrajectoryError):
    """序列位置数超预算；沿用轨迹异常机制，交由调用方排除或缩短可选尾部。"""
    pass


def system_prompt(record, config):
    # 统一使用训练协议的提示构造器，避免把参考回复和标注窗口拼进系统提示。
    from train.src.domain.agent import system_prompt as render
    return render(record['sample'], config)


class GraphRecorder:
    """临时替代原生 decoder，接收 embedding，但不执行 LLM 解码。

    保留文本 embedding 和媒体编码结果的计算图，不 detach。
    parts 还保留每次 feed 的边界，供 GRPO 在 graph.py 中按原分段回放 KV。
    """
    def __init__(self, embedding, limit):
        self.embedding, self.limit = embedding, limit
        # length 统计展开后的序列位置：媒体 embedding 也占预算，不只是文本。
        self.parts, self.length = [], 0

    def embed_token(self, ids):
        # 单个 ID 和 ID 列表都整理为一维，返回 [位置数, hidden_size]。
        ids = torch.as_tensor(ids, device=self.embedding.weight.device, dtype=torch.long).reshape(-1)
        return self.embedding(ids)

    def feed(self, embeds, return_logits=False):
        # 参数/返回值保留原 decoder 的调用形式；此处只录制，不返回 logits。
        embeds = embeds.reshape(-1, embeds.shape[-1])
        self.parts.append(embeds)
        self.length += len(embeds)
        if self.length > self.limit:
            raise SequenceTooLong('sequence_exceeds_max_sequence_tokens')
        return None, None

    def register_unit_start(self):
        # 满足原生 prefill 的接口调用；录制阶段无需维护解码器的 unit 状态。
        pass


def replay(model, duplex, record, clip, config, trajectory=None):
    """按参考调度或已采样轨迹，重建带梯度的媒体/文本输入。

    trajectory=None：SFT teacher forcing，按秒插入参考回复或 listen 标签。
    trajectory 非空：GRPO 回放原先采样的 token，不再次采样、不插入参考答案。
    是否启用梯度由调用方上下文决定；训练回放可反传，无梯度预检也复用这里。

    返回 inputs_embeds/segments 供 graph.py 前向；labels/groups/roles 供 SFT
    计算 loss；positions/token_ids/turns 供 GRPO 提取实际采样动作的概率。
    """
    schedule = None
    if trajectory is None:
        # 使用实际视频时长重新调度；要求 clip 覆盖整个所需时间线。
        schedule = compile_trajectory(record, duplex.tokenizer, config,
                                      duration=getattr(clip, 'duration', None))
        if len(clip.chunks) != schedule['horizon']:
            raise TrajectoryError('clip_does_not_cover_complete_trajectory')
    decoder = duplex.decoder
    recorder = GraphRecorder(model.llm.get_input_embeddings(), config['max_sequence_tokens'])
    duplex.decoder = recorder
    # 之后原生 prefill 产生的文本/媒体 embedding 都进入 recorder，暂不跑 LLM。
    # 下列 SFT 数组与 embedding 逐位置对齐；GRPO 三个动作数组只记录采样位置。
    labels, groups, positions, token_ids, turns = [], [], [], [], []
    # event_roles：0=非决策，1/2=首次回复前/后的沉默，3/4=首次/后续开口，
    # 5=续说，6=turn_eos，7=其他结构。event_ids 为按调度排序的回复编号。
    event_roles, event_ids = [], []
    decision_roles = []  # 1=idle listen, 2=fresh response onset, 3=structure
    def align():
        # prefill 可以一次追加多个媒体/模板位置；默认全部 mask，再单独填写输出。
        # -100 表示不计算该位置的目标 CE；这些位置仍可作为后续预测的上下文。
        labels.extend([-100] * (recorder.length - len(labels)))
        groups.extend([0] * (recorder.length - len(groups)))
        decision_roles.extend([0] * (recorder.length - len(decision_roles)))
        event_roles.extend([0] * (recorder.length - len(event_roles)))
        event_ids.extend([-1] * (recorder.length - len(event_ids)))
    def append(ids, kinds=None, actions=None):
        start = recorder.length
        if trajectory is not None and actions is not None:
            # 原生生成逐 token 回灌：连同强制插入的 token 一起保持 feed 边界。
            # 训练图据此回放分段 KV，减少与原生采样概率的数值偏差。
            for token in ids:
                recorder.feed(recorder.embed_token(token))
        else:
            recorder.feed(recorder.embed_token(ids))
        align()
        if kinds:
            # kinds 是 loss 组编号，不是数值权重：1=决策，2=回复，3=instruction。
            # 组 0 不覆盖 align() 填入的 -100；真正的权重在 loss 模块中应用。
            for offset, (token, group) in enumerate(zip(ids, kinds)):
                if group:
                    labels[start + offset], groups[start + offset] = token, group
        for action in actions or []:
            # 自回归位移：token 位于 p，必须用 p-1 的 hidden state 预测。
            # ids 包含全部输出，但 actions 仅含真正采样的动作，强制 token 不计策略项。
            positions.append(start + action['offset'] - 1)
            token_ids.append(ids[action['offset']])
            turns.append(action['turn_open'])
    try:
        # 每条 QA 从独立流式状态开始，避免上一条的音频/会话历史混入。
        duplex._reset_streaming_state()
        model.init_streaming_processor()
        prompt = '<|im_start|>system\n' + system_prompt(record, config) + '<|im_end|>'
        append(duplex.tokenizer.encode(prompt, add_special_tokens=False))
        sent = False
        processed_seconds = 0
        # 缓存媒体预处理以复用；可微媒体编码仍由下面的原生 prefill 执行。
        with clip.preprocessing_cache(duplex.processor):
            for i, inputs in enumerate(clip.chunks):
                second = i + 1
                # i 对应媒体区间 [i, i+1)，输出时刻为 i+1。
                # 保存本秒开始前的录制长度；可选尾部超限时只能整秒撤回。
                checkpoint = (len(recorder.parts), recorder.length, len(labels))
                try:
                    if not sent and i >= clip.question_time:
                        # 问题先于该秒媒体插入，且只插入一次。
                        # __wrapped__ 绕过原 prefill 的装饰层，保留训练所需梯度。
                        result = type(duplex).streaming_prefill.__wrapped__(duplex, text_list=[user_request(record['sample'])])
                        if not result['success']:
                            raise RuntimeError(result)
                        sent = True
                    # 先编码并输入本秒音视频，再追加本秒输出，保持因果输入顺序。
                    result = type(duplex).streaming_prefill.__wrapped__(duplex, **inputs)
                    if not result['success']:
                        raise RuntimeError(result)
                    align()
                    if trajectory is not None:
                        # 回灌原 rollout 的完整 token 序列，记录其采样动作预测位置。
                        chunk = trajectory['chunks'][i]
                        append(chunk['ids'], actions=chunk['actions'])
                    else:
                        # 有参考 chunk 就监督说话；否则由 second_target 决定 listen
                        # 是否计 loss（问题前 mask，问题后可监督，包含参考窗口内空闲秒）。
                        ids, kinds = second_target(schedule, second, duplex.tokenizer, config, sent)
                        start = recorder.length
                        append(ids, kinds)
                        fine, events = supervision_metadata(schedule, second, ids, kinds, duplex.tokenizer)
                        # event_balanced 使用细角色和回复编号做分层/按事件平均。
                        event_roles[start:start + len(ids)] = fine
                        event_ids[start:start + len(ids)] = events
                        fresh = any(t['onset_seconds'] == second for t in schedule['targets'])
                        # balanced_gate 只需三类：沉默、新开口、其余结构。
                        # 每个 chunk 都有 speak，但仅回复开始秒的首 token 算新开口。
                        for offset, (token, kind) in enumerate(zip(ids, kinds)):
                            if kind == 1:
                                decision_roles[start + offset] = (1 if token == duplex.tokenizer.convert_tokens_to_ids('<|listen|>')
                                    else 2 if fresh and offset == 0 else 3)
                        # 时间单元结束符只作上下文；未传 kinds，所以不直接计算 CE。
                        append([duplex.tokenizer.convert_tokens_to_ids('</unit>')])
                except SequenceTooLong:
                    # 必要回复、GRPO 和 full_video SFT 超限都继续抛错。
                    # 仅开启额外尾部且全部回复已完成的 SFT 允许缩短；判断依据是
                    # response_end_seconds，可能早于最后参考窗口结束后的 tail_start。
                    if not (schedule is not None and config.get('sft_horizon') != 'full_video'
                            and config.get('post_response_silence_seconds', 0)
                            and second > schedule['response_end_seconds']):
                        raise
                    # 撤回本秒全部 embedding/标签并结束，不丢回复、不保留半秒输入。
                    # 此分支随即退出循环，不再继续使用本秒已改变的流式状态。
                    parts, length, label_count = checkpoint
                    del recorder.parts[parts:]
                    recorder.length = length
                    del labels[label_count:]
                    del groups[label_count:]
                    del decision_roles[label_count:]
                    del event_roles[label_count:]
                    del event_ids[label_count:]
                    break
                processed_seconds = second
        # SFT 在 graph.py 拼接后的全序列上前向；GRPO 还需要 segments 的 feed 边界。
        # 标签先保存在 CPU，后续 graph.py 会将其移到 hidden state 所在设备。
        encoded = dict(inputs_embeds=torch.cat(recorder.parts), segments=recorder.parts, labels=torch.tensor(labels),
            groups=torch.tensor(groups), event_roles=torch.tensor(event_roles), event_ids=torch.tensor(event_ids), decision_roles=torch.tensor(decision_roles), positions=positions, token_ids=token_ids, turns=turns)
        if schedule is not None:
            encoded.update(schedule_stats(schedule))
            # 记录实际成功录入的尾部长度，不能沿用可能因 token 预算缩短前的计划值。
            encoded['tail_silence_seconds'] = max(0, processed_seconds - schedule['tail_start_seconds'] + 1)
        return encoded
    finally:
        # 即使报错也恢复原 decoder，释放模型持有的音频图引用。
        # 已返回的 embedding 仍保留反传所需计算图。
        duplex.decoder = decoder
        model.audio_past_key_values = None


@torch.no_grad()
def rollout(model, duplex, record, clip, config, seed):
    """无梯度生成一条完整候选轨迹，保存 GRPO 回放和奖励计算所需记录。

    每秒先输入媒体，再自主采样；上下文只反馈自身输出，不注入参考回复或执行工具。
    chunks 保存逐秒 ids/actions/new_response/idle_wait，供回放和沉默/FP 奖励使用；
    events 保存从跨秒 token 流解析出的回复/委托，供时间匹配和内容评分使用。
    此处只采样和记录；奖励、优势和反向更新均由外部训练循环完成。
    """
    from transformers import set_seed
    set_seed(seed)
    model.eval()
    # eval 控制模型运行模式；函数上的 no_grad 负责关闭采样阶段的自动求导。
    if config.get('batch_prompt_prefill', True):
        # 原生 prepare 逐 token 输入系统前缀，会触发大量 ZeRO hook。
        # 在关闭滑动窗口的使用条件下，将相同因果 token 一次 prefill，减少开销。
        duplex.clear_break_event(); duplex.clear_session_stop()
        duplex._reset_streaming_state(); duplex.decoder.reset()
        model.init_streaming_processor()
        prefix = duplex.tokenizer.encode('<|im_start|>system\n' + system_prompt(record, config), add_special_tokens=False)
        suffix = duplex.tokenizer.encode('<|im_end|>', add_special_tokens=False)
        duplex.decoder.feed(duplex.decoder.embed_tokens(prefix + suffix))
        duplex.decoder.register_system_prompt()
    else:
        duplex.prepare(prefix_system_prompt=system_prompt(record, config))
    # 问题前的整数秒强制等待；与循环中 i >= question_time 的问题插入时机对应。
    duplex.force_listen_count = math.ceil(clip.question_time)
    native_decode = duplex.decoder.decode
    actions = []
    def decode(logits, **kwargs):
        # law 是实际回灌 token 的 log-prob 分布，已处理原生 EOS 抽样和
        # 未结束回复中的 listen→speak 映射；不能直接用未经处理的 logits softmax。
        law = action_log_probs(logits, duplex.decoder.forbidden_token_ids,
            duplex.chunk_eos_token_id, duplex.listen_token_id, duplex.tts_bos_token_id,
            not duplex.current_turn_ended)[0]
        token = torch.multinomial(law.exp(), 1)
        if torch.distributed.is_initialized() and not config.get('grpo_candidate_parallel', False):
            # 所有 rank 共享同一 QA/采样轨迹，按 rank 0 的 token 继续原生解码。
            torch.distributed.broadcast(token, src=0)
        # before 和 actions 在每秒生成前重置；闭包记录该 token 的秒内位置、
        # 采样前 turn 状态和旧策略概率，训练时据此计算 new/old 概率比。
        actions.append(dict(offset=len(duplex.total_ids) - before, turn_open=not duplex.current_turn_ended,
                            old_log_prob=law[token.item()].item()))
        return token
    duplex.decoder.decode = decode
    chunks, events = [], []
    # 解析器跨秒保留文本/分隔符状态；一条回复可由多个 chunk 组成。
    parser, sent = TaskTextStream(duplex.tokenizer), False
    try:
        with clip.preprocessing_cache(duplex.processor):
            for i, inputs in enumerate(clip.chunks):
                if not sent and i >= clip.question_time:
                    # 与 replay 使用相同的问题→媒体→输出顺序；此处运行原生推理接口。
                    result = duplex.streaming_prefill(text_list=[user_request(record['sample'])])
                    if not result['success']:
                        raise RuntimeError(result)
                    sent = True
                result = duplex.streaming_prefill(**inputs)
                if not result['success']:
                    raise RuntimeError(result)
                before = len(duplex.total_ids)
                actions = []
                # 必须保存生成前的状态，才能区分真正空闲等待和未结束回复的续说。
                turn_open_before = not duplex.current_turn_ended
                # 与 action_log_probs 的分布假设一致：不额外改温度、截断、
                # 重复惩罚或 listen 缩放，否则采样概率与训练回放可能不一致。
                result = duplex.streaming_generate(max_new_speak_tokens_per_chunk=config['max_speak_tokens_per_chunk'],
                    decode_mode='sampling', temperature=1., top_k=0, top_p=1.,
                    text_repetition_penalty=1., listen_prob_scale=1.)
                ids = list(duplex.total_ids[before:])
                # ids 包括采样和原生强制插入的 token，actions 只含采样动作。
                # 从原生动作识别新开口，续说不算；空开口也不能因无正文事件而漏计。
                terminators = set(getattr(duplex, 'chunk_terminator_token_ids',
                                          [duplex.listen_token_id, duplex.chunk_eos_token_id]))
                new_response = new_response_started(ids, actions, terminators)
                # 没有新开口不等于空闲：续说时 new_response=False，但 idle_wait
                # 仍为 False。普通沉默分看前者，混淆矩阵 TN 奖励要求后者。
                chunks.append(dict(ids=ids, actions=actions, new_response=new_response,
                    idle_wait=bool(result['is_listen'] and not turn_open_before and not new_response)))
                if duplex.decoder.get_cache_length() > config['max_sequence_tokens']:
                    # 不截断后继续评分，避免只覆盖部分参考窗口却获得完整轨迹奖励。
                    raise SequenceTooLong('Rollout exceeds token limit; discard whole group')
                if sent:
                    # 仅问题输入后解析回复事件；listen/回复结束会触发解析收尾，
                    # interrupted 则保留原生打断语义。时间使用输出秒 i+1。
                    event = parser.feed(ids, i + 1, ended=result['end_of_turn'] or result['is_listen'], interrupted=result.get('is_interrupt', False))
                    if event:
                        events.append(event)
        # 视频结束仍有未完成文本时，保留未完成事件，不能静默丢掉尝试。
        pending = parser.finish(complete=False)
        if pending:
            events.append(pending)
        return dict(chunks=chunks, events=events, seed=seed)
    finally:
        # 正常返回或异常退出都恢复原采样函数，并清理本条轨迹的推理缓存。
        duplex.decoder.decode = native_decode
        duplex.decoder.reset()
        duplex.total_hidden = []
        model.audio_past_key_values = None
