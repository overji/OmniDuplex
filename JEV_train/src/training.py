"""分类器和Thinker分别进行ZeRO-3全参SFT；媒体特征每次前向重算，不使用PEFT。"""
import json
from pathlib import Path
import random
from types import SimpleNamespace

from .config import PROTOCOL, write, digest, base_identity


def optimize(rows, engine, config, output, on_epoch=None):
    """显式设置尾批边界并补偿DS的loss/GAS缩放，保持不足一组时按真实QA数平均。"""
    import torch
    rng = random.Random(config['seed']); history = []; step = 0
    gas = config['gradient_accumulation_steps']
    for epoch in range(config['epochs']):
        order = list(rows); rng.shuffle(order)
        for start in range(0, len(order), gas):
            batch = order[start:start + gas]; value = 0.
            for index, row in enumerate(batch):
                engine.set_gradient_accumulation_boundary(index == len(batch) - 1)
                try:
                    loss = engine(row)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite full-SFT loss')
                    engine.backward(loss * gas / len(batch))
                    value += loss.item() / len(batch)
                    engine.step()
                except Exception as exc:
                    write(output/'training_error.json',dict(sample_index=row['sample_index'],error=f'{type(exc).__name__}: {exc}'))
                    raise
            norm = engine.get_global_grad_norm()
            norm = float(norm) if norm is not None else None
            if norm is None or not torch.isfinite(torch.tensor(norm)) or norm == 0:
                raise RuntimeError('Missing/nonfinite full-SFT gradient')
            step += 1
            history.append(dict(step=step,epoch=epoch,loss=value,gradient_norm=norm,logical_batch=len(batch)))
            if config['rank'] == 0:write(output/'metrics.json',history)
        if on_epoch:on_epoch(epoch)


def backend_example(session,row,config):
    """只在assistant正文和结束符上监督，历史回复与全部媒体仍完整留在因果上下文。"""
    import torch
    from .backend import merge_packets
    from .media import units
    session.start_session(row['sample'],execute=False)
    events={e['second']:e for e in row['events']};target_indices=[]
    for unit in units(row,config):
        session.append_media(unit,execute=False)
        if unit.second in events:
            session.begin_reply(execute=False);target_indices.append(len(session.packets))
            session.end_reply(events[unit.second]['target'])
    batch=merge_packets(session.packets);labels=torch.full_like(batch['input_ids'],-100);offset=0
    for i,packet in enumerate(session.packets):
        n=packet['input_ids'].shape[1]
        if i in target_indices:labels[:,offset:offset+n]=packet['input_ids']
        offset+=n
    batch['labels']=labels
    return batch


def train(rows,validation,config,output,stage):
    import torch
    import deepspeed
    from transformers import set_seed
    from train.src.infrastructure.distributed import initialize_distributed
    from train.src.infrastructure.zero import register_external_parameters, wrap_audio_position_table
    from train.src.infrastructure.checkpoints import save
    from .graphs import GateGraph, BackendGraph
    config=dict(config);device_key='classifier_device' if stage=='classifier' else 'backend_device'
    chosen=torch.device(config[device_key]);args=SimpleNamespace(local_rank=config.get('local_rank',chosen.index or 0))
    # torchrun时各rank只使用其LOCAL_RANK；单进程仍沿用配置中的设备，不自动挑卡。
    import os
    if 'LOCAL_RANK' not in os.environ:args.local_rank=chosen.index or 0
    device,rank,world=initialize_distributed(args,config);config[device_key]=str(device)
    exists=torch.tensor(int(output.exists()) if rank==0 else 0,device=device)
    torch.distributed.broadcast(exists,src=0)
    if exists.item():raise ValueError('Use a fresh full-SFT output directory')
    config.update(rank=rank,world_size=world,jev_stage=stage,protocol=PROTOCOL)
    if rank==0:output.mkdir(parents=True);write(output/'run_config.json',dict(protocol=PROTOCOL,config=config))
    torch.distributed.barrier();torch.set_num_threads(config.get('cpu_threads',4));set_seed(config['seed'])
    if stage=='classifier':
        from .gate import load,encode
        state=load(config,training=True);graph=GateGraph(state,config);tokenizer=state.tokenizer
    else:
        from .backend import load
        state=load(config,training=True);graph=BackendGraph(state,config);tokenizer=state.tokenizer
    register_external_parameters(graph,graph.model)
    settings=json.loads(Path(config['deepspeed_config']).read_text())
    settings.update(train_micro_batch_size_per_gpu=1,gradient_accumulation_steps=config['gradient_accumulation_steps'],gradient_clipping=config['max_grad_norm'])
    settings['bf16']={'enabled':config['dtype']=='bfloat16'}
    settings['fp16']={'enabled':False}
    settings['optimizer']['params'].update(lr=config['learning_rate'],weight_decay=config['weight_decay'])
    engine,_,_,_=deepspeed.initialize(model=graph,model_parameters=[p for p in graph.parameters() if p.requires_grad],config=settings)
    wrap_audio_position_table(graph.model)
    excluded=[]
    if stage=='classifier':
        # 只做长度预检，不缓存embedding；正式loss逐次重新编码，保留音视频梯度。
        accepted=[];graph.eval()
        with torch.no_grad():
            for row in rows:
                try:encode(state,row,config);accepted.append(row)
                except ValueError as exc:excluded.append(dict(sample_index=row['sample_index'],error=str(exc)))
        rows=accepted
    if not rows:raise ValueError('No complete full-SFT trajectories fit the configured limits')
    if rank==0:write(output/'excluded.json',excluded)
    def validate(epoch):
        graph.eval();values=[];errors=[];probs=[];labels=[]
        with torch.no_grad():
            for row in validation:
                try:
                    value=engine(row);values.append(value.item())
                    if stage=='classifier':probs+=graph.last_probabilities;labels+=row['labels']
                except ValueError as exc:errors.append(dict(sample_index=row['sample_index'],error=str(exc)))
        report=dict(loss=sum(values)/len(values) if values else None,excluded=errors)
        if stage=='classifier':
            from .metrics import binary_metrics
            report.update(history='teacher_forced_reference_replies',metrics=binary_metrics(labels,probs,config['threshold']))
        if rank==0:write(output/f'validation_epoch_{epoch}.json',report)
        graph.train()
    graph.train();optimize(rows,engine,config,output,validate)
    identity=dict(protocol=PROTOCOL,stage=stage,base_identity=base_identity(config['classifier_model' if stage=='classifier' else 'backend_model']),
                  prepared_manifest_sha256=config.get('prepared_manifest_sha256'),tokenizer_sha256=digest(tokenizer.get_vocab()))
    save(engine,output/'checkpoint-final',config['epochs'],0,identity,config,SimpleNamespace(tokenizer=tokenizer),rank)
    torch.distributed.destroy_process_group()


def classifier(rows,validation,config,output):
    train(rows,validation,config,output,'classifier')


def backend(rows,validation,config,output):
    if config['mode']!='separate_sft':raise ValueError('train-backend requires mode=separate_sft')
    train(rows,validation,config,output,'backend')


def export_checkpoint(checkpoint,output):
    from .checkpoints import export_checkpoint as export
    export(checkpoint,output)
