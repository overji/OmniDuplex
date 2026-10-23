"""Small-scale separate LoRA SFT; no joint gradient or automatic GPU launch."""
import json
from pathlib import Path
import random

from .config import PROTOCOL,write,digest,base_identity


def setup(config,output):
    import torch
    from transformers import set_seed
    if output.exists():raise ValueError('Use a fresh training directory; optimizer resume is not implemented')
    if not torch.cuda.is_available():raise RuntimeError('Real model training requires CUDA')
    set_seed(config['seed']);output.mkdir(parents=True);write(output/'run_config.json',dict(protocol=PROTOCOL,config=config))


def optimize(items,parameters,config,loss_fn,output,on_epoch=None):
    import torch
    parameters=list(parameters);optimizer=torch.optim.AdamW(parameters,lr=config['learning_rate'],weight_decay=config['weight_decay'])
    rng=random.Random(config['seed']);step=0;history=[]
    for epoch in range(config['epochs']):
        order=list(items);rng.shuffle(order)
        for start in range(0,len(order),config['gradient_accumulation_steps']):
            batch=order[start:start+config['gradient_accumulation_steps']];optimizer.zero_grad(set_to_none=True);value=0.
            for row in batch:
                try:loss=loss_fn(row)
                except Exception as exc:
                    write(output/'training_error.json',dict(sample_index=row['sample_index'],error=f'{type(exc).__name__}: {exc}'))
                    raise
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                (loss/len(batch)).backward();value+=loss.item()/len(batch)
            norm=torch.nn.utils.clip_grad_norm_(parameters,config['max_grad_norm'])
            if not torch.isfinite(norm) or norm==0:raise RuntimeError('Missing/nonfinite training gradient')
            optimizer.step();step+=1;history.append(dict(step=step,epoch=epoch,loss=value,gradient_norm=norm.item()))
            write(output/'metrics.json',history)
        if on_epoch:on_epoch(epoch)


def classifier(rows,validation,config,output):
    import torch
    from .gate import load,encode,loss,save,core
    setup(config,output);state=load(config,training=True);features=[];excluded=[]
    for row in rows:
        try:
            with torch.no_grad():features.append(encode(state,row,config))
        except ValueError as exc:excluded.append(dict(sample_index=row['sample_index'],error=str(exc)))
    write(output/'excluded.json',excluded)
    if not features:raise ValueError('No classifier sequences fit configured limits')
    params=[p for p in state.model.llm.parameters() if p.requires_grad]+list(state.head.parameters())
    from .metrics import binary_metrics
    def validate(epoch):
        probs=[];labels=[];errors=[];state.model.llm.eval()
        with torch.no_grad():
            for row in validation:
                try:
                    item=encode(state,row,config);x=item['embeds'].to(state.model.device)
                    # Encoding uses current token rows; only previous reference replies are teacher-forced.
                    h=core(state.model.llm)(inputs_embeds=x.unsqueeze(0),use_cache=False,return_dict=True).last_hidden_state[0,item['positions']]
                    probs+=state.head(h.float()).softmax(-1)[:,1].tolist();labels+=row['labels']
                except ValueError as exc:errors.append(dict(sample_index=row['sample_index'],error=str(exc)))
        write(output/f'validation_epoch_{epoch}.json',dict(history='teacher_forced_reference_replies',
              metrics=binary_metrics(labels,probs,config['threshold']),excluded=errors))
        state.model.llm.train()
    state.model.llm.train()
    optimize(features,params,config,lambda row:loss(state,row),output,validate)
    save(state,output/'adapter')


def backend(rows,validation,config,output):
    import torch
    from .backend import load,merge_packets
    from .media import units
    if config['mode']!='separate_sft':raise ValueError('train-backend requires mode=separate_sft')
    setup(config,output);session=load(config,training=True)
    def example(row):
        session.start_session(row['sample'],execute=False);targets={e['second']:e for e in row['events']};target_indices=[]
        for unit in units(row,config):
            session.append_media(unit,execute=False)
            if unit.second in targets:
                session.begin_reply(execute=False)
                target_indices.append(len(session.packets))
                session.end_reply(targets[unit.second]['target'])
        packets=session.packets
        batch=merge_packets(packets);labels=torch.full_like(batch['input_ids'],-100);offset=0
        for i,packet in enumerate(packets):
            n=packet['input_ids'].shape[1]
            if i in target_indices:labels[:,offset:offset+n]=packet['input_ids']
            offset+=n
        batch['labels']=labels
        return batch
    def loss_fn(row):return session.model(**example(row),use_cache=False,return_dict=True).loss
    def validate(epoch):
        session.model.eval();values=[];errors=[]
        with torch.no_grad():
            for row in validation:
                try:values.append(loss_fn(row).item())
                except ValueError as exc:errors.append(dict(sample_index=row['sample_index'],error=str(exc)))
        write(output/f'validation_epoch_{epoch}.json',dict(loss=sum(values)/len(values) if values else None,excluded=errors))
        session.model.train()
    session.model.train();params=[p for p in session.model.parameters() if p.requires_grad]
    optimize(rows,params,config,loss_fn,output,validate)
    adapter=output/'adapter';session.model.save_pretrained(adapter)
    write(adapter/'jev_backend.json',dict(protocol=PROTOCOL,base=config['backend_model'],
          base_identity=base_identity(config['backend_model']),tokenizer=digest(session.tokenizer.get_vocab())))
