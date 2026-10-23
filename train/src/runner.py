"""Single-GPU pilot training, real backward profiling, and PEFT adapter checkpoints."""
import hashlib
import json
import random
import time
from pathlib import Path


def run(args, config):
    import torch
    from transformers import set_seed
    from train.src.model import load_model, encode_record, add_lora, loss_for_record
    from benchmark.data.files import append_jsonl, write_json
    if args.output.exists():
        raise ValueError('Use a fresh output directory; pilot does not resume optimizer state')
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    from train.src.agent import validate_prepared
    validate_prepared(manifest, config)
    # Identity-sensitive preparation settings must remain fixed.
    for key in ('seed', 'max_prefix_seconds', 'videos_dir'):
        if manifest['config'][key] != config[key]:
            raise ValueError(f'Preparation config mismatch: {key}')
    records = json.loads((args.prepared / 'train.json').read_text())
    validation = json.loads((args.prepared / 'validation.json').read_text())
    if args.profile:
        records = ([r for r in records if r['sample_index'] == args.sample_index]
                   if args.sample_index is not None else sorted(records, key=lambda r: r['onset_seconds'])[:1])
        validation = []
    if not records:
        raise ValueError('Empty training set')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for native MiniCPM pilot')
    set_seed(config['seed'])
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.output.mkdir(parents=True)
    identity = {'config': config, 'prepared_manifest': manifest, 'profile': args.profile,
                'device': str(device), 'max_steps': args.max_steps, 'sample_index': args.sample_index,
                'prompt_hashes': {key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
                                  for key in ('system_prompt', 'agent_prompt')}}
    write_json(args.output / 'run_config.json', identity)
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    model, duplex = load_model(config, device)
    loaded_bytes = torch.cuda.memory_allocated(device)
    print(f'Model loaded: {loaded_bytes / 2**30:.2f} GiB', flush=True)
    encoded, val_encoded = [], []
    for split, rows, output in [('train', records, encoded), ('validation', validation, val_encoded)]:
        for record in rows:
            item = encode_record(record, model, duplex, config)
            output.append(item)
            info = {k: item[k] for k in ('sample_index', 'seconds', 'target_chunks')}
            info.update(split=split, tokens=len(item['labels']), supervised=int((item['labels'] != -100).sum()))
            append_jsonl(args.output / 'sequences.jsonl', info)
            print(json.dumps(info), flush=True)
    feature_peak = torch.cuda.max_memory_allocated(device)
    llm = add_lora(model, config)
    trainable = [p for p in llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    step = 0
    grad_norm = None
    final_loss = None
    rng = random.Random(config['seed'])
    for epoch in range(1 if args.profile else config['epochs']):
        llm.train()
        order = list(encoded); rng.shuffle(order)
        for start in range(0, len(order), config['gradient_accumulation_steps']):
            batch = order[start:start + config['gradient_accumulation_steps']]
            value = 0.0
            for record in batch:
                loss = loss_for_record(llm, record, config, device)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                value += loss.item() / len(batch)
                (loss / len(batch)).backward()
            if hasattr(model, 'agent_metadata'):
                base = llm.get_base_model()
                row_grads = {name: (module.row.grad.norm().item() if module.row.grad is not None else 0.)
                    for name, module in [('input_row', base.get_input_embeddings()),
                                         ('output_row', base.get_output_embeddings())]}
                append_jsonl(args.output / 'token_gradients.jsonl', {'step': step + 1, **row_grads})
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config['max_grad_norm'])
            if not torch.isfinite(grad_norm) or grad_norm == 0:
                raise RuntimeError('Missing or nonfinite LoRA gradient')
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            step += 1; final_loss = value
            row = {'step': step, 'epoch': epoch, 'loss': value, 'gradient_norm': grad_norm.item(),
                   'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30}
            append_jsonl(args.output / 'metrics.jsonl', row); print(json.dumps(row), flush=True)
            if args.profile or (args.max_steps and step >= args.max_steps):
                break
        if val_encoded:
            llm.eval()
            with torch.no_grad():
                val_loss = sum(loss_for_record(llm, item, config, device).item() for item in val_encoded) / len(val_encoded)
            append_jsonl(args.output / 'metrics.jsonl', {'epoch': epoch, 'validation_loss': val_loss})
        if args.profile or (args.max_steps and step >= args.max_steps):
            break
    if hasattr(model, 'agent_metadata'):
        from train.src.token_rows import save_adapter
        save_adapter(llm, duplex.tokenizer, model.agent_metadata, args.output / 'adapter')
    else:
        llm.save_pretrained(args.output / 'adapter')
    report = {'profile': args.profile, 'optimizer_steps': step, 'last_loss': final_loss,
        'trainable_parameters': sum(p.numel() for p in trainable), 'gpu': torch.cuda.get_device_name(device),
        'loaded_gib': loaded_bytes / 2**30, 'feature_peak_gib': feature_peak / 2**30,
        'training_peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
        'training_peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30,
        'elapsed_seconds': time.time() - started}
    write_json(args.output / 'memory_report.json', report)
    print(json.dumps(report, indent=2), flush=True)
