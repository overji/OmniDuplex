"""Real local event files and resumable JSONL backfill; no GPU or W&B requests."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from train.src.infrastructure.tensorboard import merged_values


def test_update_averaging_weights_ce_and_ignores_missing_labels():
    rows=[{'train/step':0,'train/loss':2.,'train/loss_response':1.,'train/response_tokens':2},
          {'train/step':1,'train/loss':4.,'train/loss_response':3.,'train/response_tokens':6}]
    r=merged_values(rows)
    assert r['train/loss']==3 and r['train/loss_response']==2.5 and r['train/response_tokens']==8


def test_real_writer_records_only_numeric_metrics_on_rank_zero(tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from train.src.infrastructure.tracking import start_tracking
    config=dict(tensorboard=dict(enabled=True,log_dir=str(tmp_path/'events')),wandb=dict(mode='disabled'))
    out=tmp_path/'run';out.mkdir()
    t=start_tracking(config,out,0,4)
    t.log_train(dict(step=0,loss=2,optimizer_boundary=False,response_text='PRIVATE'))
    t.log_train(dict(step=1,loss=4,optimizer_boundary=True,reward_tp=2))
    t.log_validation(1,1.5,3,1)
    t.finish({})
    a=EventAccumulator(str(tmp_path/'events/run'));a.Reload()
    assert a.Scalars('train/loss')[0].value==3 and a.Scalars('train/loss')[0].step==1
    assert a.Scalars('validation/loss')[0].value==1.5
    assert not any('PRIVATE' in k for k in a.Tags()['scalars'])
    peer=start_tracking(config,out,1,4);assert peer.tensorboard is None


def test_jsonl_bridge_resumes_without_replaying_old_events(tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    root=Path(__file__).parents[2];run=tmp_path/'run';run.mkdir();logs=tmp_path/'logs'
    rows=[dict(step=(i+1)//4,loss=i+1,optimizer_boundary=(i+1)%4==0) for i in range(8)]
    source=run/'metrics.rank0.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows[:4])+json.dumps(rows[4]))
    command=[sys.executable,str(root/'train/scripts/tensorboard_from_jsonl.py'),'--run-dir',str(run),'--log-dir',str(logs)]
    subprocess.run(command,check=True,capture_output=True)
    with source.open('a') as f:f.write('\n'+''.join(json.dumps(r)+'\n' for r in rows[5:]))
    subprocess.run(command,check=True,capture_output=True)
    a=EventAccumulator(str(logs/'run'));a.Reload();events=a.Scalars('train/loss')
    assert [(e.step,e.value) for e in events]==[(1,2.5),(2,6.5)]
