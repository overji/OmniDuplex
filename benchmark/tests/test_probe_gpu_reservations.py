"""Runtime GPU placement must preserve probe shard identities and resume paths."""
from types import SimpleNamespace
from benchmark.scripts.run_prepared_suite import probe_assignments


def test_three_active_replicas_queue_fourth_shard_then_reuse_free_gpu():
    workers=[dict(gpus=[str(g)],argv=['same-shard',str(g)]) for g in (4,5,6,7)]
    children=[None]*4
    pool=['4','5','6','7']
    assert probe_assignments(workers,children,{},pool,{'7'})==[(0,['4']),(1,['5']),(2,['6'])]
    children[:3]=[SimpleNamespace(poll=lambda:None) for _ in range(3)]
    placements={0:['4'],1:['5'],2:['6']}
    assert probe_assignments(workers,children,placements,pool,{'7'})==[]
    children[1]=SimpleNamespace(poll=lambda:0)
    assert probe_assignments(workers,children,placements,pool,{'7'})==[(3,['5'])]
    # The worker argv, rank, manifest and output have never changed.
    assert workers[3]['argv']==['same-shard','7']
    assert probe_assignments(workers,children,placements,pool,set())==[(3,['7'])]


def test_two_gpu_replica_still_gets_two_exclusive_devices():
    workers=[dict(gpus=['4','5']),dict(gpus=['6','7'])]
    assert probe_assignments(workers,[None,None],{},['4','5','6','7'],{'7'})==[(0,['4','5'])]
