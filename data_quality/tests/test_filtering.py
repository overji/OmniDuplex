import copy
from data_quality.filtering import select_pass_only


def qa():
    return dict(mode='rtd',question='Update me.',responses=[dict(response='One',window_start=1,window_end=2),
                                                         dict(response='Two',window_start=3,window_end=4)])


def test_whole_qa_requires_every_response_pass_and_keeps_fields_unchanged():
    samples=[qa(),qa()];original=copy.deepcopy(samples)
    results={f'test:{i}:{j}':dict(status='complete',verdict='pass') for i in range(2) for j in range(2)}
    results['test:0:1']['verdict']='issue'
    kept,removed,mapping=select_pass_only(samples,'test',results)
    assert kept==[samples[1]] and len(kept[0]['responses'])==2
    assert removed[0]['qa']==samples[0]
    assert mapping[0]['new_sample_index'] is None and mapping[1]['new_sample_index']==0
    assert samples==original


def test_failed_uncertain_and_missing_review_remove_whole_qa():
    for record,reason in [(None,'unreviewed'),(dict(status='failed'),'failed'),
                          (dict(status='complete',verdict='uncertain'),'uncertain')]:
        results={'test:0:0':dict(status='complete',verdict='pass')}
        if record is not None:results['test:0:1']=record
        kept,removed,_=select_pass_only([qa()],'test',results)
        assert not kept
        assert removed[0]['reasons']==[dict(response_index=1,reason=reason)]


def test_agent_delegation_scope_keeps_unscored_source_replies_intact():
    sample=dict(mode='agent',responses=[dict(role='assistant',response='Delegating',window_start=1,window_end=2,
        tool_calls=[dict(function=dict(name='agent_call',arguments={}))]),dict(role='tool',response='tool result'),
        dict(role='assistant',response='completion')])
    kept,removed,_=select_pass_only([sample],'test',{'test:0:0':dict(status='complete',verdict='pass')})
    assert not removed and kept==[sample] and len(kept[0]['responses'])==3


def test_missing_window_cannot_hide_an_unreviewed_scored_response():
    sample=qa();sample['responses'][1].pop('window_end')
    kept,removed,_=select_pass_only([sample],'test',{'test:0:0':dict(status='complete',verdict='pass')})
    assert not kept and removed[0]['reasons']==[dict(response_index=1,reason='unreviewed')]
