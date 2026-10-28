"""Export identities and source precedence without using real video or models."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.evaluation.judge_export import export
from benchmark.evaluation.judge_package import validate
from benchmark.tests.test_probe_judge import sample, prediction
from benchmark.probe.data import sample_probes, file_hash


class ExportTests(unittest.TestCase):
    def test_sources_are_frozen_and_probe_only_exports_tp(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);qa=sample();data=root/'data.json';data.write_text(json.dumps([qa]))
            manifest,_=sample_probes(qa,0,file_hash(data),Path('/v.mp4'),
                dict(duration=30,has_audio=True,source=dict(path='/v.mp4',size=1,mtime_ns=1)))
            mp=root/'manifest.jsonl';mp.write_text(''.join(json.dumps(r)+'\n' for r in manifest))
            pp=root/'probe_predictions.jsonl'
            pp.write_text(''.join(json.dumps(r)+'\n' for r in [prediction(manifest[0],'<silent>'),prediction(manifest[1],'The cup moved.')]))
            prompt=root/'prompt.txt';prompt.write_text('Score correctness.')
            rp=root/'predictions.jsonl'
            record=dict(sample_index=0,status='complete',events=[dict(onset_seconds=11,text='The cup moved.')],
                        **{k:qa[k] for k in ('task','video','question')})
            rp.write_text(json.dumps(record)+'\n')
            registry=dict(data=str(data),data_sha256=file_hash(data),manifest=str(mp),manifest_sha256=file_hash(mp),
                public_qa=1,system_prompt=str(prompt),runs=[
                    dict(run_id='rt',model='Model',mode='realtime',variant='baseline',sources=[dict(file=str(rp),selection='all')]),
                    dict(run_id='probe',model='Model',mode='probe',variant='baseline',predictions=str(pp))])
            sources=root/'sources.json';sources.write_text(json.dumps(registry));output=root/'bundle.json'
            result=export(sources,output);validate(result)
            self.assertEqual(len(result['tasks']),3)
            self.assertEqual(sum(t['api_required'] for t in result['tasks']),2)
            self.assertEqual(result['runs'][1]['coverage']['decisions'],{'TN':1,'TP':1})
            self.assertNotIn('FUTURE EVIDENCE',json.dumps(result['tasks'][2]))
            before=hashlib.sha256(output.read_bytes()).hexdigest()
            export(sources,output);self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(),before)
            data.write_text('[]')
            with self.assertRaisesRegex(ValueError,'dataset identity changed'):export(sources,root/'other.json')


if __name__=='__main__':unittest.main()
