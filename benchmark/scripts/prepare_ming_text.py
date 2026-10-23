"""Make the released Ming speech-output import lazy for text-only probing."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'cache/official/Ming'
p = root / 'modeling_bailingmm.py'
s = p.read_text()
before = hashlib.sha256(s.encode()).hexdigest()
line = 'from modeling_bailing_talker import BailingTalkerForConditionalGeneration\n'
marker = '        if self.config.talker_config:\n'
if line in s and marker in s and '# benchmark: lazy speech-output import' not in s:
    s = s.replace(line, '# benchmark: lazy speech-output import\n', 1)
    s = s.replace(marker, marker + '            from modeling_bailing_talker import BailingTalkerForConditionalGeneration\n', 1)
    p.write_text(s)
    (root / 'text_probe_patch.json').write_text(json.dumps({'file': p.name, 'original_sha256': before,
        'patched_sha256': hashlib.sha256(s.encode()).hexdigest(),
        'change': 'Import speech-output implementation only when talker_config is enabled; probe disables Talker.'}, indent=2))
print('Ming text-only import preparation complete')
