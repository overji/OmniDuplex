"""Apply the vendor's Transformers patch only inside this isolated environment."""
import shutil
import sys
from pathlib import Path
import transformers

model = Path('/remote-home/yushengliao/speech/jimodels/openmodels/omnivinci')
source = model / 'transformers/modeling_utils.py'
target = Path(transformers.__file__).parent / 'modeling_utils.py'
if not target.resolve().is_relative_to(Path(sys.prefix).resolve()):
    raise RuntimeError('Refusing to patch a shared/base Transformers installation')
if transformers.__version__ != '4.46.0':
    raise RuntimeError('OmniVinci vendor patch requires its pinned Transformers 4.46.0')
shutil.copyfile(source, target)
print('Applied OmniVinci vendor Transformers patch inside', sys.prefix)
