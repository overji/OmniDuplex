"""Local numeric telemetry only; importing this module never imports TensorBoard."""
import math
from pathlib import Path


def merged_values(rows):
    """One point per optimizer update; token-weighted diagnostic CE, mean objective."""
    result = {}
    for key in set().union(*(r.keys() for r in rows)):
        pairs = [(r[key], r) for r in rows if key in r and isinstance(r[key], (int, float)) and math.isfinite(r[key])]
        if not pairs:
            continue
        if key.startswith('train/loss_') and not key.endswith('_contribution'):
            token_key = 'train/' + key[len('train/loss_'):] + '_tokens'
            weighted = [(v, r.get(token_key, 0)) for v, r in pairs if r.get(token_key, 0) > 0]
            if weighted:
                result[key] = sum(v*n for v,n in weighted) / sum(n for _,n in weighted)
                continue
        values = [v for v,_ in pairs]
        if key.endswith(('_tokens', '_qa')) or key == 'train/step_wall_seconds':
            result[key] = sum(values)
        elif key in ('train/step', 'train/epoch', 'train/slot', 'train/elapsed_seconds'):
            result[key] = values[-1]
        else:
            result[key] = sum(values)/len(values)
    return result


class TensorBoardSink:
    def __init__(self, writer):
        self.writer = writer
        self.pending = []

    def emit(self, values):
        step = int(values['train/step'])
        for tag,value in values.items():
            if tag != 'train/step' and isinstance(value, (int,float)) and math.isfinite(value):
                self.writer.add_scalar(tag, value, global_step=step)

    def train(self, values, record):
        if 'optimizer_boundary' in record:
            self.pending.append(values)
            if not record['optimizer_boundary']:
                return
            self.emit(merged_values(self.pending))
            self.pending.clear()
        else:
            self.emit(values)

    def validation(self, values):
        self.emit(values)

    def close(self):
        self.writer.flush()
        self.writer.close()


def create_sink(settings, output):
    from torch.utils.tensorboard import SummaryWriter
    root = Path(settings.get('log_dir', Path(output)/'tensorboard')).expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[3]/root
    directory = root/Path(output).name
    directory.mkdir(parents=True, exist_ok=True)
    return TensorBoardSink(SummaryWriter(str(directory), flush_secs=settings.get('flush_seconds', 10))), directory
