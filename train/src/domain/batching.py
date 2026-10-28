"""Deterministic QA assignment; padding slots never repeat a training example."""
import math
import random


def epoch_slots(count, seed, epoch, world_size, rank, accumulation, data_parallel=False, *, lengths=None, bucket_size=0):
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    if data_parallel:
        if bucket_size:
            if lengths is None or len(lengths) != count or bucket_size % world_size:
                raise ValueError('Length grouping needs one length per QA and buckets divisible by world size')
            order = [index for start in range(0, count, bucket_size)
                     for index in sorted(order[start:start + bucket_size], key=lambda i: lengths[i], reverse=True)]
        total = math.ceil(count / (world_size * accumulation)) * world_size * accumulation
        order += [None] * (total - count)
        return order[rank::world_size]
    total = math.ceil(count / accumulation) * accumulation
    return (order * math.ceil(total / count))[:total]
