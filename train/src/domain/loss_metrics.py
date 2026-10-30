"""Additive SFT telemetry schema, independent of model/runtime dependencies."""

COMPONENTS = ('silence', 'speak', 'control', 'response', 'agent_instruction', 'decision',
              'onset', 'continuation_speak', 'structure')
CONTRIBUTIONS = ('decision', 'response', 'agent_instruction', 'gate', 'structure')
# Keep the original 15 slots stable for historical telemetry readers.
SUM_INDICES = (0, 1, 2, 3, 4, 5, 15, 16, 17)
COUNT_INDICES = (6, 7, 8, 9, 10, 11, 18, 19, 20)
CONTRIBUTION_INDICES = (12, 13, 14, 21, 22)
# Append-only fine-role telemetry: raw token CE/count and weighted contribution.
EVENT_COMPONENTS = ('silence_before', 'silence_after', 'first_onset', 'later_onset',
                    'continue', 'end', 'marker')
COMPONENTS += EVENT_COMPONENTS
CONTRIBUTIONS += EVENT_COMPONENTS
SUM_INDICES += tuple(range(23, 30))
COUNT_INDICES += tuple(range(30, 37))
CONTRIBUTION_INDICES += tuple(range(37, 44))
METRIC_SIZE = 44


def statistics(raw, qa_count=1, presence=None, prefix=''):
    """Convert additive statistics into means; absent classes remain null."""
    result = {}
    for index, name in enumerate(COMPONENTS):
        count = int(raw[COUNT_INDICES[index]]) if len(raw) > COUNT_INDICES[index] else 0
        result[prefix + 'loss_' + name] = raw[SUM_INDICES[index]] / count if count else None
        result[prefix + name + '_tokens'] = count
        result[prefix + name + '_qa'] = int(presence[index] if presence is not None and index < len(presence) else count > 0)
    for index, name in enumerate(CONTRIBUTIONS):
        result[prefix + 'loss_' + name + '_contribution'] = (raw[CONTRIBUTION_INDICES[index]] / max(qa_count, 1)
            if len(raw) > CONTRIBUTION_INDICES[index] else 0.)
    return result
