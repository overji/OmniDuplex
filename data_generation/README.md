# Dataset task generation

Supply the original video with every task. Events and objects are fallible
references; rendered prompts do not themselves verify audiovisual evidence.

| Task | Result file in data/test_video |
| --- | --- |
| event_alert | event_alert.json |
| target_grounding | target_grounding.json |
| snapshot_counting | snapshot_counting.json |
| semantic_condition_alert | semantic_condition_alert.json |
| event_narration | event_narration.json |
| cumulative_counting | cumulative_counting.json |
| deduplicated_counting | deduplicated_counting.json |
| state_monitoring | state_monitoring.json |
| triggered_knowledge_retrieval | triggered_knowledge_retrieval.json |
| triggered_realtime_query | triggered_realtime_query.json |
| triggered_multi_source_synthesis | triggered_multi_source_synthesis.json |
| triggered_content_recording | triggered_content_recording.json |
| triggered_message_notification | triggered_message_notification.json |
| route | routing recommendation (no result file required) |

From the repository root, render a task without calling a model:

```bash
python data_generation/generate.py --task triggered_knowledge_retrieval \
  --video data_generation/videos/test.mp4 \
  data_generation/data/test_video/events.json \
  --objects data_generation/data/test_video/objects.json --print-prompt
```

Remove --print-prompt and supply the existing model/API configuration to generate
a sample; use --output to select its JSON destination. For state_monitoring,
pass objects.json as the positional input. event_alert uses event evidence only.
The remaining tasks accept event evidence and optional --objects.

To route a video, provide both events and objects as fallible references:

```bash
python data_generation/generate.py --task route \
  --video data_generation/videos/test.mp4 \
  data_generation/data/test_video/events.json \
  --objects data_generation/data/test_video/objects.json --print-prompt
```

The former triggered_information_query, triggered_tool_action and
meeting_understanding CLI tasks have been replaced by the five workflow tasks.
Existing result JSON and rendered files are historical and are not converted.
New empty result files contain {} for pasting a complete generated JSON object.

Agent samples contain one call with video-derived extracted_parameters and
three messages without association IDs. The result arrives after a sampled
duration-based delay: at least max(5 seconds, 10% of the video length) and at
most 50% of the video length, within the video. All backend outputs are simulated.
Multi-source work runs inside that single call.

Snapshot results require trigger_times matching every response in order.
Schema checks can detect inconsistent lists, but cannot prove that the model
found all real triggers or that the scene is visually countable.

Checks:

```bash
PYTHONPATH=.:data_generation python -m unittest discover -s data_generation/tests -v
ruff check data_generation/src data_generation/tests
```
