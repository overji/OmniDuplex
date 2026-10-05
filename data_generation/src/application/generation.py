"""Single generation orchestration shared by CLI and batch."""
import json
from dataclasses import asdict
from ..infrastructure.client import chat_completion
from ..infrastructure.json_io import extract_json, extract_json_array
from ..infrastructure.usage import make_usage_record
from ..domain.common import SAMPLE_SCHEMA
from ..domain.caption import validate_caption
from .registry import get_task
from .requests import build_task_request
from .video_caption import build_video_prompts
from .options import GenerationOptions, GenerationResult


def prepare_request(args: GenerationOptions):
    if args.task == "caption":
        return build_video_prompts(args.video)
    return build_task_request(args.task, args.input, args.video)


def generate(args: GenerationOptions, request=None) -> GenerationResult:
    request = request if request is not None else prepare_request(args)
    if args.task == "caption":
        bundle = request
        system_prompt, user_prompt, video_path = bundle.caption_prompt, "Annotate the attached video.", bundle.video_path
    else:
        task = get_task(args.task)
        system_prompt, user_prompt, video_path = request.system_prompt, request.user_prompt, request.video_path
    if not args.api_key:
        variable = "QWEN_API_KEY" if args.provider == "qwen" else "OPENAI_API_KEY"
        raise ValueError(f"missing API key: pass --api-key or set {variable}")
    if not args.model:
        raise ValueError("missing model: pass --model or set OPENAI_MODEL")

    raw_usage = None

    def capture_usage(value):
        nonlocal raw_usage
        raw_usage = value

    raw_response = chat_completion(
        system_prompt,
        user_prompt,
        video_path=video_path,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        retries=max(0, args.retries),
        json_array=args.task == "caption",
        provider=args.provider,
        video_url=args.video_url,
        upload_url=args.upload_url,
        usage_callback=capture_usage,
    )
    usage_record = make_usage_record(
        provider=args.provider, model=args.model, task=args.task,
        video=video_path.name, raw_usage=raw_usage,
    )
    usage_output = args.usage_output
    if usage_output:
        usage_output.parent.mkdir(parents=True, exist_ok=True)
        usage_output.write_text(json.dumps(usage_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.task == "caption":
        result = validate_caption(extract_json_array(raw_response), bundle.duration_seconds)
    else:
        result = task.validate_result(extract_json(raw_response), request.context)
        result.update({
            "schema": SAMPLE_SCHEMA,
            "task": args.task,
            "video": video_path.name,
            "generation_context": asdict(request.context),
        })
    return GenerationResult(result, usage_record)


