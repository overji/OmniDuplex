"""Text-only judge transports; imports no model or video dependencies."""
import json
import os
import urllib.request


def resolve_config(args):
    env = dict(os.environ)
    if args.env_file:
        from dotenv import dotenv_values
        env = {**dotenv_values(args.env_file, interpolate=False), **env}
    prefix = 'GEMINI' if args.provider == 'gemini' else 'OPENAI'
    key = args.api_key or env.get(prefix + '_API_KEY')
    if not key:
        raise ValueError(f'Missing {prefix}_API_KEY')
    url = args.base_url or env.get(prefix + '_BASE_URL') or (
        'https://generativelanguage.googleapis.com' if args.provider == 'gemini'
        else 'https://api.openai.com/v1')
    return key, url


def completion(config, key, system, payload):
    user = json.dumps(payload, ensure_ascii=False)
    if config['provider'] == 'gemini':
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key, http_options=types.HttpOptions(
            base_url=config['base_url'], timeout=int(config['timeout'] * 1000),
            retry_options=types.HttpRetryOptions(attempts=1)))
        try:
            result = client.models.generate_content(model=config['model'], contents=user,
                config=types.GenerateContentConfig(system_instruction=system, temperature=0,
                    response_mime_type='application/json'))
            usage = result.usage_metadata
            stats = usage.model_dump(mode='json') if usage else {}
            candidates = result.candidates or []
            valid = bool(candidates and getattr(candidates[0].finish_reason, 'name', None) == 'STOP')
            return result.text or '', stats, valid
        finally:
            close = getattr(client, 'close', None)
            if close:
                close()
    request = urllib.request.Request(config['base_url'].rstrip('/') + '/chat/completions',
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
        data=json.dumps({'model': config['model'], 'temperature': 0,
            'response_format': {'type': 'json_object'},
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        }).encode())
    with urllib.request.urlopen(request, timeout=config['timeout']) as response:
        body = json.load(response)
    choice = body['choices'][0]
    return choice['message'].get('content') or '', body.get('usage') or {}, choice.get('finish_reason') == 'stop'
