import os
import json
import time
import urllib.request
import urllib.error

RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 5
TIMEOUT = 120


def _post_json(url, headers, payload):
    body = json.dumps(payload).encode()
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(url, data=body, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read()), resp.status
        except urllib.error.HTTPError as e:
            status = e.code
            detail = e.read().decode(errors='replace')[:500]
            if status in RETRY_STATUS and attempt < MAX_RETRIES - 1:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise RuntimeError(f"http {status}: {detail}")
        except urllib.error.URLError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise
    raise RuntimeError("exhausted retries")


def _api_key(name):
    key = os.environ.get(name)
    if not key:
        raise RuntimeError(f"{name} not set in environment")
    return key


def _anthropic_api(system, user, model, max_tokens, temperature, top_p):
    headers = {
        'x-api-key': _api_key('ANTHROPIC_API_KEY'),
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
    }
    payload = {
        'model': model,
        'max_tokens': max_tokens,
        'system': system,
        'messages': [{'role': 'user', 'content': user}],
    }
    if temperature is not None:
        payload['temperature'] = temperature
    if top_p is not None:
        payload['top_p'] = top_p
    data, _ = _post_json('https://api.anthropic.com/v1/messages', headers, payload)
    text = ''.join(block.get('text', '') for block in data.get('content', []))
    usage = data.get('usage', {})
    return text, {'input_tokens': usage.get('input_tokens'), 'output_tokens': usage.get('output_tokens')}


def _env(name):
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"{name} not set; required for the aws backend")
    return val


RETRY_AWS_CODES = {'ThrottlingException', 'ServiceUnavailableException', 'InternalServerException',
                   'ModelTimeoutException', 'ModelNotReadyException'}
RETRY_AWS_MESSAGE = 'countries, regions, or territories'


def _aws_is_retryable(exc):
    from botocore.exceptions import ClientError, EndpointConnectionError, ConnectionClosedError
    if isinstance(exc, (EndpointConnectionError, ConnectionClosedError)):
        return True
    if isinstance(exc, ClientError):
        err = exc.response.get('Error', {})
        return err.get('Code') in RETRY_AWS_CODES or RETRY_AWS_MESSAGE in err.get('Message', '')
    return False


def _anthropic_aws(system, user, model, max_tokens, temperature, top_p):
    import boto3

    client = boto3.client(_env('CW_LLM_AWS_SERVICE'), region_name=_env('CW_LLM_REGION'))
    payload = {
        'anthropic_version': _env('CW_LLM_AWS_PROTOCOL_VERSION'),
        'max_tokens': max_tokens,
        'system': system,
        'messages': [{'role': 'user', 'content': user}],
    }
    if temperature is not None:
        payload['temperature'] = temperature
    if top_p is not None:
        payload['top_p'] = top_p
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.invoke_model(modelId=model, body=json.dumps(payload))
            break
        except Exception as e:
            if _aws_is_retryable(e) and attempt < MAX_RETRIES - 1:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise
    data = json.loads(resp['body'].read())
    text = ''.join(block.get('text', '') for block in data.get('content', []))
    usage = data.get('usage', {})
    return text, {'input_tokens': usage.get('input_tokens'), 'output_tokens': usage.get('output_tokens')}


def complete(system, user, *, model, max_tokens=1024, temperature=1.0, top_p=1.0, backend=None):
    backend = backend or os.environ.get('CW_LLM_BACKEND', 'api')
    if backend == 'api':
        return _anthropic_api(system, user, model, max_tokens, temperature, top_p)
    if backend == 'aws':
        return _anthropic_aws(system, user, model, max_tokens, temperature, top_p)
    raise ValueError(f"unknown CW_LLM_BACKEND '{backend}' (expected 'api' or 'aws')")


def openai_complete(system, user, *, model, max_tokens=256, temperature=0.0, json_object=True):
    headers = {
        'authorization': f"Bearer {_api_key('OPENAI_API_KEY')}",
        'content-type': 'application/json',
    }
    payload = {
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
    }
    if json_object:
        payload['response_format'] = {'type': 'json_object'}
    data, _ = _post_json('https://api.openai.com/v1/chat/completions', headers, payload)
    text = data['choices'][0]['message']['content']
    usage = data.get('usage', {})
    return text, {'input_tokens': usage.get('prompt_tokens'), 'output_tokens': usage.get('completion_tokens')}
