"""The agent under test: anything that takes a message list and returns text and/or tool calls.

    --agent openai:gpt-4.1-mini        OpenAI
    --agent gemini:gemini-2.5-flash    Google, through its OpenAI-compatible endpoint
    --agent groq:llama-3.3-70b-versatile, cerebras:…, mistral:…, together:…, openrouter:…
    --agent ollama:qwen2.5             a local model
    --agent anthropic:claude-sonnet-5  Anthropic Messages API
    --agent compat:MODEL --base-url https://…/v1 --api-key-env MY_KEY    any other endpoint
    --agent python:my_module:make_agent  your own code (see docs/replay.md)
    --agent demo                       the scripted agent the demo uses; no model at all

Messages are always in OpenAI Chat Completions shape; adapters translate if they need to.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .model import ToolCall, _args

PRESETS = {
    'openai': ('https://api.openai.com/v1', 'OPENAI_API_KEY'),
    'gemini': ('https://generativelanguage.googleapis.com/v1beta/openai', 'GEMINI_API_KEY'),
    'groq': ('https://api.groq.com/openai/v1', 'GROQ_API_KEY'),
    'cerebras': ('https://api.cerebras.ai/v1', 'CEREBRAS_API_KEY'),
    'mistral': ('https://api.mistral.ai/v1', 'MISTRAL_API_KEY'),
    'together': ('https://api.together.xyz/v1', 'TOGETHER_API_KEY'),
    'openrouter': ('https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY'),
    'ollama': ('http://localhost:11434/v1', ''),
}
_FUNCTION_TAG = re.compile(r'<function=([\w.-]+)>\s*(\{.*?\})\s*</function>', re.S)


@dataclass
class AgentReply:
    text: str = ''
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0


class Agent(Protocol):
    name: str

    def reply(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AgentReply: ...


class AgentError(RuntimeError):
    """The model could not be reached or answered with something unusable. A replay that hits
    this is reported as 'error', never as a failure of the agent."""


def from_spec(spec: str, base_url: str | None = None, api_key_env: str | None = None,
              temperature: float = 0.2, max_tokens: int = 800, timeout: float = 60) -> Agent:
    kind, _, rest = spec.partition(':')
    if kind == 'demo':
        from .demo_agent import DemoAgent
        return DemoAgent(rest or 'v2')
    if kind == 'python':
        return _python_agent(rest)
    if kind == 'anthropic':
        return AnthropicAgent(rest, api_key_env or 'ANTHROPIC_API_KEY', temperature, max_tokens, timeout)
    if kind == 'compat' or base_url:
        if not base_url:
            raise ValueError('compat:MODEL needs --base-url')
        return OpenAICompatAgent(rest if kind == 'compat' else rest or kind, base_url, api_key_env or '',
                                 temperature, max_tokens, timeout)
    if kind not in PRESETS:
        raise ValueError(f'unknown agent "{spec}". Use one of: {", ".join([*PRESETS, "anthropic", "compat", "python", "demo"])}')
    if not rest:
        raise ValueError(f'{kind}: needs a model, e.g. {kind}:MODEL')
    url, env = PRESETS[kind]
    return OpenAICompatAgent(rest, url, api_key_env or env, temperature, max_tokens, timeout, label=kind)


# --------------------------------------------------------------------------- http
def _post(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST with retries on rate limits and server errors (waits 2, 5 and 15 s)."""
    body = json.dumps(payload).encode()
    last = ''
    for wait in (2, 5, 15, None):
        req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json', **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = f'HTTP {e.code}: {e.read()[:300].decode("utf-8", "replace")}'
            if e.code not in (408, 409, 429, 500, 502, 503, 504, 529):
                raise AgentError(last) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = f'{type(e).__name__}: {e}'
        if wait is None:
            break
        time.sleep(wait)
    raise AgentError(last)


class OpenAICompatAgent:
    def __init__(self, model: str, base_url: str, key_env: str, temperature: float, max_tokens: int,
                 timeout: float, label: str = 'compat'):
        self.model, self.base_url, self.temperature, self.max_tokens, self.timeout = \
            model, base_url.rstrip('/'), temperature, max_tokens, timeout
        self.key = os.environ.get(key_env, '') if key_env else ''
        if key_env and not self.key:
            raise ValueError(f'{label}: set {key_env}')
        self.name = f'{label}:{model}'

    def reply(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AgentReply:
        payload: dict[str, Any] = {'model': self.model, 'messages': messages, 'temperature': self.temperature,
                                   'max_tokens': self.max_tokens}
        if tools:
            payload.update(tools=tools, tool_choice='auto')
        headers = {'Authorization': f'Bearer {self.key}'} if self.key else {}
        t0 = time.perf_counter()
        data = _post(f'{self.base_url}/chat/completions', headers, payload, self.timeout)
        ms = (time.perf_counter() - t0) * 1000
        try:
            msg = data['choices'][0]['message']
        except (KeyError, IndexError, TypeError):
            raise AgentError(f'no choices in the response: {str(data)[:300]}') from None
        text = msg.get('content') or ''
        calls = [ToolCall(name=c['function']['name'], args=_args(c['function'].get('arguments')), id=c.get('id', ''))
                 for c in msg.get('tool_calls') or []]
        if not calls and '<function=' in text:          # some open models write calls as text
            calls = [ToolCall(name=m.group(1), args=_args(m.group(2)), id=f'call_{n}')
                     for n, m in enumerate(_FUNCTION_TAG.finditer(text))]
            text = _FUNCTION_TAG.sub('', text).strip()
        usage = data.get('usage') or {}
        return AgentReply(text, calls, ms, usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))


class AnthropicAgent:
    """Anthropic Messages API over plain HTTP, translating from and to the OpenAI shape."""

    def __init__(self, model: str, key_env: str, temperature: float, max_tokens: int, timeout: float):
        if not model:
            raise ValueError('anthropic: needs a model, e.g. anthropic:claude-sonnet-5')
        self.model, self.temperature, self.max_tokens, self.timeout = model, temperature, max_tokens, timeout
        self.key = os.environ.get(key_env, '')
        if not self.key:
            raise ValueError(f'anthropic: set {key_env}')
        self.name = f'anthropic:{model}'

    def reply(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AgentReply:
        system, converted = to_anthropic(messages)
        payload: dict[str, Any] = {'model': self.model, 'max_tokens': self.max_tokens, 'messages': converted,
                                   'temperature': self.temperature}
        if system:
            payload['system'] = system
        if tools:
            payload['tools'] = [{'name': t['function']['name'], 'description': t['function'].get('description', ''),
                                 'input_schema': t['function'].get('parameters') or {'type': 'object', 'properties': {}}}
                                for t in tools]
        headers = {'x-api-key': self.key, 'anthropic-version': '2023-06-01'}
        t0 = time.perf_counter()
        data = _post('https://api.anthropic.com/v1/messages', headers, payload, self.timeout)
        ms = (time.perf_counter() - t0) * 1000
        text = '\n'.join(b.get('text', '') for b in data.get('content', []) if b.get('type') == 'text').strip()
        calls = [ToolCall(name=b['name'], args=b.get('input') or {}, id=b.get('id', ''))
                 for b in data.get('content', []) if b.get('type') == 'tool_use']
        usage = data.get('usage') or {}
        return AgentReply(text, calls, ms, usage.get('input_tokens', 0), usage.get('output_tokens', 0))


def to_anthropic(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """OpenAI messages -> (system, Anthropic messages). Tool results become tool_result blocks in
    a user message, and consecutive same-role messages are merged, as the API requires."""
    system = '\n\n'.join(m['content'] for m in messages if m['role'] == 'system' and m.get('content'))
    out: list[dict[str, Any]] = []
    for m in messages:
        if m['role'] == 'system':
            continue
        if m['role'] == 'tool':
            role, blocks = 'user', [{'type': 'tool_result', 'tool_use_id': m['tool_call_id'],
                                     'content': m.get('content') or ''}]
        elif m['role'] == 'assistant':
            role, blocks = 'assistant', []
            if m.get('content'):
                blocks.append({'type': 'text', 'text': m['content']})
            for c in m.get('tool_calls') or []:
                blocks.append({'type': 'tool_use', 'id': c['id'], 'name': c['function']['name'],
                               'input': _args(c['function'].get('arguments'))})
        else:
            role, blocks = 'user', [{'type': 'text', 'text': m.get('content') or ''}]
        if not blocks:
            continue
        if out and out[-1]['role'] == role:
            out[-1]['content'].extend(blocks)
        else:
            out.append({'role': role, 'content': blocks})
    if out and out[0]['role'] == 'assistant':           # the greeting: the API wants a user turn first
        out.insert(0, {'role': 'user', 'content': [{'type': 'text', 'text': '(call connected)'}]})
    return system, out


def _python_agent(ref: str) -> Agent:
    """python:package.module:attr, where attr is an Agent object, a class or factory returning
    one, or a plain function reply(messages, tools) -> AgentReply."""
    module, _, attr = ref.partition(':')
    if not module or not attr:
        raise ValueError('python agent: use python:module:attribute')
    sys.path.insert(0, os.getcwd())
    obj = getattr(importlib.import_module(module), attr)
    if hasattr(obj, 'reply'):
        return obj
    if isinstance(obj, type):
        return obj()
    made = obj() if callable(obj) and obj.__code__.co_argcount == 0 else None
    if made is not None and hasattr(made, 'reply'):
        return made

    class _Fn:
        name = f'python:{ref}'

        def reply(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AgentReply:
            return obj(messages, tools)
    return _Fn()
