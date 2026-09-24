"""Reading conversations from disk, whatever platform recorded them.

`load(path)` accepts a folder of .json/.jsonl files, a single .json (one conversation or a list)
or a .jsonl file (one conversation per line). Each record is recognised by its shape:

    native      {"id", "turns": [...]}                        this project's format
    openai      {"messages": [...]}                           Chat Completions message lists
    elevenlabs  {"conversation_id", "transcript": [...]}      ElevenLabs Agents conversation details
    livekit     {"items": [...]}                              LiveKit Agents session.history.to_dict()
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .model import Conversation, ToolCall, Turn, _args

FORMATS = ('native', 'openai', 'elevenlabs', 'livekit')


def load(path: str | Path, fmt: str = 'auto') -> list[Conversation]:
    path = Path(path)
    if path.is_dir():
        sub = path / 'conversations'
        folder = sub if sub.is_dir() else path
        files = sorted(p for p in folder.iterdir() if p.suffix in ('.json', '.jsonl'))
    else:
        files = [path]
    out: list[Conversation] = []
    for f in files:
        for n, record in enumerate(_records(f)):
            try:
                conv = convert(record, fmt)
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(f'{f.name}: {e}') from None
            if not conv.id:
                conv.id = f.stem if n == 0 else f'{f.stem}-{n}'
            out.append(conv)
    return out


def _records(f: Path) -> Iterator[dict[str, Any]]:
    text = f.read_text(encoding='utf-8')
    if f.suffix == '.jsonl':
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)
        return
    data = json.loads(text)
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'role' not in data[0]:
        yield from data                      # a list of conversations
    elif isinstance(data, list):
        yield {'messages': data}             # a bare OpenAI message list
    else:
        yield data


def detect(record: dict[str, Any]) -> str:
    def rows(key: str, field: str) -> bool:
        v = record.get(key)
        return isinstance(v, list) and all(isinstance(x, dict) and field in x for x in v[:5])
    if rows('turns', 'role'):
        return 'native'
    if rows('messages', 'role'):
        return 'openai'
    if rows('transcript', 'role'):
        return 'elevenlabs'
    if rows('items', 'type'):
        return 'livekit'
    raise ValueError(f'not a conversation this tool can read (keys: {", ".join(sorted(record)[:8])}); '
                     'see docs/formats.md')


def convert(record: dict[str, Any], fmt: str = 'auto') -> Conversation:
    fmt = detect(record) if fmt == 'auto' else fmt
    return {'native': Conversation.from_dict, 'openai': from_openai,
            'elevenlabs': from_elevenlabs, 'livekit': from_livekit}[fmt](record)


# --------------------------------------------------------------------------- openai
def from_openai(record: dict[str, Any]) -> Conversation:
    turns: list[Turn] = []
    system = ''
    for m in record.get('messages') or []:
        role = m.get('role')
        if role in ('system', 'developer'):
            system = system or _text(m.get('content'))
        elif role == 'user':
            turns.append(Turn('user', _text(m.get('content'))))
        elif role == 'assistant':
            calls = [ToolCall(name=(c.get('function') or {}).get('name', ''),
                              args=_args((c.get('function') or {}).get('arguments')), id=c.get('id', ''))
                     for c in m.get('tool_calls') or []]
            turns.append(Turn('agent', _text(m.get('content')), tool_calls=calls))
        elif role == 'tool':
            turns.append(Turn('tool', tool_call_id=m.get('tool_call_id', ''), name=m.get('name', ''),
                              result=_maybe_json(m.get('content'))))
    _name_tool_turns(turns)
    meta = dict(record.get('metadata') or {})
    if system:
        meta.setdefault('system_prompt', system)
    return Conversation(id=str(record.get('id', '')), turns=turns, intent=record.get('intent'),
                        caller=record.get('caller'), started_at=record.get('started_at'), metadata=meta)


# --------------------------------------------------------------------------- elevenlabs
def from_elevenlabs(record: dict[str, Any]) -> Conversation:
    """ElevenLabs Agents: GET /v1/convai/conversations/{id}. Tool calls and their results travel
    inside the agent's transcript entries, linked by request_id."""
    turns: list[Turn] = []
    for entry in record.get('transcript') or []:
        role = 'agent' if entry.get('role') == 'agent' else 'user'
        message = entry.get('message') or ''
        calls = [ToolCall(name=c.get('tool_name', ''), args=_args(c.get('params_as_json')), id=c.get('request_id', ''))
                 for c in entry.get('tool_calls') or []]
        if message or calls:
            turns.append(Turn(role, message, tool_calls=calls if role == 'agent' else []))
        for r in entry.get('tool_results') or []:
            turns.append(Turn('tool', tool_call_id=r.get('request_id', ''), name=r.get('tool_name', ''),
                              result=_maybe_json(r.get('result_value'))))
    meta_in = record.get('metadata') or {}
    meta: dict[str, Any] = {'source': 'elevenlabs'}
    if meta_in.get('call_duration_secs') is not None:
        meta['duration_s'] = meta_in['call_duration_secs']
    successful = (record.get('analysis') or {}).get('call_successful')
    if successful:
        meta['platform_verdict'] = successful
    started = None
    if meta_in.get('start_time_unix_secs'):
        started = datetime.fromtimestamp(meta_in['start_time_unix_secs'], UTC).isoformat()
    phone = meta_in.get('phone_call') or {}
    return Conversation(id=str(record.get('conversation_id', '')), turns=turns, started_at=started,
                        caller=phone.get('external_number'), metadata=meta)


# --------------------------------------------------------------------------- livekit
def from_livekit(record: dict[str, Any]) -> Conversation:
    """LiveKit Agents: json.dumps(session.history.to_dict()). Items are messages, function
    calls and their outputs, in order. Assistant messages carry latency metrics in seconds."""
    turns: list[Turn] = []
    for item in record.get('items') or []:
        kind = item.get('type')
        if kind == 'message' and item.get('role') in ('user', 'assistant'):
            metrics = item.get('metrics') or {}
            latency = None
            if item['role'] == 'assistant' and ('llm_node_ttft' in metrics or 'tts_node_ttfb' in metrics):
                latency = 1000 * (metrics.get('llm_node_ttft', 0) + metrics.get('tts_node_ttfb', 0))
            turns.append(Turn('agent' if item['role'] == 'assistant' else 'user', _text(item.get('content')),
                              latency_ms=latency))
        elif kind == 'function_call':
            call = ToolCall(name=item.get('name', ''), args=_args(item.get('arguments')), id=item.get('call_id', ''))
            last = turns[-1] if turns else None
            if last and last.role == 'agent' and not last.text and last.tool_calls:
                last.tool_calls.append(call)          # parallel calls from the same response
            else:
                turns.append(Turn('agent', '', tool_calls=[call]))
        elif kind == 'function_call_output':
            result = _maybe_json(item.get('output'))
            if item.get('is_error'):
                result = {'error': result}
            turns.append(Turn('tool', tool_call_id=item.get('call_id', ''), name=item.get('name', ''), result=result))
    return Conversation(id=str(record.get('id', '')), turns=turns, metadata={'source': 'livekit'})


# --------------------------------------------------------------------------- helpers
def _text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):                 # content parts
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get('type') in ('text', 'input_text', 'output_text'):
                parts.append(p.get('text', ''))
        return '\n'.join(x for x in parts if x)
    return str(content)


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _name_tool_turns(turns: list[Turn]) -> None:
    """OpenAI tool messages don't repeat the tool name; take it from the call."""
    names = {c.id: c.name for t in turns for c in t.tool_calls if c.id}
    for t in turns:
        if t.role == 'tool' and not t.name:
            t.name = names.get(t.tool_call_id, '')


def save(conversations: list[Conversation], path: str | Path) -> None:
    path = Path(path)
    if path.suffix == '.jsonl':
        path.write_text(''.join(json.dumps(c.as_dict(), ensure_ascii=False) + '\n' for c in conversations),
                        encoding='utf-8')
        return
    path.mkdir(parents=True, exist_ok=True)
    for c in conversations:
        safe = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in c.id) or 'conversation'
        (path / f'{safe}.json').write_text(json.dumps(c.as_dict(), ensure_ascii=False, indent=1) + '\n',
                                           encoding='utf-8')
