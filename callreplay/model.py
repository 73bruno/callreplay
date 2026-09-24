"""The conversation model every other module works on.

A conversation is an ordered list of turns. Three roles:

    user   what the caller said (the transcript of their speech, or chat text)
    agent  what the agent said, and/or the tool calls it made in that turn
    tool   the result of one tool call, linked by tool_call_id

Recordings from other platforms are converted into this shape by `formats.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str = ''

    def as_dict(self) -> dict[str, Any]:
        return {'id': self.id, 'name': self.name, 'args': self.args}


@dataclass
class Turn:
    role: str                                   # 'user' | 'agent' | 'tool'
    text: str = ''
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ''                      # tool turns: which call this answers
    name: str = ''                              # tool turns: which tool
    result: Any = None                          # tool turns: what it returned
    latency_ms: float | None = None             # agent turns: time to respond, if recorded
    source: str = ''                            # tool turns in a replay: where the result came from

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {'role': self.role}
        if self.role == 'tool':
            d.update(tool_call_id=self.tool_call_id, name=self.name, result=self.result)
            if self.source:
                d['source'] = self.source
            return d
        d['text'] = self.text
        if self.tool_calls:
            d['tool_calls'] = [c.as_dict() for c in self.tool_calls]
        if self.latency_ms is not None:
            d['latency_ms'] = round(self.latency_ms, 1)
        return d


@dataclass
class Conversation:
    id: str
    turns: list[Turn] = field(default_factory=list)
    intent: str | None = None                   # a label, if the recording has one
    caller: str | None = None                   # phone number or user id, if known
    started_at: str | None = None               # ISO timestamp
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ views
    def user_texts(self) -> list[str]:
        return [t.text for t in self.turns if t.role == 'user' and t.text.strip()]

    def agent_texts(self) -> list[str]:
        return [t.text for t in self.turns if t.role == 'agent' and t.text.strip()]

    def calls(self) -> list[tuple[int, ToolCall]]:
        """Every tool call with the index of the turn that made it, in order."""
        return [(i, c) for i, t in enumerate(self.turns) if t.role == 'agent' for c in t.tool_calls]

    def tool_names(self) -> list[str]:
        return [c.name for _, c in self.calls()]

    def results_by_call(self) -> dict[str, Any]:
        return {t.tool_call_id: t.result for t in self.turns if t.role == 'tool' and t.tool_call_id}

    def latencies(self) -> list[float]:
        return [t.latency_ms for t in self.turns if t.role == 'agent' and t.latency_ms is not None]

    # ------------------------------------------------------------------ io
    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {'id': self.id}
        for k in ('intent', 'caller', 'started_at'):
            if getattr(self, k):
                d[k] = getattr(self, k)
        if self.metadata:
            d['metadata'] = self.metadata
        d['turns'] = [t.as_dict() for t in self.turns]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Conversation:
        turns = []
        for t in d.get('turns') or []:
            role = t.get('role', '')
            if role == 'assistant':
                role = 'agent'
            calls = [ToolCall(name=c.get('name', ''), args=_args(c.get('args', c.get('arguments'))), id=c.get('id', ''))
                     for c in t.get('tool_calls') or []]
            turns.append(Turn(role=role, text=t.get('text') or '', tool_calls=calls,
                              tool_call_id=t.get('tool_call_id', ''), name=t.get('name', ''),
                              result=t.get('result'), latency_ms=t.get('latency_ms'), source=t.get('source', '')))
        return cls(id=str(d.get('id', '')), turns=turns, intent=d.get('intent'), caller=d.get('caller'),
                   started_at=d.get('started_at'), metadata=d.get('metadata') or {})


def _args(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a dict or as a JSON string, sometimes broken."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {'_raw': str(raw)}
    return parsed if isinstance(parsed, dict) else {'_raw': parsed}
