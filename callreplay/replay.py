"""Replay: send the recorded caller turns to a new agent and see what it does now.

The caller's words are fixed (they're what real people said). The agent's side is regenerated:
the new prompt or model answers, calls tools, answers again. Tools never touch anything real:
each call gets the result the recording had for that tool (same arguments first, then the next
one in order), then the contract's [mocks], then {"ok": true}. Where every result came from is
kept, because a replay that ran on made-up tool results deserves less trust.

The replayed conversation is scored with the same contract and compared with the recording:
fixed, regressed, better, worse or same.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .agents import Agent, AgentError
from .checks import Finding
from .contract import Contract
from .evaluate import RANK, Result, evaluate
from .model import Conversation, ToolCall, Turn

SOURCES = ('recorded', 'recorded, other arguments', 'contract mock', 'default')


class ToolMock:
    def __init__(self, recorded: Conversation, contract: Contract):
        results = recorded.results_by_call()
        self.queue: dict[str, list[tuple[dict[str, Any], Any]]] = defaultdict(list)
        for _, call in recorded.calls():
            if call.id in results:
                self.queue[call.name].append((call.args, results[call.id]))
        self.mocks = contract.mocks

    def __call__(self, name: str, args: dict[str, Any]) -> tuple[Any, str]:
        q = self.queue.get(name) or []
        for i, (a, r) in enumerate(q):
            if a == args:
                q.pop(i)
                return r, 'recorded'
        if q:
            return q.pop(0)[1], 'recorded, other arguments'
        if name in self.mocks:
            return self.mocks[name], 'contract mock'
        return {'ok': True}, 'default'


@dataclass
class Replayed:
    id: str
    reference: Result
    replay: Result
    change: str                                  # fixed | regressed | better | worse | same | unscorable | error
    same_tools: bool
    reference_conv: Conversation
    replay_conv: Conversation | None
    sources: Counter = field(default_factory=Counter)
    tokens_in: int = 0
    tokens_out: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {'id': self.id, 'change': self.change, 'same_tools': self.same_tools,
                'reference': self.reference.as_dict(), 'replay': self.replay.as_dict(),
                'mock_sources': dict(self.sources), 'tokens_in': self.tokens_in, 'tokens_out': self.tokens_out,
                'reference_conversation': self.reference_conv.as_dict(),
                'replay_conversation': self.replay_conv.as_dict() if self.replay_conv else None}


def render_prompt(prompt: str, conv: Conversation) -> str:
    now = conv.started_at or datetime.now(UTC).isoformat(timespec='seconds')
    return prompt.replace('{caller}', conv.caller or 'unknown').replace('{now}', now)


def replay_one(conv: Conversation, agent: Agent, contract: Contract, reference: Result | None = None,
               max_tool_rounds: int = 4, turn_limit: int | None = None) -> Replayed:
    reference = reference or evaluate(conv, contract)
    if reference.status == 'unscorable':
        return Replayed(conv.id, reference, reference, 'unscorable', True, conv, None)

    mock = ToolMock(conv, contract)
    out = Conversation(id=conv.id, intent=conv.intent, caller=conv.caller, started_at=conv.started_at,
                       metadata={'agent': agent.name})
    prompt = render_prompt(contract.prompt, conv)
    messages: list[dict[str, Any]] = [{'role': 'system', 'content': prompt}] if prompt else []
    greeting = next((t.text for t in conv.turns[:1] if t.role == 'agent' and t.text and not t.tool_calls), '') \
        or contract.greeting
    if greeting:
        messages.append({'role': 'assistant', 'content': greeting})
        out.turns.append(Turn('agent', greeting))

    rep = Replayed(conv.id, reference, reference, 'same', True, conv, out)
    extra: list[Finding] = []
    users = [t.text for t in conv.turns if t.role == 'user' and t.text.strip()]
    if turn_limit:
        users = users[:turn_limit]
    ids = 0
    try:
        ended = False
        for said in users:
            if ended:
                break
            messages.append({'role': 'user', 'content': said})
            out.turns.append(Turn('user', said))
            for rnd in range(max_tool_rounds + 1):
                reply = agent.reply(messages, contract.tools_schema)
                rep.tokens_in += reply.tokens_in
                rep.tokens_out += reply.tokens_out
                calls = []
                for c in reply.tool_calls:
                    ids += 1
                    calls.append(ToolCall(c.name, c.args, c.id or f'call_{ids}'))
                msg: dict[str, Any] = {'role': 'assistant', 'content': reply.text or ''}
                if calls:
                    msg['tool_calls'] = [{'id': c.id, 'type': 'function',
                                          'function': {'name': c.name, 'arguments': json.dumps(c.args, ensure_ascii=False)}}
                                         for c in calls]
                messages.append(msg)
                out.turns.append(Turn('agent', reply.text or '', calls, latency_ms=reply.latency_ms))
                if not calls:
                    break
                for c in calls:
                    result, source = mock(c.name, c.args)
                    final = c.name in contract.ending_tools()
                    if not (final and source == 'default'):
                        rep.sources[source] += 1
                    if source == 'default' and not final:
                        extra.append(Finding('info', 'unmocked_tool',
                                             f'No recorded or mocked result for {c.name}; it got {{"ok": true}}.',
                                             len(out.turns) - 1))
                    messages.append({'role': 'tool', 'tool_call_id': c.id,
                                     'content': result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)})
                    out.turns.append(Turn('tool', tool_call_id=c.id, name=c.name, result=result, source=source))
                    ended = ended or final
                if ended:
                    break
                if rnd == max_tool_rounds:
                    extra.append(Finding('warning', 'tool_loop',
                                         f'Still calling tools after {max_tool_rounds} rounds in one turn.', len(out.turns) - 1))
    except Exception as e:  # noqa: BLE001 - whatever broke, the run must go on and say so
        kind = '' if isinstance(e, AgentError) else f'{type(e).__name__}: '
        rep.replay = Result(id=conv.id, status='error', score=None, intent=reference.intent,
                            reason=f'{kind}{e}'[:400])
        rep.change = 'error'
        return rep

    result = evaluate(out, contract, intent=reference.intent, context=prompt)
    result.intent_source = reference.intent_source
    if extra:
        from .evaluate import score_of, status_of
        result.findings.extend(extra)
        result.status = status_of(result.findings)
        result.score = score_of(result.findings, result.status, contract.weights)
    rep.replay = result
    rep.change = compare(reference, result)
    rep.same_tools = _unique(conv.tool_names()) == _unique(out.tool_names())
    return rep


def compare(ref: Result, new: Result) -> str:
    if ref.status == 'unscorable':
        return 'unscorable'
    if new.status == 'error':
        return 'error'
    if ref.status == 'fail' and new.status != 'fail':
        return 'fixed'
    if ref.status != 'fail' and new.status == 'fail':
        return 'regressed'
    a, b = RANK.get(ref.status, 0), RANK.get(new.status, 0)
    return 'better' if b > a else 'worse' if b < a else 'same'


def replay_all(conversations: list[Conversation], agent: Agent, contract: Contract, concurrency: int = 4,
               max_tool_rounds: int = 4, turn_limit: int | None = None,
               progress: Callable[[int, int, Replayed], None] | None = None) -> list[Replayed]:
    refs = {c.id: evaluate(c, contract) for c in conversations}
    done = 0
    results: dict[str, Replayed] = {}

    def run(conv: Conversation) -> Replayed:
        return replay_one(conv, agent, contract, refs[conv.id], max_tool_rounds, turn_limit)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for r in pool.map(run, conversations):
            done += 1
            results[r.id] = r
            if progress:
                progress(done, len(conversations), r)
    return [results[c.id] for c in conversations]


def sample(conversations: list[Conversation], contract: Contract, limit: int) -> list[Conversation]:
    """Up to `limit` conversations, spread evenly across intents, so a small replay still
    covers every kind of call."""
    if limit <= 0 or len(conversations) <= limit:
        return conversations
    buckets: dict[str, deque[Conversation]] = defaultdict(deque)
    for c in conversations:
        buckets[evaluate(c, contract).intent or '?'].append(c)
    picked: list[Conversation] = []
    while len(picked) < limit:
        for key in sorted(buckets):
            if buckets[key] and len(picked) < limit:
                picked.append(buckets[key].popleft())
        if not any(buckets.values()):
            break
    order = {c.id: i for i, c in enumerate(conversations)}
    return sorted(picked, key=lambda c: order[c.id])


def _unique(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(names))


def elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    if s < 0.1:
        return 'under 0.1 s'
    return f'{s:.1f} s' if s < 60 else f'{int(s // 60)} min {int(s % 60)} s'
