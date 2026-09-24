"""The checks. Each one looks at a conversation and adds findings; none of them calls a model.

A finding has a severity (error, warning, review, info), a stable code (listed in
docs/contracts.md) and, when it can, the index of the turn it is about.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .contract import Contract, Intent
from .grounding import known_values, mentions
from .model import Conversation

SEVERITIES = ('error', 'warning', 'review', 'info')


@dataclass
class Finding:
    severity: str
    code: str
    message: str
    turn: int | None = None

    def as_dict(self) -> dict[str, Any]:
        d = {'severity': self.severity, 'code': self.code, 'message': self.message}
        if self.turn is not None:
            d['turn'] = self.turn
        return d


def no_agent_response(conv: Conversation, c: Contract, out: list[Finding]) -> None:
    if not any(t.role == 'agent' and (t.text.strip() or t.tool_calls) for t in conv.turns):
        out.append(Finding('error', 'no_agent_response', 'The caller spoke but the agent never answered.'))


def intent_contract(conv: Conversation, intent: Intent | None, out: list[Finding]) -> None:
    if intent is None:
        return
    calls = conv.calls()
    called = {c.name for _, c in calls}
    for tool in intent.requires:
        if tool not in called:
            out.append(Finding('error', 'missing_required_tool', f'"{intent.name}" requires {tool}, never called.'))
    for group in intent.requires_any:
        if not called & set(group):
            out.append(Finding('error', 'missing_required_tool',
                               f'"{intent.name}" requires one of {", ".join(group)}; none was called.'))
    for tool in intent.prefers:
        if tool not in called:
            out.append(Finding('warning', 'missing_preferred_tool', f'"{intent.name}" usually calls {tool}; it didn\'t.'))
    for i, call in calls:
        if call.name in intent.forbids:
            out.append(Finding('error', 'forbidden_tool', f'{call.name} must not be called for "{intent.name}".', i))
    first: dict[str, int] = {}
    for n, (_, call) in enumerate(calls):
        first.setdefault(call.name, n)
    for before, after in intent.order:
        if after in first and (before not in first or first[before] > first[after]):
            turn = calls[first[after]][0]
            out.append(Finding('error', 'wrong_order', f'{after} was called before {before}.', turn))


def unexpected_writes(conv: Conversation, c: Contract, intent: Intent | None, out: list[Finding]) -> None:
    """A tool that changes something (a booking, a refund) used where the intent doesn't call for it."""
    if intent is None and c.intents:
        allowed: set[str] = set()                   # unknown intent: no write is expected
    elif intent is None:
        return                                      # no intents defined: nothing to compare with
    else:
        allowed = intent.expects()
    for i, call in conv.calls():
        rule = c.tools.get(call.name)
        if rule and rule.writes and call.name not in allowed:
            what = f'"{intent.name}"' if intent else 'a conversation with no clear intent'
            out.append(Finding('error', 'unexpected_write', f'{call.name} changes data, but {what} doesn\'t call for it.', i))


def tool_arguments(conv: Conversation, c: Contract, out: list[Finding]) -> None:
    known = c.known_tools()
    for i, call in conv.calls():
        if known and call.name not in known:
            out.append(Finding('warning', 'unknown_tool', f'{call.name} is not one of the agent\'s tools.', i))
        rule = c.tools.get(call.name)
        if not rule:
            continue
        missing = ['|'.join(alts) for alts in rule.required_args
                   if not any(call.args.get(a) not in (None, '', []) for a in alts)]
        if missing:
            out.append(Finding('error', 'missing_args', f'{call.name} called without {", ".join(missing)}.', i))


def repeated_calls(conv: Conversation, c: Contract, out: list[Finding]) -> None:
    seen: dict[str, int] = {}
    for i, call in conv.calls():
        key = call.name + json.dumps(call.args, sort_keys=True, ensure_ascii=False)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == c.max_same_call + 1:
            out.append(Finding('warning', 'repeated_call',
                               f'{call.name} called {seen[key]} times with the same arguments.', i))


def silent_turns(conv: Conversation, out: list[Finding]) -> None:
    """An agent turn with nothing in it. On a phone call, that's dead air."""
    for i, t in enumerate(conv.turns):
        if t.role == 'agent' and not t.text.strip() and not t.tool_calls:
            out.append(Finding('warning', 'silent_turn', 'The agent produced an empty turn.', i))


def grounding(conv: Conversation, c: Contract, out: list[Finding], context: str = '') -> None:
    if not c.grounding_kinds:
        return
    kinds = c.grounding_kinds
    known = set().union(*(known_values(a, kinds) for a in c.grounding_allow)) if c.grounding_allow else set()
    if context:
        known |= known_values(context, kinds)
    for i, t in enumerate(conv.turns):
        if t.role == 'user':
            known |= known_values(t.text, kinds)
        elif t.role == 'tool':
            known |= known_values(t.result, kinds)
        elif t.role == 'agent' and t.text:
            for m in mentions(t.text, kinds):
                if not m.readings & known:
                    what = 'time' if m.kind == 'time' else 'amount'
                    out.append(Finding(c.grounding_severity, 'ungrounded_value',
                                       f'Said the {what} "{m.raw}", which no tool returned and the caller didn\'t say.', i))


def end_call(conv: Conversation, c: Contract, out: list[Finding]) -> None:
    if not c.end_call_tool:
        return
    users = conv.user_texts()
    if not users or c.end_call_tool in conv.tool_names():
        return
    last = users[-1].lower()
    if any(re.search(rf'\b{re.escape(g)}\b', last) for g in c.goodbye):
        out.append(Finding('warning', 'missing_end_call',
                           f'The caller said goodbye but the agent didn\'t call {c.end_call_tool}.'))
