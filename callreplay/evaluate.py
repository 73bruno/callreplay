"""Scoring one conversation against a contract.

Status, from worst to best:

    fail        an error-level finding: the agent did something wrong
    warn        warnings only
    review      nothing wrong, but a person should look (e.g. the intent is unclear)
    pass        clean

And two that are kept apart from the rest, because they say nothing about the agent:

    unscorable  there is nothing to judge: the caller never spoke, only noise, a test call
    error       the evaluation itself broke (in a replay: the model timed out, a 500...)
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from . import checks
from .checks import Finding
from .contract import Contract, Intent
from .model import Conversation

RANK = {'fail': 0, 'warn': 1, 'review': 2, 'pass': 3}


@dataclass
class Result:
    id: str
    status: str
    score: int | None
    intent: str | None = None
    intent_source: str | None = None           # 'label' | 'keywords' | 'tools'
    findings: list[Finding] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    user_turns: int = 0
    agent_turns: int = 0
    latency_ms: float | None = None            # median of the agent's turns, if recorded
    reason: str = ''                           # why it's unscorable / what broke

    def as_dict(self) -> dict[str, Any]:
        return {'id': self.id, 'status': self.status, 'score': self.score, 'intent': self.intent,
                'intent_source': self.intent_source, 'findings': [f.as_dict() for f in self.findings],
                'tools': self.tools, 'user_turns': self.user_turns, 'agent_turns': self.agent_turns,
                'latency_ms': self.latency_ms, 'reason': self.reason}


def evaluate(conv: Conversation, contract: Contract, intent: str | None = None, context: str = '') -> Result:
    """`intent` forces the intent (a replay uses the one found in the recording). `context` is
    text whose times and prices count as known for grounding, like the system prompt."""
    users = conv.user_texts()
    res = Result(id=conv.id, status='pass', score=100, tools=conv.tool_names(), user_turns=len(users),
                 agent_turns=len(conv.agent_texts()))
    lat = conv.latencies()
    if lat:
        res.latency_ms = round(statistics.median(lat), 1)

    reason = unscorable_reason(conv, contract)
    if reason:
        res.status, res.score, res.reason = 'unscorable', None, reason
        return res

    found = resolve_intent(conv, contract) if intent is None else (intent, 'given')
    res.intent, res.intent_source = found
    spec: Intent | None = contract.intents.get(res.intent) if res.intent else None

    out: list[Finding] = []
    checks.no_agent_response(conv, contract, out)
    checks.intent_contract(conv, spec, out)
    checks.unexpected_writes(conv, contract, spec, out)
    checks.tool_arguments(conv, contract, out)
    checks.repeated_calls(conv, contract, out)
    checks.silent_turns(conv, out)
    checks.grounding(conv, contract, out, context)
    checks.end_call(conv, contract, out)
    if contract.intents and spec is None:
        out.append(Finding('review', 'unclear_intent', 'No intent in the contract matches this conversation.'))

    res.findings = out
    res.status = status_of(out)
    res.score = score_of(out, res.status, contract.weights)
    return res


def unscorable_reason(conv: Conversation, contract: Contract) -> str:
    marked = conv.metadata.get('unscorable')
    if marked:
        return str(marked) if isinstance(marked, str) else 'marked as unscorable'
    users = conv.user_texts()
    if not users:
        return 'the caller never spoke'
    noise = set(contract.noise)
    words = [w for u in users for w in re.findall(r"[\w']+", u.lower()) if u.strip().lower() not in noise]
    if len(words) < contract.min_user_words:
        return 'too little speech to judge' if words else 'only noise from the caller'
    return ''


def resolve_intent(conv: Conversation, contract: Contract) -> tuple[str | None, str | None]:
    if not contract.intents:
        return conv.intent, 'label' if conv.intent else None
    if conv.intent and conv.intent in contract.intents:
        return conv.intent, 'label'
    text = ' '.join(conv.user_texts()[:2]).lower()          # callers say what they want early
    for name, spec in contract.intents.items():             # in file order: the first match wins
        if any(re.search(rf'\b{re.escape(k)}', text) for k in spec.keywords):
            return name, 'keywords'
    called = set(conv.tool_names())
    candidates = [s for s in contract.intents.values() if s.requires and set(s.requires) <= called]
    if candidates:
        return max(candidates, key=lambda s: len(s.requires)).name, 'tools'
    return None, None


def status_of(findings: list[Finding]) -> str:
    severities = {f.severity for f in findings}
    for sev, status in (('error', 'fail'), ('warning', 'warn'), ('review', 'review')):
        if sev in severities:
            return status
    return 'pass'


def score_of(findings: list[Finding], status: str, weights: dict[str, int]) -> int:
    """100 minus a penalty per finding, capped so the score always agrees with the status."""
    score = 100 - sum(weights.get(f.severity, 0) for f in findings)
    cap = {'fail': 59, 'warn': 84, 'review': 89}.get(status, 100)
    return max(0, min(score, cap))
