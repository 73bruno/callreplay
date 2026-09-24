"""Contracts: what a correct conversation looks like, per intent, in a TOML file.

    [intents.cancel]
    keywords = ["cancel", "can't make it"]       # to recognise the intent when there's no label
    requires = ["cancel_booking"]                # every one must be called
    requires_any = [["find_booking", "list_bookings"]]   # at least one of each group
    prefers = []                                 # missing -> warning instead of failure
    forbids = ["create_booking"]
    order = [["find_booking", "cancel_booking"]] # the first must come before the second

    [tools.create_booking]
    required_args = ["name", "date", "time", "phone|caller_id"]   # a|b: either will do
    writes = true            # changes something: only allowed in intents that expect it

See docs/contracts.md for every option.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_WEIGHTS = {'error': 25, 'warning': 10, 'review': 5, 'info': 0}

_KNOWN = {
    'agent': {'prompt', 'tools', 'greeting'},
    'end_call': {'tool', 'goodbye', 'also_ends'},
    'grounding': {'kinds', 'allow', 'severity'},
    'loops': {'max_same_call'},
    'unscorable': {'min_user_words', 'noise'},
    'score': set(DEFAULT_WEIGHTS),
    'intent': {'description', 'keywords', 'requires', 'requires_any', 'prefers', 'forbids', 'order'},
    'tool': {'required_args', 'writes'},
}
_SECTIONS = {'agent', 'intents', 'tools', 'end_call', 'grounding', 'loops', 'unscorable', 'score', 'mocks'}


@dataclass
class Intent:
    name: str
    description: str = ''
    keywords: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    requires_any: list[list[str]] = field(default_factory=list)
    prefers: list[str] = field(default_factory=list)
    forbids: list[str] = field(default_factory=list)
    order: list[tuple[str, str]] = field(default_factory=list)

    def expects(self) -> set[str]:
        """Tools this intent is meant to use."""
        return {*self.requires, *self.prefers, *(t for g in self.requires_any for t in g)}


@dataclass
class ToolRule:
    required_args: list[list[str]] = field(default_factory=list)   # each entry: alternatives
    writes: bool = False


@dataclass
class Contract:
    intents: dict[str, Intent] = field(default_factory=dict)
    tools: dict[str, ToolRule] = field(default_factory=dict)
    end_call_tool: str | None = None
    also_ends: list[str] = field(default_factory=list)     # e.g. transfer_call: the agent's part is over
    goodbye: list[str] = field(default_factory=lambda: ['bye', 'goodbye', 'thanks', 'thank you', "that's all"])
    grounding_kinds: list[str] = field(default_factory=list)
    grounding_allow: list[str] = field(default_factory=list)
    grounding_severity: str = 'error'
    max_same_call: int = 2
    min_user_words: int = 2
    noise: list[str] = field(default_factory=lambda: ['[inaudible]', '[silence]', '...', '(noise)'])
    weights: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    # what a replay needs; paths are relative to the contract file
    prompt: str = ''
    tools_schema: list[dict[str, Any]] = field(default_factory=list)
    greeting: str = ''
    mocks: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def ending_tools(self) -> set[str]:
        return ({self.end_call_tool} if self.end_call_tool else set()) | set(self.also_ends)

    def known_tools(self) -> set[str]:
        names = {t['function']['name'] for t in self.tools_schema if 'function' in t}
        return names | set(self.tools) | ({self.end_call_tool} if self.end_call_tool else set())


def load(path: str | Path | None) -> Contract:
    """A contract from a TOML file, or from the contract.toml inside a folder. No file: the
    generic checks only."""
    if path is None:
        return Contract()
    path = Path(path)
    if path.is_dir():
        path = path / 'contract.toml'
        if not path.exists():
            return Contract()
    with path.open('rb') as f:
        raw = tomllib.load(f)
    return parse(raw, base=path.parent, path=path)


def parse(raw: dict[str, Any], base: Path | None = None, path: Path | None = None) -> Contract:
    base = base or Path('.')
    unknown = set(raw) - _SECTIONS
    if unknown:
        raise ContractError(f'unknown section(s): {", ".join(sorted(unknown))}')
    c = Contract(path=path)

    agent = _section(raw, 'agent')
    if agent.get('prompt'):
        p = base / agent['prompt']
        c.prompt = p.read_text(encoding='utf-8') if p.exists() else agent['prompt']
    if agent.get('tools'):
        c.tools_schema = _tool_schemas(json.loads((base / agent['tools']).read_text(encoding='utf-8')))
    c.greeting = agent.get('greeting', '')

    for name, spec in (raw.get('intents') or {}).items():
        _check_keys(spec, 'intent', f'intents.{name}')
        order = []
        for pair in spec.get('order', []):
            if len(pair) != 2:
                raise ContractError(f'intents.{name}.order: each entry is [first, then], got {pair}')
            order.append((pair[0], pair[1]))
        c.intents[name] = Intent(name=name, description=spec.get('description', ''),
                                 keywords=[k.lower() for k in spec.get('keywords', [])],
                                 requires=list(spec.get('requires', [])),
                                 requires_any=[list(g) for g in spec.get('requires_any', [])],
                                 prefers=list(spec.get('prefers', [])), forbids=list(spec.get('forbids', [])),
                                 order=order)

    for name, spec in (raw.get('tools') or {}).items():
        _check_keys(spec, 'tool', f'tools.{name}')
        c.tools[name] = ToolRule(required_args=[a.split('|') for a in spec.get('required_args', [])],
                                 writes=bool(spec.get('writes', False)))

    end = _section(raw, 'end_call')
    c.end_call_tool = end.get('tool') or None
    c.also_ends = list(end.get('also_ends', []))
    if 'goodbye' in end:
        c.goodbye = [g.lower() for g in end['goodbye']]

    g = _section(raw, 'grounding')
    c.grounding_kinds = list(g.get('kinds', []))
    bad = set(c.grounding_kinds) - {'time', 'money'}
    if bad:
        raise ContractError(f'grounding.kinds: unknown kind(s) {", ".join(sorted(bad))} (time, money)')
    c.grounding_allow = [str(a) for a in g.get('allow', [])]
    c.grounding_severity = g.get('severity', 'error')

    c.max_same_call = int(_section(raw, 'loops').get('max_same_call', c.max_same_call))
    u = _section(raw, 'unscorable')
    c.min_user_words = int(u.get('min_user_words', c.min_user_words))
    if 'noise' in u:
        c.noise = [n.lower() for n in u['noise']]
    c.weights.update({k: int(v) for k, v in _section(raw, 'score').items()})
    c.mocks = dict(raw.get('mocks') or {})
    return c


class ContractError(ValueError):
    pass


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    s = raw.get(name) or {}
    _check_keys(s, name, name)
    return s


def _check_keys(spec: dict[str, Any], kind: str, where: str) -> None:
    unknown = set(spec) - _KNOWN[kind]
    if unknown:
        raise ContractError(f'{where}: unknown key(s) {", ".join(sorted(unknown))} '
                            f'(expected: {", ".join(sorted(_KNOWN[kind]))})')


def _tool_schemas(data: Any) -> list[dict[str, Any]]:
    """Accept OpenAI tools ([{type, function}]) or bare function definitions."""
    items = data.get('tools', data) if isinstance(data, dict) else data
    out = []
    for t in items:
        out.append(t if 'function' in t else {'type': 'function', 'function': t})
    return out
