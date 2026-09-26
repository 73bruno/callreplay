"""A first contract, drafted from what the agent did in the recorded calls: `callreplay init`.

It writes down:

- the tools the agent used, and the arguments it passed every single time (required_args)
- the tool that hangs up and the ones that transfer the call (end_call)
- the tools that change something, going by their names: create_, cancel_, update_... (writes)
- one intent per kind of call, named after the tool the call was for: the tools those calls
  always used (requires), in the order they used them (order), and the words callers used for
  them and hardly ever for other calls (keywords)
- the times and prices the agent says in several calls without a tool returning them, which are
  usually opening hours or fixed prices from the prompt (grounding.allow)

A draft describes what the agent did, not what it should do: read it, rename the intents and
loosen or tighten the rules before trusting it. Calls where the agent went wrong will show up as
failures against it, which is the point.
"""
from __future__ import annotations

import json
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import Contract
from .evaluate import unscorable_reason
from .grounding import known_values, mentions
from .model import Conversation

PROMPT_FILES = ('prompt.md', 'prompt.txt', 'system_prompt.md', 'system_prompt.txt', 'system-prompt.md')
NO_TOOLS = 'no_tools'
KINDS = ('time', 'money')
_WRITE = re.compile(r'(?:^|_)(create|book|cancel|update|delete|remove|add|send|schedule|reschedule|modify|insert|'
                    r'submit|place|charge|pay|refund|edit|save|register|reserve|change|move|set|write)(?:_|$)')
_END = re.compile(r'^(?:end|hang|terminate|finish|close|stop)_?(?:the_)?(?:call|conversation|session|chat)$|^hang_?up$')
_TRANSFER = re.compile(r'transfer|forward|handoff|hand_off|escalat|human|operator')
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
STOP = set("""a about after again all also am an and any are as at be because been but by can could did do does for from
get go going got had has have hello hey hi how i i'd i'll i'm if in is it it's just know like me my need no not now of oh
ok okay on one or please right so some that that's the them then there this to um uh want was we well were what when
where which who will with would yes you your yeah thanks thank sure could can't don't there's here anything something
nothing everything fine great good morning afternoon evening today tomorrow name calling""".split())


@dataclass
class Intent:
    name: str
    calls: list[str]
    keywords: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    prefers: list[tuple[str, int]] = field(default_factory=list)      # (tool, in how many calls)
    order: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Draft:
    toml: str
    tools_json: list[dict[str, Any]] | None       # a tools.json to write, when the folder has none
    total: int
    usable: int
    tools: dict[str, int]                         # tool -> how many calls
    end_tool: str | None
    transfers: list[str]
    writes: list[str]
    intents: list[Intent]
    allowed: dict[str, int]                        # value said without a tool -> in how many calls
    prompt_file: str | None                        # relative to the contract
    tools_file: str                                # relative to the contract


def draft(conversations: list[Conversation], folder: Path, out_dir: Path | None = None, source: str = '') -> Draft:
    """`folder` holds the calls (and maybe a prompt and a tools.json); the contract goes to `out_dir`,
    and the paths in it are relative to there."""
    out_dir = out_dir or folder
    usable = [c for c in conversations if not unscorable_reason(c, Contract())]
    found = next((folder / n for n in PROMPT_FILES if (folder / n).is_file()), None)
    prompt = found.read_text(encoding='utf-8') if found else ''
    prompt_file = _relative(found, out_dir) if found else None
    tools_file = next((_relative(p, out_dir) for p in (out_dir / 'tools.json', folder / 'tools.json') if p.is_file()), None)

    args: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in usable:
        for _, call in c.calls():
            args[call.name].append(call.args)
    names = sorted(args, key=lambda n: (-len(args[n]), n))
    end_tool = next((n for n in names if _END.match(_snake(n))), None)
    transfers = [n for n in names if n != end_tool and _TRANSFER.search(_snake(n))]
    ending = {n for n in (end_tool, *transfers) if n}
    names.sort(key=lambda n: n in ending)                                  # the ones doing the work first
    writes = [n for n in names if n not in ending and _WRITE.search(_snake(n))]

    groups: dict[str, list[Conversation]] = defaultdict(list)
    for c in usable:
        groups[_anchor(c.tool_names(), writes, transfers, ending)].append(c)
    intents = [_intent(name, cs, ending) for name, cs in groups.items()]
    intents.sort(key=lambda i: (len(i.calls), -len(i.requires), i.name))    # specific (small) first
    _keywords(intents, usable)

    allowed = _said_without_a_tool(usable, prompt)
    d = Draft(toml='', tools_json=None if tools_file else [_schema(n, args[n]) for n in names],
              total=len(conversations), usable=len(usable), tools={n: len(args[n]) for n in names},
              end_tool=end_tool, transfers=transfers, writes=writes, intents=intents,
              allowed={v: k for v, k in allowed.items() if k >= 3}, prompt_file=prompt_file,
              tools_file=tools_file or 'tools.json')
    d.toml = _render(d, args, source, mentions_seen=bool(allowed) or _any_mention(usable))
    return d


# --------------------------------------------------------------------------- inference
def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _snake(name: str) -> str:
    return re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name).lower().replace('-', '_')


def _anchor(names: list[str], writes: list[str], transfers: list[str], ending: set[str]) -> str:
    """What the call was for: the last tool that changed something, else a transfer, else the last
    tool used. A call with no tools at all goes with the others like it."""
    core = [n for n in names if n not in ending]
    changed = [n for n in core if n in writes]
    if changed:
        return changed[-1]
    moved = [n for n in names if n in transfers]
    if moved:
        return moved[0]
    return core[-1] if core else NO_TOOLS


def _intent(name: str, calls: list[Conversation], ending: set[str]) -> Intent:
    n = len(calls)
    seqs = [c.tool_names() for c in calls]
    freq = Counter(t for s in seqs for t in set(s) if t not in ending or t == name)
    requires = [t for t, k in freq.items() if k / n >= 0.8]
    prefers = sorted(((t, k) for t, k in freq.items() if 0.5 <= k / n < 0.8), key=lambda x: -x[1])
    first = {t: statistics.median(s.index(t) for s in seqs if t in s) for t in requires}
    requires.sort(key=lambda t: first[t])
    order = [(a, b) for a, b in zip(requires, requires[1:], strict=False) if _before(seqs, a, b) >= 0.9]
    return Intent(name=name, calls=[c.id for c in calls], requires=requires, prefers=prefers, order=order)


def _before(seqs: list[list[str]], a: str, b: str) -> float:
    both = [s for s in seqs if a in s and b in s]
    return sum(s.index(a) < s.index(b) for s in both) / len(both) if both else 0.0


def _keywords(intents: list[Intent], usable: list[Conversation]) -> None:
    """For each intent, a few words its callers say early and other callers hardly ever do. Words
    are matched as prefixes, like the contract does: "book" also finds "booking"."""
    words: dict[str, set[str]] = {}
    for c in usable:
        said = c.user_texts()
        opening = next((t for t in said if len(t.split()) >= 3), said[0] if said else '')   # "hello?" says nothing
        names = {w.lower() for w in re.findall(r'(?<=[a-z,] )[A-Z][a-z]+', opening)}      # Priya, Friday: not an intent
        tokens = _WORD.findall(opening.lower())
        grams = {w for w in tokens if len(w) >= 3 and w not in STOP and w not in names}
        grams |= {_stem(w) for w in grams}
        grams |= {f'{a} {b}' for a, b in zip(tokens, tokens[1:], strict=False)
                  if a not in STOP and b not in STOP and not {a, b} & names}
        words[c.id] = grams
    matched: dict[str, set[str]] = defaultdict(set)
    for cid, grams in words.items():
        for g in grams:
            matched[g].add(cid)
    for intent in intents:
        mine, left, picked = set(intent.calls), set(intent.calls), []
        support = 1 if intent.name == NO_TOOLS else 2        # with no tools to go by, words are all there is
        while left and len(picked) < (8 if intent.name == NO_TOOLS else 5):
            best = max(((len(ids & left), -len(k), k) for k, ids in matched.items()
                        if len(ids & mine) >= support and len(ids & mine) / len(ids) >= 0.85 and ids & left
                        and not any(k.startswith(p) or p.startswith(k) for p in picked)), default=None)
            if best is None:
                break
            picked.append(best[2])
            left -= matched[best[2]]
        intent.keywords = sorted(picked, key=lambda k: -len(matched[k] & mine))


def _stem(word: str) -> str:
    for suffix in ('ing', 'ed', 's'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def _said_without_a_tool(usable: list[Conversation], prompt: str) -> dict[str, int]:
    """Times and prices the agent said that nothing in the call gave it, and in how many calls."""
    counts: Counter = Counter()
    base = known_values(prompt, KINDS) if prompt else set()
    for c in usable:
        known, said = set(base), set()
        for t in c.turns:
            if t.role == 'user':
                known |= known_values(t.text, KINDS)
            elif t.role == 'tool':
                known |= known_values(t.result, KINDS)
            elif t.role == 'agent' and t.text:
                said |= {m.raw.lower() for m in mentions(t.text, KINDS) if not m.readings & known}
        counts.update(said)
    return dict(counts.most_common())


def _any_mention(usable: list[Conversation]) -> bool:
    return any(mentions(t, KINDS) for c in usable for t in c.agent_texts())


def _schema(name: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """An OpenAI tool definition guessed from the arguments seen, for replays to offer the model."""
    types: dict[str, set[str]] = defaultdict(set)
    for a in calls:
        for k, v in a.items():
            if k != '_raw' and v is not None:
                types[k].add(_json_type(v))
    props: dict[str, Any] = {}
    for k, ts in types.items():
        props[k] = {'type': ts.pop() if len(ts) == 1 else 'string'}
        if props[k]['type'] == 'array':
            props[k]['items'] = {}
    params: dict[str, Any] = {'type': 'object', 'properties': props}
    required = [k for k in props if all(a.get(k) not in (None, '', []) for a in calls)]
    if required:
        params['required'] = required
    return {'type': 'function', 'function': {'name': name, 'parameters': params}}


def _json_type(v: Any) -> str:
    if isinstance(v, bool):
        return 'boolean'
    if isinstance(v, int):
        return 'integer'
    if isinstance(v, float):
        return 'number'
    if isinstance(v, list):
        return 'array'
    if isinstance(v, dict):
        return 'object'
    return 'string'


def _required_args(calls: list[dict[str, Any]]) -> list[str]:
    keys = Counter(k for a in calls for k, v in a.items() if k != '_raw' and v not in (None, '', []))
    return [k for k, n in keys.items() if n == len(calls)]


# --------------------------------------------------------------------------- the file
def _q(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)          # a JSON string is a valid TOML basic string


def _list(values: list[str]) -> str:
    return '[' + ', '.join(_q(v) for v in values) + ']'


def _c(code: str, comment: str = '', col: int = 36) -> str:
    """A line of TOML with its comment in a column, so the file reads like a table."""
    return code if not comment else f"{code}{' ' * max(2, col - len(code))}# {comment}"


def _render(d: Draft, args: dict[str, list[dict[str, Any]]], source: str, mentions_seen: bool) -> str:
    L = [f'# Drafted by `callreplay init` from {d.total} recorded calls{f" in {source}" if source else ""}'
         f' ({d.usable} with enough speech to use).',
         '# It describes what the agent did, not what it should do: rename the intents, check their',
         '# keywords, and loosen or tighten the rules. Every option is explained in',
         '# https://github.com/73bruno/callreplay/blob/main/docs/contracts.md', '', '[agent]']
    if d.prompt_file:
        L.append(_c(f'prompt = {_q(d.prompt_file)}', 'the system prompt a replay sends'))
    else:
        L.append(_c('# prompt = "prompt.md"', 'your system prompt: a replay sends it to the model'))
    guessed = '' if d.tools_json is None else ', guessed from the calls: swap in your real ones'
    L += [_c(f'tools = {_q(d.tools_file)}', f'the tools a replay offers the model{guessed}'), '']

    L += ['# One intent per kind of call, named after the tool the call was for. They are tried in this',
          "# order: the first with a keyword in the caller's first two turns wins; with no keyword, the",
          '# one whose required tools were all called.']
    for i in d.intents:
        example = ', '.join(i.calls[:3]) + (', ...' if len(i.calls) > 3 else '')
        L += ['', _c(f'[intents.{i.name}]', f'{len(i.calls)} calls: {example}')]
        if i.name == NO_TOOLS:
            L.append('description = "Calls the agent handled without any tool"')
        L.append(_c(f'keywords = {_list(i.keywords)}', '' if i.keywords else 'no word set these calls apart'))
        if i.requires:
            L.append(f'requires = {_list(i.requires)}')
        if i.prefers:
            L.append(_c(f'prefers = {_list([t for t, _ in i.prefers])}',
                        'in ' + ', '.join(f'{k} of {len(i.calls)} calls' for _, k in i.prefers)))
        if i.order:
            L.append('order = [' + ', '.join(f'[{_q(a)}, {_q(b)}]' for a, b in i.order) + ']')

    L += ['', '# Arguments the agent passed every time it called the tool. `writes = true`: it changes',
          '# something, so it is only allowed in the intents that expect it.']
    for name in d.tools:
        if name in (d.end_tool, *d.transfers):
            continue
        req = _required_args(args[name])
        if not req and name not in d.writes:
            continue
        L += ['', _c(f'[tools.{name}]', f'{len(args[name])} calls')]
        if req:
            L.append(f'required_args = {_list(req)}')
        if name in d.writes:
            L.append('writes = true')

    if d.end_tool or d.transfers:
        L += ['', '[end_call]']
        if d.end_tool:
            L.append(_c(f'tool = {_q(d.end_tool)}', 'not called after a goodbye: a warning'))
            L.append(_c(f'goodbye = {_list(Contract().goodbye)}', 'add the ways your callers say it'))
        if d.transfers:
            L.append(_c(f'also_ends = {_list(d.transfers)}', "after these, the agent's part is over"))

    if mentions_seen:
        L += ['', '# Every time and price the agent says must come from a tool, the caller, the prompt or this list.',
              '[grounding]', 'kinds = ["time", "money"]']
        if d.allowed:
            counts = ', '.join(f'"{v}" in {k} calls' for v, k in d.allowed.items())
            L += ['# Said in several calls without a tool returning them. From your prompt (opening hours,',
                  f'# fixed prices)? Keep them. If not, remove them and they will be flagged: {counts}.',
                  f'allow = {_list(list(d.allowed))}']
    return '\n'.join(L) + '\n'
