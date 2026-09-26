"""Reports: one summary, several outputs.

    report.html   interactive: what broke and what got fixed, filters, each conversation side by side
    summary.md    short, for a pull request; in GitHub Actions it also goes to the job summary
    results.json  everything, for your own analysis
    results.csv   one row per conversation
"""
from __future__ import annotations

import csv
import difflib
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..evaluate import Result
from ..model import Conversation
from ..replay import Replayed

CHANGES = ('regressed', 'worse', 'same', 'better', 'fixed', 'error', 'unscorable')
STATUSES = ('fail', 'warn', 'review', 'pass', 'unscorable', 'error')
SEVERITIES = ('error', 'warning', 'review', 'info')
TEMPLATE = Path(__file__).with_name('template.html')


# --------------------------------------------------------------------------- data
def from_eval(pairs: list[tuple[Conversation, Result]], **info: Any) -> dict[str, Any]:
    items = [{'id': r.id, 'change': None, 'same_tools': None, 'reference': r.as_dict(), 'replay': None,
              'mock_sources': {}, 'reference_conversation': c.as_dict(), 'replay_conversation': None,
              'tool_diff': None}
             for c, r in pairs]
    return _package('eval', items, info)


def from_replay(runs: list[Replayed], **info: Any) -> dict[str, Any]:
    items = [r.as_dict() for r in runs]
    for i in items:
        i['tool_diff'] = tool_diff(i['reference_conversation'], i['replay_conversation']) \
            if i['replay_conversation'] else None
    return _package('replay', items, info)


def _package(mode: str, items: list[dict[str, Any]], info: dict[str, Any]) -> dict[str, Any]:
    return {'mode': mode, 'generated_at': datetime.now(UTC).isoformat(timespec='seconds'),
            **info, 'summary': summarize(items, mode), 'items': items}


def summarize(items: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    ref = Counter(i['reference']['status'] for i in items)
    scored = [i for i in items if i['reference']['status'] != 'unscorable']
    s: dict[str, Any] = {'total': len(items), 'scored': len(scored), 'recorded': _ordered(ref, STATUSES),
                         'recorded_pass_rate': _rate(sum(1 for i in scored if i['reference']['status'] == 'pass'), len(scored))}
    s['findings'] = _findings(items)
    s['causes'] = causes(items, mode)
    s['unscorable'] = dict(Counter(i['reference']['reason'] for i in items if i['reference']['status'] == 'unscorable'))
    s['latency'] = {'recorded': _percentiles(items, 'reference_conversation')}
    intents: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for i in scored:
        key = i['reference']['intent'] or '(unclear)'
        intents[key]['n'] += 1
        intents[key]['recorded_pass'] += i['reference']['status'] == 'pass'
        if mode == 'replay' and i['replay']['status'] != 'error':
            intents[key]['replayed'] += 1
            intents[key]['replay_pass'] += i['replay']['status'] == 'pass'
            intents[key]['regressed'] += i['change'] == 'regressed'
            intents[key]['fixed'] += i['change'] == 'fixed'
    s['intents'] = {k: dict(v) for k, v in sorted(intents.items(), key=lambda kv: -kv[1]['n'])}
    if mode == 'replay':
        replayed = [i for i in scored if i['replay']['status'] != 'error']
        s['replay'] = _ordered(Counter(i['replay']['status'] for i in items), STATUSES)
        s['changes'] = _ordered(Counter(i['change'] for i in items), CHANGES)
        s['replay_pass_rate'] = _rate(sum(1 for i in replayed if i['replay']['status'] == 'pass'), len(replayed))
        s['recorded_pass_rate_same_set'] = _rate(sum(1 for i in replayed if i['reference']['status'] == 'pass'), len(replayed))
        s['same_tools'] = sum(1 for i in replayed if i['same_tools'])
        s['replayed'] = len(replayed)
        s['mock_sources'] = dict(sum((Counter(i['mock_sources']) for i in items), Counter()))
        s['latency']['replay'] = _percentiles(items, 'replay_conversation')
        s['tokens'] = {'in': sum(i['tokens_in'] for i in items), 'out': sum(i['tokens_out'] for i in items)}
    return s


def _findings(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rec: Counter = Counter()
    rep: Counter = Counter()
    sev: dict[str, str] = {}
    for i in items:
        for f in i['reference']['findings']:
            rec[f['code']] += 1
            sev[f['code']] = f['severity']
        for f in (i['replay'] or {}).get('findings', []) if i['replay'] is not i['reference'] else []:
            rep[f['code']] += 1
            sev[f['code']] = f['severity']
    order = {s: n for n, s in enumerate(SEVERITIES)}
    codes = sorted(set(rec) | set(rep), key=lambda c: (order.get(sev[c], 9), -(rec[c] + rep[c])))
    return [{'code': c, 'severity': sev[c], 'recorded': rec[c], 'replay': rep[c]} for c in codes]


# --------------------------------------------------------------------------- what changed, and why
def tool_sequence(conv: dict[str, Any] | None) -> list[tuple[str, int]]:
    """The tools a conversation called, in order, with repeats in a row folded: [(name, times)]."""
    out: list[list[Any]] = []
    calls = [c for t in (conv or {}).get('turns', []) if t.get('role') == 'agent' for c in t.get('tool_calls') or []]
    for c in calls:
        if out and out[-1][0] == c['name']:
            out[-1][1] += 1
        else:
            out.append([c['name'], 1])
    return [(n, k) for n, k in out]


def _label(tool: tuple[str, int] | None) -> str | None:
    return None if tool is None else tool[0] if tool[1] == 1 else f'{tool[0]} ×{tool[1]}'


def tool_diff(recorded: dict[str, Any] | None, replayed: dict[str, Any] | None) -> list[list[str | None]]:
    """The two tool sequences aligned: [[recorded, replayed], ...], None where one side has no match.
    [['check_availability', None], ['create_booking', 'create_booking']] reads "the replay skipped
    check_availability"."""
    a, b = tool_sequence(recorded), tool_sequence(replayed)
    pairs: list[list[str | None]] = []
    ops = difflib.SequenceMatcher(None, [n for n, _ in a], [n for n, _ in b], autojunk=False).get_opcodes()
    for op, i1, i2, j1, j2 in ops:
        if op == 'equal':
            pairs += [[_label(x), _label(y)] for x, y in zip(a[i1:i2], b[j1:j2], strict=True)]
        else:
            pairs += [[_label(x), None] for x in a[i1:i2]] + [[None, _label(y)] for y in b[j1:j2]]
    return pairs


def aligned(pairs: list[list[str | None]], paint: Any = None) -> tuple[str, str]:
    """Two lines in which the same tool sits in the same column, so a gap shows what's missing.
    `paint(text)` colours the tools only one side has."""
    widths = [max(len(x or ''), len(y or '')) for x, y in pairs]

    def line(side: int) -> str:
        cells = [p[side] for p in pairs]
        last = max((n for n, c in enumerate(cells) if c), default=-1)
        out = ''
        for n, (cell, w) in enumerate(zip(cells, widths, strict=True)):
            text = (cell or '').ljust(w)
            if cell and paint and pairs[n][1 - side] is None:
                text = paint(cell) + ' ' * (w - len(cell))
            out += text + ((' → ' if cell and n < last else '   ') if n < len(cells) - 1 else '')
        return out.rstrip()

    return line(0), line(1)


def worst(result: dict[str, Any] | None) -> dict[str, Any] | None:
    fs = sorted((result or {}).get('findings') or [], key=lambda f: SEVERITIES.index(f['severity']))
    return fs[0] if fs else None


def generic(finding: dict[str, Any]) -> str:
    """A finding's message without the details that differ from call to call, to group by."""
    m = finding['message']
    if finding['code'] == 'ungrounded_value':
        what = 'a time' if m.startswith('Said the time') else 'an amount'
        return f"Said {what} that no tool returned and the caller didn't say."
    if finding['code'] == 'repeated_call':
        return re.sub(r' called \d+ times', ' called again and again', m)
    return m


def causes(items: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Conversations grouped by why they changed (a replay) or didn't pass (an evaluation), with
    one example each. Thirty regressions with the same cause are one thing to fix, not thirty."""
    kinds = ('regressed', 'worse', 'fixed', 'better', 'error') if mode == 'replay' else ('fail', 'warn', 'review')
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for i in items:
        kind = i['change'] if mode == 'replay' else i['reference']['status']
        if kind not in kinds:
            continue
        before = mode == 'eval' or kind in ('fixed', 'better')
        side = i['reference'] if before else i['replay']
        f = worst(side)
        label = generic(f) if f and kind != 'error' else (side.get('reason') or 'no reason given')[:160]
        key = (kind, f['code'] if f and kind != 'error' else '', label)
        g = groups.get(key)
        if g is None:
            conv = i['reference_conversation'] if before else i['replay_conversation']
            g = groups[key] = {'kind': kind, 'code': key[1], 'label': label, 'ids': [], 'intents': [],
                               'example': _example(i, conv, f)}
        g['ids'].append(i['id'])
        if i['reference']['intent'] and i['reference']['intent'] not in g['intents']:
            g['intents'].append(i['reference']['intent'])
    order = {k: n for n, k in enumerate(kinds)}
    return sorted(groups.values(), key=lambda g: (order[g['kind']], -len(g['ids'])))


def _example(item: dict[str, Any], conv: dict[str, Any] | None, finding: dict[str, Any] | None) -> dict[str, Any]:
    turns = (item['reference_conversation'] or {}).get('turns', [])
    caller = next((t['text'] for t in turns if t['role'] == 'user' and t.get('text', '').strip()), '')
    said = ''
    if finding and finding.get('turn') is not None and conv:
        t = conv['turns'][finding['turn']] if finding['turn'] < len(conv['turns']) else {}
        said = t.get('text', '') if t.get('role') in ('agent', 'user') else ''
    return {'id': item['id'], 'caller': caller, 'said': said, 'finding': finding['message'] if finding else '',
            'tools': [_label(x) for x in tool_sequence(item['reference_conversation'])], 'diff': item.get('tool_diff')}


def _percentiles(items: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = [t['latency_ms'] for i in items for t in (i.get(key) or {}).get('turns', [])
              if t.get('role') == 'agent' and t.get('latency_ms') is not None]
    if len(values) < 2:
        return None
    q = statistics.quantiles(values, n=20)
    return {'p50': round(statistics.median(values)), 'p95': round(q[18]), 'n': len(values)}


def _ordered(counter: Counter, keys: tuple[str, ...]) -> dict[str, int]:
    return {k: counter[k] for k in keys if counter.get(k)}


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


# --------------------------------------------------------------------------- writers
def write(out: str | Path, data: dict[str, Any]) -> dict[str, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {'html': out / 'report.html', 'md': out / 'summary.md', 'json': out / 'results.json',
             'csv': out / 'results.csv'}
    paths['json'].write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    paths['md'].write_text(markdown(data), encoding='utf-8')
    paths['html'].write_text(html(data), encoding='utf-8')
    _csv(paths['csv'], data)
    return paths


def html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    return TEMPLATE.read_text(encoding='utf-8').replace('/*__DATA__*/null', payload)


def _csv(path: Path, data: dict[str, Any]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'intent', 'recorded_status', 'recorded_score', 'replay_status', 'replay_score', 'change',
                    'same_tools', 'recorded_tools', 'replay_tools', 'findings', 'reason'])
        for i in data['items']:
            ref, rep = i['reference'], i['replay'] or {}
            findings = rep.get('findings', ref['findings']) if i['replay'] else ref['findings']
            w.writerow([i['id'], ref['intent'] or '', ref['status'], _n(ref['score']), rep.get('status', ''),
                        _n(rep.get('score')), i['change'] or '', '' if i['same_tools'] is None else i['same_tools'],
                        ' '.join(ref['tools']), ' '.join(rep.get('tools', [])),
                        ' '.join(f['code'] for f in findings), rep.get('reason') or ref['reason']])


def _n(v: Any) -> str:
    return '' if v is None else str(v)


# sections of the summaries: title, kinds, whether each cause gets a worked example
REPLAY_SECTIONS = (('Regressed', ('regressed',), True), ('Worse', ('worse',), True), ('Fixed', ('fixed',), False),
                   ('Better', ('better',), False), ('Errors, the run and not the agent', ('error',), False))
EVAL_SECTIONS = (('Failures', ('fail',), True), ('Warnings', ('warn',), False), ('To review', ('review',), False))


def _example_lines(g: dict[str, Any], replay: bool, paint: Any = None) -> list[tuple[str, str]]:
    """(label, text) rows that show one call of a cause: what the caller asked, and what changed."""
    ex = g['example']
    rows = [('caller', f'"{_cut(ex["caller"], 84)}"')] if ex['caller'] else []
    if replay and ex['diff']:
        a, b = aligned(ex['diff'], paint)
        rows += [('recorded', a or '(no tools)'), ('replayed', b or '(no tools)')]
    else:
        if ex['said']:
            rows.append(('agent', f'"{_cut(ex["said"], 84)}"'))
        rows.append(('tools', ' → '.join(ex['tools']) or '(none)'))
    return rows


def markdown(data: dict[str, Any]) -> str:
    s, replay = data['summary'], data['mode'] == 'replay'
    lines = []
    if replay:
        ch = s['changes']
        lines += [f"### callreplay · {s['total']} recorded calls replayed with `{data.get('agent', '?')}`", '',
                  f"**Pass rate {_pct(s['recorded_pass_rate_same_set'])} → {_pct(s['replay_pass_rate'])}** on the "
                  f"{s['replayed']} calls both runs could judge · **{ch.get('regressed', 0)} regressed** · "
                  f"{ch.get('fixed', 0)} fixed" + ''.join(f' · {ch[k]} {k}' for k in ('worse', 'better', 'same') if ch.get(k))
                  + (f" · not judged: {ch.get('unscorable', 0)} unscorable, {_plural(ch.get('error', 0), 'error')}"
                     if ch.get('unscorable') or ch.get('error') else ''), '']
    else:
        r = s['recorded']
        lines += [f"### callreplay · {s['total']} recorded calls evaluated", '',
                  f"**Pass rate {_pct(s['recorded_pass_rate'])}** of the {s['scored']} calls that could be judged · "
                  f"{r.get('fail', 0)} fail · {r.get('warn', 0)} warn · {r.get('review', 0)} review · "
                  f"{r.get('unscorable', 0)} unscorable", '']
    for title, kinds, detailed in REPLAY_SECTIONS if replay else EVAL_SECTIONS:
        groups = [g for g in s['causes'] if g['kind'] in kinds]
        if not groups:
            continue
        lines += [f"#### {title} ({sum(len(g['ids']) for g in groups)})", '']
        for g in groups:
            ids = _ids(g['ids'], 8, fmt='`{}`')
            if detailed:
                rows = _example_lines(g, replay)
                w = max(len(k) for k, _ in rows)
                lines += [f"**{g['label']}** {ids}", '', '```text']
                lines += [f"{g['example']['id'] if n == 0 else '':<{len(g['example']['id'])}}  {k:<{w}}  {v}".rstrip()
                          for n, (k, v) in enumerate(rows)]
                lines += ['```', '']
                continue
            else:
                was = 'was: ' if g['kind'] in ('fixed', 'better') else ''
                lines.append(f"- {was}{g['label']} {ids}")
        if lines[-1]:
            lines.append('')
    return '\n'.join(lines).rstrip('\n') + '\n'


def _ids(ids: list[str], n: int, fmt: str = '{}') -> str:
    shown = ' '.join(fmt.format(i) for i in ids[:n])
    return shown + (f' +{len(ids) - n} more' if len(ids) > n else '')


def _cut(text: str, n: int) -> str:
    text = ' '.join(text.split())
    return text if len(text) <= n else text[:n - 1].rstrip() + '…'


def _plural(n: int, word: str) -> str:
    return f'{n} {word}' + ('' if n == 1 else 's')


def _pct(v: float | None) -> str:
    return '–' if v is None else f'{v * 100:.0f}%'


# --------------------------------------------------------------------------- terminal
class Style:
    def __init__(self, stream: Any = sys.stdout):
        self.on = bool(os.environ.get('FORCE_COLOR')) or (hasattr(stream, 'isatty') and stream.isatty()
                                                          and not os.environ.get('NO_COLOR'))

    def __call__(self, text: str, *codes: str) -> str:
        table = {'bold': '1', 'dim': '2', 'red': '31', 'green': '32', 'yellow': '33', 'blue': '34', 'magenta': '35',
                 'cyan': '36', 'grey': '90'}
        codes = tuple(c for c in codes if c)
        return f"\033[{';'.join(table[c] for c in codes)}m{text}\033[0m" if self.on and codes else text


STATUS_COLOR = {'pass': 'green', 'warn': 'yellow', 'review': 'blue', 'fail': 'red', 'unscorable': 'grey', 'error': 'magenta'}
CHANGE_COLOR = {'fixed': 'green', 'better': 'green', 'same': 'grey', 'worse': 'yellow', 'regressed': 'red',
                'error': 'magenta', 'unscorable': 'grey'}
MARK = {'regressed': '✕', 'worse': '↓', 'fixed': '✓', 'better': '↑', 'error': '!', 'same': '·', 'unscorable': '·',
        'fail': '✕', 'warn': '!', 'review': '?'}


def change_line(r: Replayed, st: Style, width: int = 14) -> str:
    """One replayed conversation whose result changed, as it finishes."""
    a, b = r.reference.status, r.replay.status
    if r.change == 'error':
        note = r.replay.reason
    elif r.change in ('fixed', 'better'):
        note = 'was: ' + _first(r.reference.as_dict())
    else:
        note = _first(r.replay.as_dict())
    color = CHANGE_COLOR[r.change]
    return (f"  {st(MARK[r.change], color, 'bold')} {r.id:<{width}} {st(f'{a:>6}', STATUS_COLOR[a])} → "
            f"{st(f'{b:<6}', STATUS_COLOR[b])} {st(f'{r.change:<10}', color, 'bold')} {st(_cut(note, 76), 'dim')}")


class Progress:
    """A bar that redraws in place at the bottom; lines written through it scroll above it."""

    def __init__(self, total: int, st: Style, stream: Any = sys.stdout):
        self.total, self.st, self.stream, self.done, self.counts = total, st, stream, 0, Counter()

    def update(self, r: Replayed, line: str | None = None) -> None:
        self.done += 1
        self.counts[r.change] += 1
        self.stream.write('\r\033[2K' + (line + '\n' if line else '') + self._bar())
        self.stream.flush()

    def close(self) -> None:
        self.stream.write('\r\033[2K' + self._bar() + '\n')
        self.stream.flush()

    def _bar(self, width: int = 30) -> str:
        n = round(width * self.done / self.total) if self.total else width
        counts = ' · '.join(self.st(f'{self.counts[k]} {k}', CHANGE_COLOR[k]) for k in ('regressed', 'fixed', 'error')
                            if self.counts[k])
        return (f"  {self.st('━' * n, 'blue')}{self.st('─' * (width - n), 'grey')}  {self.done}/{self.total}"
                + (f'   {counts}' if counts else ''))


def progress_line(n: int, total: int, r: Replayed, st: Style) -> str:
    """Kept for callers of 0.1: the line for one replayed conversation."""
    return change_line(r, st)


def terminal(data: dict[str, Any], st: Style) -> str:
    s, replay = data['summary'], data['mode'] == 'replay'
    L: list[str] = ['']
    if replay:
        ch = s['changes']
        L.append(st(f"  {s['total']} recorded calls replayed with {data.get('agent')} · {data.get('duration', '')}", 'bold'))
        L.append('')
        L.append(f"  pass rate    {_pct(s['recorded_pass_rate_same_set'])} → {st(_pct(s['replay_pass_rate']), 'bold')}"
                 f"   on the {s['replayed']} calls both runs could judge")
        L.append(f"  changes      {st(str(ch.get('regressed', 0)) + ' regressed', 'red', 'bold')} · "
                 f"{st(str(ch.get('fixed', 0)) + ' fixed', 'green')}"
                 + ''.join(f" · {st(str(ch[k]) + ' ' + k, CHANGE_COLOR[k] if k == 'worse' else '')}"
                           for k in ('worse', 'better', 'same') if ch.get(k)))
        L.append(f"  not judged   {ch.get('unscorable', 0)} unscorable (nothing to judge) · "
                 f"{st(_plural(ch.get('error', 0), 'error'), 'magenta')} (the run, not the agent)")
        ms = s.get('mock_sources') or {}
        if ms:
            L.append(st('  tool results ' + ' · '.join(f'{v} {k}' for k, v in ms.items()), 'dim'))
    else:
        r = s['recorded']
        L.append(st(f"  {s['total']} recorded calls evaluated · {data.get('duration', '')}", 'bold'))
        L.append('')
        L.append(f"  pass rate    {st(_pct(s['recorded_pass_rate']), 'bold')} of the {s['scored']} calls that could be judged")
        L.append('  status       ' + ' · '.join(st(f'{v} {k}', STATUS_COLOR[k]) for k, v in r.items()))
    for title, kinds, detailed in REPLAY_SECTIONS if replay else EVAL_SECTIONS:
        groups = [g for g in s['causes'] if g['kind'] in kinds]
        if not groups:
            continue
        n = sum(len(g['ids']) for g in groups)
        color = CHANGE_COLOR.get(kinds[0]) or STATUS_COLOR.get(kinds[0], 'grey')
        L += ['', st(f"  {title} · {_plural(n, 'call')}", color, 'bold')]
        for g in groups:
            was = 'was: ' if g['kind'] in ('fixed', 'better') else ''
            L.append(f"  {st(MARK[g['kind']], color, 'bold')} {was}{g['label']}  {st(_ids(g['ids'], 4), 'dim')}")
            if detailed:
                rows = _example_lines(g, replay, paint=lambda t, c=color: st(t, c, 'bold'))
                w = max(len(k) for k, _ in rows)
                pad = len(g['example']['id'])
                for k, (key, value) in enumerate(rows):
                    first = st(g['example']['id'], 'dim') if k == 0 else ' ' * pad
                    L.append(f"      {first}  {st(f'{key:<{w}}', 'dim')}  {value}")
    return '\n'.join(L)


def _first(result: dict[str, Any]) -> str:
    f = worst(result)
    return f['message'] if f else result.get('reason', '')
