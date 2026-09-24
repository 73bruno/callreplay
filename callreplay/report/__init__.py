"""Reports: one summary, several outputs.

    report.html   interactive: filters, and each conversation side by side (recording | replay)
    summary.md    short, for a pull-request comment or a CI job summary
    results.json  everything, for your own analysis
    results.csv   one row per conversation
"""
from __future__ import annotations

import csv
import json
import os
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
TEMPLATE = Path(__file__).with_name('template.html')


# --------------------------------------------------------------------------- data
def from_eval(pairs: list[tuple[Conversation, Result]], **info: Any) -> dict[str, Any]:
    items = [{'id': r.id, 'change': None, 'same_tools': None, 'reference': r.as_dict(), 'replay': None,
              'mock_sources': {}, 'reference_conversation': c.as_dict(), 'replay_conversation': None}
             for c, r in pairs]
    return _package('eval', items, info)


def from_replay(runs: list[Replayed], **info: Any) -> dict[str, Any]:
    return _package('replay', [r.as_dict() for r in runs], info)


def _package(mode: str, items: list[dict[str, Any]], info: dict[str, Any]) -> dict[str, Any]:
    return {'mode': mode, 'generated_at': datetime.now(UTC).isoformat(timespec='seconds'),
            **info, 'summary': summarize(items, mode), 'items': items}


def summarize(items: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    ref = Counter(i['reference']['status'] for i in items)
    scored = [i for i in items if i['reference']['status'] != 'unscorable']
    s: dict[str, Any] = {'total': len(items), 'scored': len(scored), 'recorded': _ordered(ref, STATUSES),
                         'recorded_pass_rate': _rate(sum(1 for i in scored if i['reference']['status'] == 'pass'), len(scored))}
    s['findings'] = _findings(items)
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
    order = {'error': 0, 'warning': 1, 'review': 2, 'info': 3}
    codes = sorted(set(rec) | set(rep), key=lambda c: (order.get(sev[c], 9), -(rec[c] + rep[c])))
    return [{'code': c, 'severity': sev[c], 'recorded': rec[c], 'replay': rep[c]} for c in codes]


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


def markdown(data: dict[str, Any]) -> str:
    s = data['summary']
    lines = []
    if data['mode'] == 'replay':
        ch = s['changes']
        lines += [f"### callreplay: {s['total']} recorded conversations replayed with `{data.get('agent', '?')}`", '',
                  f"| | |\n|---|---|\n| Pass rate | {_pct(s['recorded_pass_rate_same_set'])} → **{_pct(s['replay_pass_rate'])}** |",
                  f"| Regressed | **{ch.get('regressed', 0)}** (worse: {ch.get('worse', 0)}) |",
                  f"| Fixed | {ch.get('fixed', 0)} (better: {ch.get('better', 0)}) |",
                  f"| Unchanged | {ch.get('same', 0)} |",
                  f"| Not judged | {ch.get('unscorable', 0)} unscorable, {ch.get('error', 0)} errors |", '']
        reg = [i for i in data['items'] if i['change'] == 'regressed']
        if reg:
            lines += ['**Regressions**', '']
            lines += [f"- `{i['id']}` ({i['reference']['intent'] or '?'}): {_first(i['replay'])}" for i in reg]
            lines.append('')
        err = [i for i in data['items'] if i['change'] == 'error']
        if err:
            lines += ['**Errors** (the run, not the agent)', '']
            lines += [f"- `{i['id']}`: {i['replay']['reason'][:160]}" for i in err]
            lines.append('')
    else:
        r = s['recorded']
        lines += [f"### callreplay: {s['total']} conversations evaluated", '',
                  f"| | |\n|---|---|\n| Pass rate | **{_pct(s['recorded_pass_rate'])}** of {s['scored']} |",
                  f"| Fail · warn · review | {r.get('fail', 0)} · {r.get('warn', 0)} · {r.get('review', 0)} |",
                  f"| Unscorable | {r.get('unscorable', 0)} |", '']
        bad = [i for i in data['items'] if i['reference']['status'] == 'fail']
        if bad:
            lines += ['**Failures**', '']
            lines += [f"- `{i['id']}` ({i['reference']['intent'] or '?'}): {_first(i['reference'])}" for i in bad]
            lines.append('')
    return '\n'.join(lines) + '\n'


def _first(result: dict[str, Any]) -> str:
    worst = sorted(result['findings'], key=lambda f: ['error', 'warning', 'review', 'info'].index(f['severity']))
    return worst[0]['message'] if worst else result.get('reason', '')


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
        return f"\033[{';'.join(table[c] for c in codes)}m{text}\033[0m" if self.on and codes else text


STATUS_COLOR = {'pass': 'green', 'warn': 'yellow', 'review': 'blue', 'fail': 'red', 'unscorable': 'grey', 'error': 'magenta'}
CHANGE_COLOR = {'fixed': 'green', 'better': 'green', 'same': 'grey', 'worse': 'yellow', 'regressed': 'red',
                'error': 'magenta', 'unscorable': 'grey'}


def progress_line(n: int, total: int, r: Replayed, st: Style) -> str:
    w = len(str(total))
    a, b = r.reference.status, r.replay.status
    move = f"{st(f'{a:<10}', STATUS_COLOR[a])} → {st(f'{b:<10}', STATUS_COLOR[b])}" if r.change != 'unscorable' \
        else st(f"{'unscorable':<10}   {'':<10}", 'grey')
    if r.change in ('error', 'unscorable'):
        note = r.replay.reason
    elif r.change in ('regressed', 'worse'):
        note = _first(r.replay.as_dict())
    elif r.change in ('fixed', 'better'):
        note = 'was: ' + _first(r.reference.as_dict())
    else:
        note = ''
    return (f"  {st(f'[{n:>{w}}/{total}]', 'dim')} {r.id:<14} {move} "
            f"{st(f'{r.change:<10}', CHANGE_COLOR[r.change], 'bold' if r.change in ('regressed', 'error') else 'dim')} "
            f"{st(note[:70], 'dim')}")


def terminal(data: dict[str, Any], st: Style) -> str:
    s = data['summary']
    L: list[str] = ['']
    if data['mode'] == 'replay':
        ch = s['changes']
        L.append(st(f"  {s['total']} conversations · replayed with {data.get('agent')} · {data.get('duration', '')}", 'bold'))
        L.append('')
        L.append(f"  pass rate      {_pct(s['recorded_pass_rate_same_set'])} recorded  →  "
                 f"{st(_pct(s['replay_pass_rate']), 'bold')} replayed   ({s['replayed']} comparable)")
        L.append(f"  changed        {st(str(ch.get('fixed', 0)) + ' fixed', 'green')} · {ch.get('better', 0)} better · "
                 f"{ch.get('same', 0)} same · {ch.get('worse', 0)} worse · {st(str(ch.get('regressed', 0)) + ' regressed', 'red', 'bold')}")
        L.append(f"  not judged     {ch.get('unscorable', 0)} unscorable (nothing to judge) · "
                 f"{st(_plural(ch.get('error', 0), 'error'), 'magenta')} (the run, not the agent)")
        ms = s.get('mock_sources') or {}
        if ms:
            L.append(st('  tool results   ' + ' · '.join(f'{v} {k}' for k, v in ms.items()), 'dim'))
        reg = [i for i in data['items'] if i['change'] == 'regressed']
        if reg:
            L += ['', st('  Regressions', 'red', 'bold')]
            L += [f"  {i['id']:<14} {i['reference']['intent'] or '?':<11} {_first(i['replay'])[:90]}" for i in reg]
    else:
        r = s['recorded']
        L.append(st(f"  {s['total']} conversations evaluated", 'bold'))
        L.append('')
        L.append(f"  pass rate      {st(_pct(s['recorded_pass_rate']), 'bold')} of {s['scored']} scored")
        L.append('  status         ' + ' · '.join(st(f'{v} {k}', STATUS_COLOR[k]) for k, v in r.items()))
        bad = [i for i in data['items'] if i['reference']['status'] == 'fail']
        if bad:
            L += ['', st('  Failures', 'red', 'bold')]
            L += [f"  {i['id']:<14} {i['reference']['intent'] or '?':<11} {_first(i['reference'])[:90]}" for i in bad]
    top = [f for f in s['findings'] if f['severity'] in ('error', 'warning')][:6]
    if top:
        L += ['', st('  Findings', 'bold')]
        col = 'replay' if data['mode'] == 'replay' else 'recorded'
        for f in top:
            n = f[col] if data['mode'] == 'eval' else f"{f['recorded']:>2} → {f['replay']}"
            code = f"{f['code']:<24}"
            L.append(f"  {st(code, 'red' if f['severity'] == 'error' else 'yellow')} {n}")
    return '\n'.join(L)
