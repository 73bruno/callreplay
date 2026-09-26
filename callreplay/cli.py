"""callreplay: evaluate recorded agent conversations, and replay them against a new agent.

    callreplay demo                         the bundled example, replayed against a new prompt, no API key
    callreplay init PATH                    draft a contract.toml from what your agent did in the calls
    callreplay eval PATH                    check recordings against the contract
    callreplay replay PATH --agent SPEC     re-run the caller's turns through another agent
    callreplay convert PATH OUT             turn ElevenLabs / LiveKit / OpenAI logs into this format

PATH is a folder of conversations (or a .json / .jsonl file). If the folder has a contract.toml,
it is used unless --contract says otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import webbrowser
from pathlib import Path

from . import __version__, formats, report
from . import contract as contracts
from . import draft as drafts
from .agents import from_spec
from .evaluate import evaluate
from .replay import elapsed, replay_all, sample

EXAMPLE = Path(__file__).parent / 'examples' / 'dental'


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog='callreplay', description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', action='version', version=f'callreplay {__version__}')
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument('path', help='folder of conversations, or a .json / .jsonl file')
        sp.add_argument('--contract', help='contract TOML (default: contract.toml next to the conversations)')
        sp.add_argument('--format', default='auto', choices=['auto', *formats.FORMATS], help='input format (default: auto)')
        sp.add_argument('--out', default='callreplay-report', help='where to write the reports (default: ./callreplay-report)')
        sp.add_argument('--open', action='store_true', help='open the HTML report when done')
        sp.add_argument('--quiet', action='store_true', help='only the summary')

    d = sub.add_parser('demo', help='replay the bundled example (a fictional dental clinic) against a scripted agent')
    d.add_argument('--out', default='callreplay-demo')
    d.add_argument('--no-open', action='store_true')
    d.add_argument('--eval', action='store_true', help='only evaluate the recordings, no replay')

    i = sub.add_parser('init', help='draft a contract.toml (and a tools.json) from recorded conversations')
    i.add_argument('path', help='folder of conversations, or a .json / .jsonl file')
    i.add_argument('--format', default='auto', choices=['auto', *formats.FORMATS], help='input format (default: auto)')
    i.add_argument('--out', help='where to write the contract (default: contract.toml next to the conversations)')
    i.add_argument('--force', action='store_true', help='overwrite an existing contract')

    e = sub.add_parser('eval', help='check recorded conversations against a contract')
    common(e)
    e.add_argument('--fail-on', default='', help='exit 1 if any conversation has this status: fail, warn')

    r = sub.add_parser('replay', help='replay recorded conversations against an agent')
    common(r)
    r.add_argument('--agent', required=True, help='openai:MODEL, gemini:MODEL, anthropic:MODEL, ollama:MODEL, '
                                                  'compat:MODEL, python:module:attr, demo (see docs/replay.md)')
    r.add_argument('--base-url', help='for compat:MODEL, the /v1 URL of any OpenAI-compatible server')
    r.add_argument('--api-key-env', help='environment variable holding the key (overrides the preset)')
    r.add_argument('--temperature', type=float, default=0.2)
    r.add_argument('--limit', type=int, default=0, help='replay at most N conversations, spread across intents')
    r.add_argument('--only', default='', help='only conversations whose recording has these statuses, e.g. fail,warn')
    r.add_argument('--concurrency', type=int, default=4)
    r.add_argument('--max-tool-rounds', type=int, default=4, help='tool calls per turn before giving up (default 4)')
    r.add_argument('--turn-limit', type=int, default=0, help='replay only the first N caller turns')
    r.add_argument('--fail-on', default='', help='exit 1 on: regressions, worse, fail, errors (comma-separated)')

    c = sub.add_parser('convert', help='convert ElevenLabs / LiveKit / OpenAI logs into callreplay conversations')
    c.add_argument('path')
    c.add_argument('out', help='a folder (one .json per conversation) or a .jsonl file')
    c.add_argument('--format', default='auto', choices=['auto', *formats.FORMATS])

    args = p.parse_args(argv)
    try:
        return {'demo': cmd_demo, 'init': cmd_init, 'eval': cmd_eval, 'replay': cmd_replay,
                'convert': cmd_convert}[args.cmd](args)
    except (contracts.ContractError, ValueError, FileNotFoundError) as exc:
        print(f'callreplay: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _conversations(path: Path, fmt: str) -> list:
    if not path.exists():
        raise FileNotFoundError(f'{path} does not exist')
    convs = formats.load(path, fmt)
    if not convs:
        raise ValueError(f'no conversations found in {path}')
    return convs


def _load(args: argparse.Namespace) -> tuple[list, contracts.Contract]:
    path = Path(args.path)
    convs = _conversations(path, args.format)
    contract = contracts.load(args.contract or (path if path.is_dir() else path.parent))
    return convs, contract


def cmd_eval(args: argparse.Namespace) -> int:
    st = report.Style()
    t0 = time.perf_counter()
    convs, contract = _load(args)
    pairs = [(c, evaluate(c, contract, context=contract.prompt)) for c in convs]
    data = report.from_eval(pairs, source=_shown(args.path), contract=_name(contract), duration=elapsed(t0))
    code = _finish(args, data, st, _exit_eval(args.fail_on, data))
    if contract.path is None:
        print(st(f'  No contract.toml here, so only the generic checks ran. Draft one from these calls:\n'
                 f'  callreplay init {args.path}\n', 'yellow'))
    return code


def cmd_replay(args: argparse.Namespace, agent=None, intro: str = '') -> int:
    st = report.Style()
    convs, contract = _load(args)
    if not contract.prompt and agent is None and not args.agent.startswith(('demo', 'python')):
        print(st('  warning: the contract has no [agent] prompt; the agent will get no system prompt', 'yellow'))
    if args.only:
        wanted = {s.strip() for s in args.only.split(',')}
        convs = [c for c in convs if evaluate(c, contract).status in wanted]
    convs = sample(convs, contract, args.limit)
    agent = agent or from_spec(args.agent, args.base_url, args.api_key_env, args.temperature)
    print(st(f'\n  callreplay {__version__} · replaying {len(convs)} recorded calls with {agent.name}', 'bold'))
    intro = intro or '  the caller says the same words; the agent answers again; tools answer from the recording'
    print('\n'.join(st(line, 'dim') for line in intro.split('\n')))      # one style per line: logs reset colours
    print()
    t0 = time.perf_counter()
    bar = report.Progress(len(convs), st) if not args.quiet and sys.stdout.isatty() else None
    width = min(max((len(c.id) for c in convs), default=8), 40)

    def progress(n: int, total: int, r) -> None:
        if args.quiet:
            return
        line = report.change_line(r, st, width) if r.change in ('regressed', 'worse', 'fixed', 'error') else None
        if bar:
            bar.update(r, line)
        elif line:
            print(line, flush=True)

    runs = replay_all(convs, agent, contract, concurrency=args.concurrency, max_tool_rounds=args.max_tool_rounds,
                      turn_limit=args.turn_limit or None, progress=progress)
    if bar:
        bar.close()
    data = report.from_replay(runs, source=_shown(args.path), contract=_name(contract), agent=agent.name,
                              duration=elapsed(t0))
    return _finish(args, data, st, _exit_replay(args.fail_on, data))


def cmd_demo(args: argparse.Namespace) -> int:
    folder = EXAMPLE if EXAMPLE.exists() else Path(__file__).resolve().parents[1] / 'examples' / 'dental'
    ns = argparse.Namespace(path=str(folder), contract=None, format='auto', out=args.out, open=not args.no_open,
                            quiet=False, fail_on='', agent='demo', limit=0, only='', concurrency=4,
                            max_tool_rounds=4, turn_limit=0)
    if args.eval:
        return cmd_eval(ns)
    from .demo_agent import DemoAgent
    intro = ('  42 recorded calls to a fictional dental clinic, replayed against a new version of its prompt\n'
             '  (a scripted stand-in for the model, so no API key: point --agent at a real one for real runs)')
    return cmd_replay(ns, DemoAgent('v2', delay=0.06), intro=intro)      # a little pace, like a real model


def cmd_init(args: argparse.Namespace) -> int:
    st = report.Style()
    path = Path(args.path)
    convs = _conversations(path, args.format)
    folder = path if path.is_dir() else path.parent
    out = Path(args.out) if args.out else folder / 'contract.toml'
    if out.exists() and not args.force:
        raise ValueError(f'{out} already exists: --force overwrites it, --out writes somewhere else')
    d = drafts.draft(convs, folder, out.parent, source=_shown(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(d.toml, encoding='utf-8')
    written = [(out, '')]
    if d.tools_json is not None:
        tools = out.parent / d.tools_file
        tools.write_text(json.dumps(d.tools_json, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        written.append((tools, 'guessed from the calls: swap in your real tool definitions if you have them'))
    contract = contracts.load(out)
    statuses = [evaluate(c, contract, context=contract.prompt).status for c in convs]
    print(_init_summary(d, written, statuses, args.path, st))
    return 0


def _init_summary(d: drafts.Draft, written: list[tuple[Path, str]], statuses: list[str], shown: str,
                  st: report.Style) -> str:
    def row(key: str, value: str) -> str:
        return f"  {st(f'{key:<11}', 'dim')} {value}"

    named = list(d.tools.items())
    tools = ', '.join(f'{n} ({k})' for n, k in named[:3]) + (f', +{len(named) - 3} more' if len(named) > 3 else '')
    L = ['', st(f'  callreplay init · {d.total} recorded calls in {shown}, {d.usable} with enough speech to use', 'bold'), '',
         row('tools', f'{len(d.tools)} used, with how many calls: {tools}' if d.tools else 'none: no call used a tool')]
    if d.end_tool or d.transfers:
        L.append(row('hangs up', ' · '.join(filter(None, [d.end_tool, d.transfers and 'transfers: ' + ', '.join(d.transfers)]))))
    if d.writes:
        L.append(row('writes', ', '.join(d.writes)))
    L.append(row('intents', f'{len(d.intents)}, one per kind of call, in the order they are tried:'))
    w = max((len(i.name) for i in d.intents), default=0)
    needs = {i.name: 'requires ' + ', '.join(i.requires) if i.requires else 'no tools' for i in d.intents}
    wn = max((len(v) for v in needs.values()), default=0)
    for i in d.intents:
        words = f"keywords: {', '.join(i.keywords)}" if i.keywords else ''
        L.append(f"  {'':<11}   {i.name:<{w}}  {len(i.calls):>3} calls   {needs[i.name]:<{wn}}   {st(words, 'dim')}".rstrip())
    if d.allowed:
        L.append(row('grounding', 'allowed without a tool: ' + ', '.join(f'"{v}" ({k} calls)' for v, k in d.allowed.items())))
    L.append('')
    for n, (p, note) in enumerate(written):
        L.append(row('wrote' if n == 0 else '', str(p) + (st(f'   {note}', 'dim') if note else '')))
    counts = [f'{statuses.count(s)} {s}' for s in report.STATUSES if statuses.count(s)]
    L += ['', row('checked', 'the same calls against this draft: ' + ' · '.join(
              st(c, report.STATUS_COLOR[c.split()[1]]) for c in counts)),
          row('next', f'rename the intents and check their keywords, then: callreplay eval {shown}'), '']
    return '\n'.join(L)


def cmd_convert(args: argparse.Namespace) -> int:
    convs = formats.load(args.path, args.format)
    formats.save(convs, args.out)
    print(f'{len(convs)} conversations written to {args.out}')
    return 0


def _finish(args: argparse.Namespace, data: dict, st: report.Style, code: int) -> int:
    paths = report.write(args.out, data)
    print(report.terminal(data, st))
    print(f"\n  {st('report', 'dim')}  {paths['html']}\n")
    step = os.environ.get('GITHUB_STEP_SUMMARY')
    if step and os.environ.get('CALLREPLAY_STEP_SUMMARY', '1') != '0':
        with open(step, 'a', encoding='utf-8') as f:          # GitHub Actions shows it on the run's page
            f.write(paths['md'].read_text(encoding='utf-8') + '\n')
    if args.open:
        webbrowser.open(paths['html'].resolve().as_uri())
    return code


def _exit_eval(fail_on: str, data: dict) -> int:
    wanted = {s.strip() for s in fail_on.split(',') if s.strip()}
    return int(bool(wanted) and any(i['reference']['status'] in wanted for i in data['items']))


def _exit_replay(fail_on: str, data: dict) -> int:
    wanted = {s.strip() for s in fail_on.split(',') if s.strip()}
    ch = data['summary']['changes']
    hit = (('regressions' in wanted and ch.get('regressed')) or ('worse' in wanted and (ch.get('worse') or ch.get('regressed')))
           or ('errors' in wanted and ch.get('error'))
           or ('fail' in wanted and any((i['replay'] or {}).get('status') == 'fail' for i in data['items'])))
    return int(bool(hit))


def _name(contract: contracts.Contract) -> str:
    return _shown(contract.path) if contract.path else ''


def _shown(path: str | Path) -> str:
    """Paths as they go into a report: relative, never the home folder of whoever ran it."""
    p = Path(path).resolve()
    for base in (Path.cwd(), Path(__file__).parent, Path(__file__).resolve().parents[1]):
        try:
            return str(p.relative_to(base))
        except ValueError:
            continue
    return p.name


if __name__ == '__main__':
    raise SystemExit(main())
