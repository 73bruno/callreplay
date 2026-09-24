"""callreplay: evaluate recorded agent conversations, and replay them against a new agent.

    callreplay eval PATH                    score recordings against a contract
    callreplay replay PATH --agent SPEC     re-run the caller's turns through another agent
    callreplay demo                         both, on the bundled example, no API key
    callreplay convert PATH OUT             turn ElevenLabs / LiveKit / OpenAI logs into this format

PATH is a folder of conversations (or a .json / .jsonl file). If the folder has a contract.toml,
it is used unless --contract says otherwise.
"""
from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path

from . import __version__, formats, report
from . import contract as contracts
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

    e = sub.add_parser('eval', help='score recorded conversations against a contract')
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

    d = sub.add_parser('demo', help='replay the bundled example (a fictional dental clinic) against a scripted agent')
    d.add_argument('--out', default='callreplay-demo')
    d.add_argument('--no-open', action='store_true')
    d.add_argument('--eval', action='store_true', help='only evaluate the recordings, no replay')

    c = sub.add_parser('convert', help='convert ElevenLabs / LiveKit / OpenAI logs into callreplay conversations')
    c.add_argument('path')
    c.add_argument('out', help='a folder (one .json per conversation) or a .jsonl file')
    c.add_argument('--format', default='auto', choices=['auto', *formats.FORMATS])

    args = p.parse_args(argv)
    try:
        return {'eval': cmd_eval, 'replay': cmd_replay, 'demo': cmd_demo, 'convert': cmd_convert}[args.cmd](args)
    except (contracts.ContractError, ValueError, FileNotFoundError) as exc:
        print(f'callreplay: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _load(args: argparse.Namespace) -> tuple[list, contracts.Contract]:
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f'{path} does not exist')
    convs = formats.load(path, args.format)
    if not convs:
        raise ValueError(f'no conversations found in {path}')
    contract = contracts.load(args.contract or (path if path.is_dir() else path.parent))
    return convs, contract


def cmd_eval(args: argparse.Namespace) -> int:
    st = report.Style()
    t0 = time.perf_counter()
    convs, contract = _load(args)
    pairs = [(c, evaluate(c, contract, context=contract.prompt)) for c in convs]
    data = report.from_eval(pairs, source=_shown(args.path), contract=_name(contract), duration=elapsed(t0))
    return _finish(args, data, st, _exit_eval(args.fail_on, data))


def cmd_replay(args: argparse.Namespace, agent=None) -> int:
    st = report.Style()
    convs, contract = _load(args)
    if not contract.prompt and agent is None and not args.agent.startswith(('demo', 'python')):
        print(st('  warning: the contract has no [agent] prompt; the agent will get no system prompt', 'yellow'))
    if args.only:
        wanted = {s.strip() for s in args.only.split(',')}
        convs = [c for c in convs if evaluate(c, contract).status in wanted]
    convs = sample(convs, contract, args.limit)
    agent = agent or from_spec(args.agent, args.base_url, args.api_key_env, args.temperature)
    print(st(f'\n  callreplay {__version__} · replaying {len(convs)} conversations with {agent.name}\n', 'bold'))
    t0 = time.perf_counter()

    def progress(n: int, total: int, r) -> None:
        if not args.quiet:
            print(report.progress_line(n, total, r, st), flush=True)

    runs = replay_all(convs, agent, contract, concurrency=args.concurrency, max_tool_rounds=args.max_tool_rounds,
                      turn_limit=args.turn_limit or None, progress=progress)
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
    return cmd_replay(ns, DemoAgent('v2', delay=0.06))       # a little pace, like a real model


def cmd_convert(args: argparse.Namespace) -> int:
    convs = formats.load(args.path, args.format)
    formats.save(convs, args.out)
    print(f'{len(convs)} conversations written to {args.out}')
    return 0


def _finish(args: argparse.Namespace, data: dict, st: report.Style, code: int) -> int:
    paths = report.write(args.out, data)
    print(report.terminal(data, st))
    print(f"\n  {st('report', 'dim')}  {paths['html']}\n")
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
    for base, prefix in ((Path.cwd(), ''), (Path(__file__).parent, ''), (Path(__file__).resolve().parents[1], '')):
        try:
            return prefix + str(p.relative_to(base))
        except ValueError:
            continue
    return p.name


if __name__ == '__main__':
    raise SystemExit(main())
