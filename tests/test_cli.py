import json
import os
from pathlib import Path

from callreplay.cli import main

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'dental'


def test_demo_writes_every_report(tmp_path, capsys):
    assert main(['demo', '--no-open', '--out', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert '5 regressed' in out and '4 fixed' in out and '1 error ' in out
    files = {p.name for p in tmp_path.iterdir()}
    assert files == {'report.html', 'summary.md', 'results.json', 'results.csv'}
    data = json.loads((tmp_path / 'results.json').read_text())
    assert data['summary']['changes']['regressed'] == 5 and data['source'] == 'examples/dental'
    html = (tmp_path / 'report.html').read_text()
    assert '"mode": "replay"' in html and '/*__DATA__*/' not in html
    home = os.path.expanduser('~')
    assert home not in html and home not in (tmp_path / 'results.json').read_text()
    assert 'call-003' in (tmp_path / 'summary.md').read_text()


def test_eval_exit_code_for_ci(tmp_path):
    assert main(['eval', str(EXAMPLE), '--out', str(tmp_path), '--quiet']) == 0
    assert main(['eval', str(EXAMPLE), '--out', str(tmp_path), '--quiet', '--fail-on', 'fail']) == 1


def test_replay_exit_code_for_ci(tmp_path):
    args = ['replay', str(EXAMPLE), '--agent', 'demo', '--out', str(tmp_path), '--quiet']
    assert main(args) == 0
    assert main([*args, '--fail-on', 'regressions']) == 1
    assert main(['replay', str(EXAMPLE), '--agent', 'demo:v1', '--out', str(tmp_path), '--quiet',
                 '--fail-on', 'regressions']) == 0


def test_only_and_limit(tmp_path):
    assert main(['replay', str(EXAMPLE), '--agent', 'demo', '--only', 'fail', '--out', str(tmp_path), '--quiet']) == 0
    data = json.loads((tmp_path / 'results.json').read_text())
    assert data['summary']['total'] == 4 and data['summary']['changes'] == {'fixed': 4}


def test_convert(tmp_path):
    src = tmp_path / 'lk.json'
    src.write_text(json.dumps({'items': [{'type': 'message', 'role': 'user', 'content': ['hi there']}]}))
    assert main(['convert', str(src), str(tmp_path / 'out.jsonl')]) == 0
    assert json.loads((tmp_path / 'out.jsonl').read_text())['turns'][0]['text'] == 'hi there'


def test_friendly_errors(tmp_path, capsys):
    assert main(['eval', str(tmp_path / 'missing')]) == 2
    assert 'does not exist' in capsys.readouterr().err
