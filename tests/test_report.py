import io
import json
from pathlib import Path

import pytest

from callreplay import contract as contracts
from callreplay import formats, report
from callreplay.cli import main
from callreplay.demo_agent import DemoAgent
from callreplay.replay import replay_all

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'dental'


def conv(*tools):
    return {'turns': [{'role': 'agent', 'text': '', 'tool_calls': [{'name': t} for t in tools]}]}


def test_tool_diff_lines_up_what_is_missing():
    pairs = report.tool_diff(conv('check_availability', 'create_booking', 'end_call'), conv('create_booking', 'end_call'))
    assert pairs == [['check_availability', None], ['create_booking', 'create_booking'], ['end_call', 'end_call']]
    a, b = report.aligned(pairs)
    assert a == 'check_availability → create_booking → end_call'
    assert b == '                     create_booking → end_call'
    assert b.index('create_booking') == a.index('create_booking')


def test_tool_diff_folds_repeats_and_shows_extra_calls():
    pairs = report.tool_diff(conv('find', 'cancel'), conv('find', 'find', 'find', 'create', 'cancel'))
    assert pairs == [['find', 'find ×3'], [None, 'create'], ['cancel', 'cancel']]
    assert report.tool_diff(conv(), conv('x')) == [[None, 'x']]


@pytest.fixture(scope='module')
def demo():
    contract, convs = contracts.load(EXAMPLE), formats.load(EXAMPLE)
    return report.from_replay(replay_all(convs, DemoAgent('v2'), contract), agent='demo:v2')


def test_causes_group_the_same_mistake(demo):
    causes = demo['summary']['causes']
    regressed = [g for g in causes if g['kind'] == 'regressed']
    assert [(g['label'], g['ids']) for g in regressed] == [
        ('"book" requires check_availability, never called.', ['call-003', 'call-004', 'call-005']),
        ('"cancel" requires find_booking, never called.', ['call-014', 'call-015'])]
    example = regressed[0]['example']
    assert example['id'] == 'call-003' and example['caller'].startswith('Hi, this is Priya Shah')
    assert example['diff'][0] == ['check_availability', None]
    fixed = {g['label']: g['ids'] for g in causes if g['kind'] == 'fixed'}
    assert fixed == {"Said a time that no tool returned and the caller didn't say.": ['call-006', 'call-007'],
                     "Said an amount that no tool returned and the caller didn't say.": ['call-027', 'call-028']}
    assert [g['ids'] for g in causes if g['kind'] == 'error'] == [['call-008']]
    assert [g['kind'] for g in causes] == sorted((g['kind'] for g in causes),
                                                 key=['regressed', 'worse', 'fixed', 'better', 'error'].index)


def test_summaries_show_the_cause_and_the_tools(demo):
    md = report.markdown(demo)
    assert '#### Regressed (5)' in md and '**"book" requires check_availability, never called.** `call-003`' in md
    assert 'recorded  check_availability → create_booking → end_call' in md
    assert 'replayed                       create_booking → end_call' in md
    assert '- was: Said a time that no tool returned' in md and '\n\n\n' not in md
    text = report.terminal(demo, report.Style(io.StringIO()))
    assert 'Regressed · 5 calls' in text and 'call-003 call-004 call-005' in text and '\033[' not in text


def test_progress_bar_redraws_in_place(demo):
    out = io.StringIO()
    bar = report.Progress(2, report.Style(out), out)
    runs = [r for r in replay_all(formats.load(EXAMPLE)[:4], DemoAgent('v2'), contracts.load(EXAMPLE))][2:4]
    bar.update(runs[0], 'a line above the bar')
    bar.update(runs[1])
    bar.close()
    text = out.getvalue()
    assert text.startswith('\r\033[2Ka line above the bar\n') and text.endswith('2/2   2 regressed\n')


def test_github_step_summary(tmp_path, monkeypatch, capsys):
    step = tmp_path / 'step.md'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(step))
    assert main(['replay', str(EXAMPLE), '--agent', 'demo', '--quiet', '--out', str(tmp_path / 'a')]) == 0
    assert '#### Regressed (5)' in step.read_text()
    step.unlink()
    monkeypatch.setenv('CALLREPLAY_STEP_SUMMARY', '0')
    assert main(['replay', str(EXAMPLE), '--agent', 'demo', '--quiet', '--out', str(tmp_path / 'b')]) == 0
    assert not step.exists()


def test_report_html_carries_causes_and_diffs(tmp_path):
    assert main(['replay', str(EXAMPLE), '--agent', 'demo', '--quiet', '--out', str(tmp_path)]) == 0
    data = json.loads((tmp_path / 'results.json').read_text())
    item = next(i for i in data['items'] if i['id'] == 'call-014')
    assert item['tool_diff'][0] == ['find_booking', None]
    html = (tmp_path / 'report.html').read_text()
    assert 'What broke' in html and '"causes"' in html
