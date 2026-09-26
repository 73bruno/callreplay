import json
import shutil
import tomllib
from pathlib import Path

import pytest

from callreplay import contract as contracts
from callreplay import draft, formats
from callreplay.cli import main
from callreplay.evaluate import evaluate
from callreplay.model import Conversation, ToolCall, Turn

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'dental'


@pytest.fixture
def calls(tmp_path):
    """The example's calls in a flat folder, with its prompt and without its contract."""
    folder = tmp_path / 'calls'
    shutil.copytree(EXAMPLE / 'conversations', folder)
    shutil.copy(EXAMPLE / 'prompt.md', folder)
    return folder


def test_init_drafts_a_contract_that_finds_the_same_failures(calls, capsys):
    assert main(['init', str(calls)]) == 0
    out = capsys.readouterr().out
    assert 'intents     7, one per kind of call' in out and '4 fail' in out
    raw = tomllib.loads((calls / 'contract.toml').read_text())
    assert raw['agent'] == {'prompt': 'prompt.md', 'tools': 'tools.json'}
    assert raw['intents']['cancel_booking']['requires'] == ['find_booking', 'cancel_booking']
    assert raw['intents']['cancel_booking']['order'] == [['find_booking', 'cancel_booking']]
    assert raw['intents']['cancel_booking']['keywords'] == ['cancel']
    assert raw['intents']['reschedule_booking']['prefers'] == ['check_availability']
    assert raw['tools']['create_booking'] == {'required_args': ['name', 'phone', 'date', 'time', 'service'], 'writes': True}
    assert raw['end_call']['tool'] == 'end_call' and raw['end_call']['also_ends'] == ['transfer_call']
    tools = json.loads((calls / 'tools.json').read_text())
    assert {t['function']['name'] for t in tools} >= {'check_availability', 'create_booking', 'end_call'}

    # the draft catches what the hand-written contract catches: invented times and prices
    contract = contracts.load(calls)
    failed = {c.id for c in formats.load(calls) if evaluate(c, contract, context=contract.prompt).status == 'fail'}
    assert failed == {'call-006', 'call-007', 'call-027', 'call-028'}


def test_init_does_not_overwrite(calls, capsys):
    assert main(['init', str(calls)]) == 0
    assert main(['init', str(calls)]) == 2
    assert 'already exists' in capsys.readouterr().err
    assert main(['init', str(calls), '--force']) == 0


def test_init_writes_paths_relative_to_the_contract(calls, tmp_path):
    out = tmp_path / 'elsewhere' / 'contract.toml'
    assert main(['init', str(calls), '--out', str(out)]) == 0
    raw = tomllib.loads(out.read_text())
    assert raw['agent']['prompt'] == '../calls/prompt.md' and (out.parent / 'tools.json').exists()
    assert contracts.load(out).prompt.startswith('You are Ava')


def test_eval_after_init_skips_the_tools_file(calls, tmp_path, capsys):
    main(['init', str(calls)])
    assert main(['eval', str(calls), '--out', str(tmp_path / 'r'), '--quiet']) == 0
    assert len(formats.load(calls)) == 42


def test_eval_without_contract_suggests_init(calls, tmp_path, capsys):
    assert main(['eval', str(calls), '--out', str(tmp_path / 'r')]) == 0
    assert 'callreplay init' in capsys.readouterr().out


def test_draft_names_and_guesses():
    def call(*tools, said='I would like to book a table for two'):
        return Conversation('c', [Turn('user', said), Turn('agent', '', [ToolCall(t, {'id': 1}) for t in tools])])
    convs = [call('lookupTable', 'bookTable', 'hangUp'), call('lookupTable', 'bookTable', 'hangUp'),
             call('forward_to_human', said='I want to talk to a person please')]
    d = draft.draft(convs, Path('.'))
    assert d.end_tool == 'hangUp' and d.transfers == ['forward_to_human'] and d.writes == ['bookTable']
    assert [i.name for i in d.intents] == ['forward_to_human', 'bookTable']
    assert d.tools_json[0]['function']['parameters']['properties'] == {'id': {'type': 'integer'}}
    contracts.parse(tomllib.loads(d.toml.replace('tools = "tools.json"', '')))      # valid, known keys only
