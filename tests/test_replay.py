import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from callreplay import agents, formats
from callreplay import contract as contracts
from callreplay.agents import AgentError, AgentReply, OpenAICompatAgent, to_anthropic
from callreplay.demo_agent import DemoAgent
from callreplay.model import Conversation, ToolCall, Turn
from callreplay.replay import ToolMock, compare, replay_all, replay_one, sample

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'dental'


@pytest.fixture(scope='module')
def dental():
    return contracts.load(EXAMPLE), formats.load(EXAMPLE)


def test_demo_story(dental):
    contract, convs = dental
    runs = {r.id: r for r in replay_all(convs, DemoAgent('v2'), contract)}
    changes = Counter(r.change for r in runs.values())
    assert changes == {'same': 22, 'better': 5, 'regressed': 5, 'unscorable': 5, 'fixed': 4, 'error': 1}
    assert {i for i, r in runs.items() if r.change == 'regressed'} == {'call-003', 'call-004', 'call-005', 'call-014', 'call-015'}
    assert runs['call-008'].replay.status == 'error' and '503' in runs['call-008'].replay.reason
    assert runs['call-038'].replay_conv is None                         # unscorable: not even replayed
    assert runs['call-005'].sources['recorded, other arguments'] == 1   # booked 11:00, recording had 11:30


def test_replaying_the_recording_agent_changes_nothing(dental):
    contract, convs = dental
    runs = replay_all(convs, DemoAgent('v1'), contract)
    assert Counter(r.change for r in runs) == {'same': 37, 'unscorable': 5}
    assert all(r.same_tools for r in runs)
    assert sum(r.sources['recorded'] for r in runs) > 0 and not any(r.sources['default'] for r in runs)


def test_tool_mock_order():
    rec = Conversation('x', [Turn('agent', '', [ToolCall('check', {'d': 1}, 'a'), ToolCall('check', {'d': 2}, 'b')]),
                             Turn('tool', tool_call_id='a', name='check', result='one'),
                             Turn('tool', tool_call_id='b', name='check', result='two')])
    mock = ToolMock(rec, contracts.Contract(mocks={'price': {'price': 5}}))
    assert mock('check', {'d': 2}) == ('two', 'recorded')
    assert mock('check', {'d': 9}) == ('one', 'recorded, other arguments')
    assert mock('check', {'d': 1}) == ({'ok': True}, 'default')
    assert mock('price', {}) == ({'price': 5}, 'contract mock')


def test_compare():
    from callreplay.evaluate import Result
    r = lambda s: Result('x', s, None)
    assert compare(r('fail'), r('pass')) == 'fixed'
    assert compare(r('pass'), r('fail')) == 'regressed'
    assert compare(r('warn'), r('pass')) == 'better'
    assert compare(r('pass'), r('review')) == 'worse'
    assert compare(r('pass'), r('error')) == 'error'


class Scripted:
    name = 'scripted'

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def reply(self, messages, tools):
        self.seen.append([m['role'] for m in messages])
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_replay_loop_and_end_call(dental):
    contract, _ = dental
    rec = Conversation('x', [Turn('agent', 'Hi'), Turn('user', 'when is my appointment'),
                             Turn('user', 'thanks, bye'), Turn('user', 'never reached')], caller='+1 555 0000')
    agent = Scripted(AgentReply('', [ToolCall('find_booking', {'phone': '+1 555 0000'})]),
                     AgentReply('Friday at 10.'),
                     AgentReply('Bye!', [ToolCall('end_call', {})]))
    run = replay_one(rec, agent, contract)
    assert agent.seen[0] == ['system', 'assistant', 'user']          # prompt, recorded greeting, caller
    assert agent.seen[1][-1] == 'tool'
    assert [t.role for t in run.replay_conv.turns] == ['agent', 'user', 'agent', 'tool', 'agent', 'user', 'agent', 'tool']
    assert 'Caller: +1 555 0000' in contract.prompt.replace('{caller}', '+1 555 0000')
    assert any(f.code == 'unmocked_tool' for f in run.replay.findings)   # find_booking had no recorded result


def test_agent_errors_are_not_agent_failures(dental):
    contract, _ = dental
    rec = Conversation('x', [Turn('user', 'can I book a cleaning')])
    run = replay_one(rec, Scripted(AgentError('HTTP 500: boom')), contract)
    assert run.change == 'error' and run.replay.reason == 'HTTP 500: boom'
    run = replay_one(rec, Scripted(KeyError('oops')), contract)
    assert run.change == 'error' and run.replay.reason.startswith('KeyError')


def test_sample_spreads_across_intents(dental):
    contract, convs = dental
    picked = sample(convs, contract, 7)
    from callreplay.evaluate import evaluate
    assert len(picked) == 7 and len({evaluate(c, contract).intent for c in picked}) >= 6


# ----------------------------------------------------------------------------- http agents
class _Server(BaseHTTPRequestHandler):
    responses: list = []
    bodies: list = []

    def do_POST(self):
        _Server.bodies.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
        code, body = _Server.responses.pop(0)
        self.send_response(code)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(agents.time, 'sleep', lambda s: None)
    httpd = HTTPServer(('127.0.0.1', 0), _Server)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _Server.responses, _Server.bodies = [], []
    yield f'http://127.0.0.1:{httpd.server_port}/v1'
    httpd.shutdown()


def test_openai_compatible_agent(server):
    _Server.responses = [
        (429, {'error': 'slow down'}),
        (200, {'choices': [{'message': {'content': None, 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'check', 'arguments': '{"d": 1}'}}]}}],
               'usage': {'prompt_tokens': 12, 'completion_tokens': 3}}),
        (200, {'choices': [{'message': {'content': 'Sure <function=end_call>{"reason": "done"}</function>'}}]}),
        (400, {'error': 'bad request'})]
    a = OpenAICompatAgent('m', server, '', 0.2, 100, 5)
    r = a.reply([{'role': 'user', 'content': 'hi'}], [{'type': 'function', 'function': {'name': 'check'}}])
    assert r.tool_calls[0].name == 'check' and r.tool_calls[0].args == {'d': 1} and r.tokens_in == 12
    assert _Server.bodies[-1]['tools'][0]['function']['name'] == 'check'
    r = a.reply([], [])
    assert r.text == 'Sure' and r.tool_calls[0].name == 'end_call'       # calls written as text
    with pytest.raises(AgentError, match='400'):
        a.reply([], [])


def test_agent_specs(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'k')
    a = agents.from_spec('groq:llama-3.3-70b')
    assert a.name == 'groq:llama-3.3-70b' and a.base_url.startswith('https://api.groq.com')
    assert agents.from_spec('ollama:qwen2.5').key == ''
    assert agents.from_spec('demo').name == 'demo:v2'
    with pytest.raises(ValueError, match='set OPENAI_API_KEY'):
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        agents.from_spec('openai:gpt-x')
    with pytest.raises(ValueError, match='unknown agent'):
        agents.from_spec('nope:x')


def test_python_agent(tmp_path, monkeypatch):
    (tmp_path / 'my_agent.py').write_text(
        'from callreplay.agents import AgentReply\n'
        'def reply(messages, tools):\n    return AgentReply("echo: " + messages[-1]["content"])\n')
    monkeypatch.chdir(tmp_path)
    a = agents.from_spec('python:my_agent:reply')
    assert a.reply([{'role': 'user', 'content': 'hi'}], []).text == 'echo: hi'


def test_anthropic_translation():
    system, msgs = to_anthropic([
        {'role': 'system', 'content': 'Be brief.'},
        {'role': 'assistant', 'content': 'Hello'},
        {'role': 'user', 'content': 'Book me'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 't1', 'type': 'function', 'function': {'name': 'a', 'arguments': '{"x": 1}'}},
            {'id': 't2', 'type': 'function', 'function': {'name': 'b', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 't1', 'content': 'one'},
        {'role': 'tool', 'tool_call_id': 't2', 'content': 'two'}])
    assert system == 'Be brief.'
    assert [m['role'] for m in msgs] == ['user', 'assistant', 'user', 'assistant', 'user']
    assert [b['type'] for b in msgs[3]['content']] == ['tool_use', 'tool_use']
    assert [b['tool_use_id'] for b in msgs[4]['content']] == ['t1', 't2']
