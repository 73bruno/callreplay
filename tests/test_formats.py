import json

import pytest

from callreplay import formats


def test_openai_messages():
    c = formats.convert({'id': 'x', 'messages': [
        {'role': 'system', 'content': 'You are a receptionist.'},
        {'role': 'assistant', 'content': 'Hello!'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Book me in'}]},
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'check', 'arguments': '{"day": "fri"}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': '{"slots": ["10:00"]}'},
        {'role': 'assistant', 'content': 'I have 10 am.'}]})
    assert [t.role for t in c.turns] == ['agent', 'user', 'agent', 'tool', 'agent']
    assert c.turns[1].text == 'Book me in'
    assert c.turns[2].tool_calls[0].args == {'day': 'fri'}
    assert c.turns[3].name == 'check' and c.turns[3].result == {'slots': ['10:00']}
    assert c.metadata['system_prompt'] == 'You are a receptionist.'


def test_elevenlabs_conversation():
    c = formats.convert({
        'conversation_id': 'conv_1', 'metadata': {'start_time_unix_secs': 1789000000, 'call_duration_secs': 42,
                                                  'phone_call': {'external_number': '+15550100'}},
        'analysis': {'call_successful': 'success'},
        'transcript': [
            {'role': 'agent', 'message': 'Hi!', 'time_in_call_secs': 0},
            {'role': 'user', 'message': 'Cancel my booking', 'time_in_call_secs': 2},
            {'role': 'agent', 'message': None,
             'tool_calls': [{'tool_name': 'find_booking', 'params_as_json': '{"phone": "+15550100"}', 'request_id': 'r1'}],
             'tool_results': [{'tool_name': 'find_booking', 'result_value': '{"bookings": []}', 'request_id': 'r1'}]},
            {'role': 'agent', 'message': "I can't find it."}]})
    assert c.id == 'conv_1' and c.caller == '+15550100' and c.metadata['duration_s'] == 42
    assert [t.role for t in c.turns] == ['agent', 'user', 'agent', 'tool', 'agent']
    assert c.turns[2].tool_calls[0].name == 'find_booking' and c.turns[3].result == {'bookings': []}
    assert c.started_at.startswith('2026-')


def test_livekit_history():
    c = formats.convert({'items': [
        {'type': 'message', 'role': 'assistant', 'content': ['Hello'], 'interrupted': False},
        {'type': 'message', 'role': 'user', 'content': ['Is 3 pm free?']},
        {'type': 'function_call', 'call_id': 'a', 'name': 'check', 'arguments': '{"t": "15:00"}'},
        {'type': 'function_call', 'call_id': 'b', 'name': 'price', 'arguments': '{}'},
        {'type': 'function_call_output', 'call_id': 'a', 'name': 'check', 'output': '{"free": true}', 'is_error': False},
        {'type': 'function_call_output', 'call_id': 'b', 'name': 'price', 'output': 'timeout', 'is_error': True},
        {'type': 'message', 'role': 'assistant', 'content': ['Yes, 3 pm is free.'],
         'metrics': {'llm_node_ttft': 0.4, 'tts_node_ttfb': 0.2}},
        {'type': 'agent_handoff', 'new_agent_id': 'x'}]})
    assert [t.role for t in c.turns] == ['agent', 'user', 'agent', 'tool', 'tool', 'agent']
    assert [x.name for x in c.turns[2].tool_calls] == ['check', 'price']       # parallel calls, one turn
    assert c.turns[4].result == {'error': 'timeout'}
    assert c.turns[5].latency_ms == pytest.approx(600)


def test_native_round_trip_and_folders(tmp_path):
    conv = formats.convert({'id': 'a', 'turns': [{'role': 'user', 'text': 'hi'}, {'role': 'assistant', 'text': 'hello',
                                                                                 'latency_ms': 800}]})
    formats.save([conv], tmp_path / 'out')
    formats.save([conv], tmp_path / 'all.jsonl')
    again = formats.load(tmp_path / 'out')
    assert again[0].as_dict() == conv.as_dict() and again[0].turns[1].role == 'agent'
    assert formats.load(tmp_path / 'all.jsonl')[0].id == 'a'


def test_unknown_shape_names_the_file(tmp_path):
    (tmp_path / 'results.json').write_text(json.dumps({'mode': 'replay', 'items': [{'id': 1}]}))
    with pytest.raises(ValueError, match='results.json'):
        formats.load(tmp_path)
