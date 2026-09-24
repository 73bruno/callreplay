# Conversation formats

`callreplay` reads a folder of `.json` / `.jsonl` files (or a `conversations/` folder inside it),
a single `.json` with one conversation or a list, or a `.jsonl` with one per line. The format of
each record is detected from its shape; force one with `--format`. To store them in the native
format: `callreplay convert IN OUT`.

## Native

```json
{
  "id": "call-001",
  "caller": "+1 555 0101",
  "started_at": "2026-09-14T09:12:00-04:00",
  "intent": "book",
  "metadata": {"channel": "phone", "duration_s": 74},
  "turns": [
    {"role": "agent", "text": "Harbor Dental, how can I help?"},
    {"role": "user", "text": "I'd like a cleaning on Thursday."},
    {"role": "agent", "text": "", "latency_ms": 540,
     "tool_calls": [{"id": "c1", "name": "check_availability", "args": {"date": "2026-09-17", "service": "cleaning"}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "check_availability", "result": {"slots": ["10:30", "14:15"]}},
    {"role": "agent", "text": "I have 10:30 am or 2:15 pm.", "latency_ms": 910}
  ]
}
```

Only `turns` is required. `intent` skips intent detection. `metadata.unscorable: "reason"` keeps
a conversation out of scoring. `latency_ms` on agent turns feeds the latency figures.

## OpenAI messages

`{"id": "...", "messages": [...]}` or a bare list of messages, as sent to Chat Completions:
`system`, `user`, `assistant` (with `tool_calls`) and `tool` messages. The system message is kept
in `metadata.system_prompt`.

## ElevenLabs Agents

The conversation details from `GET /v1/convai/conversations/{conversation_id}`: `transcript`
entries with `role`, `message`, `tool_calls` (`tool_name`, `params_as_json`, `request_id`) and
`tool_results` (`result_value`, `request_id`). The caller number, start time, duration and the
platform's own `call_successful` verdict are kept.

```bash
curl -s -H "xi-api-key: $ELEVENLABS_API_KEY" \
  "https://api.elevenlabs.io/v1/convai/conversations/$ID" > calls/$ID.json
```

## LiveKit Agents

`session.history.to_dict()` (LiveKit Agents 1.x): `message`, `function_call` and
`function_call_output` items. Assistant messages' `llm_node_ttft` + `tts_node_ttfb` become the
turn's latency; tool outputs with `is_error` become `{"error": ...}`.

```python
async def entrypoint(ctx: JobContext):
    session = AgentSession(...)

    async def save_history():
        Path("calls").mkdir(exist_ok=True)
        Path(f"calls/{ctx.room.name}.json").write_text(json.dumps(session.history.to_dict()))

    ctx.add_shutdown_callback(save_history)
    await session.start(agent=MyAgent(), room=ctx.room)
```

## Anything else

Write the native format from your own call logs; a converter is usually a dozen lines. Pull
requests with importers for other platforms are welcome (`callreplay/formats.py`, plus a test in
`tests/test_formats.py`).
