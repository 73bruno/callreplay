# Replay

```bash
callreplay replay calls/ --agent openai:gpt-4.1-mini
```

For every recorded conversation that can be judged, callreplay sends the caller's turns, in
order, to the agent under test, with the contract's system prompt and tools. The agent answers
and calls tools as it would live. The replayed conversation is checked with the same contract as
the recording and the two are compared.

| Change | Meaning |
|---|---|
| `fixed` | the recording failed, the replay doesn't |
| `regressed` | the recording didn't fail, the replay does |
| `better` / `worse` | the status moved without crossing into or out of `fail` (e.g. warn → pass) |
| `same` | same status |
| `error` | the replay couldn't run: HTTP errors, timeouts, an exception in your agent. Not counted as a failure of the agent |
| `unscorable` | nothing to replay |

## Tools are never called for real

Each tool call in the replay gets, in this order:

1. the recorded result of the same tool with **the same arguments** → `recorded`
2. the next recorded result of the same tool → `recorded, other arguments`
3. the contract's `[mocks.<tool>]` → `contract mock`
4. `{"ok": true}` → `default` (and an `unmocked_tool` info finding)

The report shows the source next to every tool result. Anything other than `recorded` means the
agent took a different path from the recording and the result it got may not match what the real
system would have returned. Read those conversations before trusting their verdict.

## What a replay can't do

The caller's words are fixed. If the new agent asks something the original didn't, the next
recorded turn still answers the old question. Replays are best at catching the things a prompt
or model change usually breaks: skipped tool calls, wrong order, missing arguments, invented
values, calls that never end. Conversations that diverge a lot deserve a look, not just a score.

## Agents

| `--agent` | Endpoint | Key |
|---|---|---|
| `openai:MODEL` | api.openai.com | `OPENAI_API_KEY` |
| `gemini:MODEL` | Gemini's OpenAI-compatible endpoint | `GEMINI_API_KEY` |
| `anthropic:MODEL` | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `groq:MODEL`, `cerebras:MODEL`, `mistral:MODEL`, `together:MODEL`, `openrouter:MODEL` | their OpenAI-compatible APIs | `GROQ_API_KEY`, … |
| `ollama:MODEL` | `localhost:11434` | none |
| `compat:MODEL --base-url URL [--api-key-env VAR]` | any other OpenAI-compatible server (vLLM, LM Studio, Azure…) | yours |
| `python:module:attr` | your own code | – |
| `demo`, `demo:v1` | the scripted agent of the example | – |

Rate limits and server errors are retried (2, 5 and 15 s) before the conversation is marked
`error`. Some open models write tool calls as `<function=name>{...}</function>` text; those are
parsed too.

### Your own agent

When the agent isn't just a prompt (it has its own orchestration, a LiveKit `llm_node`, a
framework), wrap the step that produces the next reply:

```python
# my_agent.py
from callreplay.agents import AgentReply
from callreplay.model import ToolCall

def reply(messages, tools):
    """messages: OpenAI-style list (system, user, assistant with tool_calls, tool).
    Return what the agent says and the tools it calls next."""
    out = my_pipeline.next_step(messages)
    return AgentReply(text=out.text, tool_calls=[ToolCall(c.name, c.args, c.id) for c in out.calls])
```

```bash
callreplay replay calls/ --agent python:my_agent:reply
```

`attr` can also be a class or a factory returning an object with `reply(messages, tools)`.

## Options

| | |
|---|---|
| `--limit N` | at most N conversations, spread evenly across intents |
| `--only fail,warn` | only conversations whose recording has these statuses: "did the fix fix them?" |
| `--concurrency N` | conversations in parallel (default 4) |
| `--max-tool-rounds N` | tool calls in one turn before giving up (default 4) |
| `--turn-limit N` | only the first N caller turns |
| `--fail-on regressions,worse,fail,errors` | exit code 1 when any happens, for CI |
| `--temperature` | default 0.2 |

## In CI

```yaml
- run: pip install git+https://github.com/73bruno/callreplay
- run: callreplay replay calls/ --agent openai:gpt-4.1-mini --fail-on regressions
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
- run: cat callreplay-report/summary.md >> "$GITHUB_STEP_SUMMARY"
  if: always()
- uses: actions/upload-artifact@v4
  if: always()
  with: { name: callreplay-report, path: callreplay-report }
```

`summary.md` lists the regressions with their first finding, ready for a pull-request comment.
