# Contracts

A contract says what a correct conversation looks like. It is a TOML file, usually
`contract.toml` next to the conversations. Without one, only the generic checks run (empty turns,
repeated calls, the agent never answering).

The example in [`examples/dental/contract.toml`](../examples/dental/contract.toml) uses every
section.

## Intents

```toml
[intents.cancel]
description = "Cancel an existing appointment"      # for people; not used by the checks
keywords = ["cancel", "can't make it"]
requires = ["cancel_booking"]
requires_any = [["find_booking", "list_bookings"]]
prefers = []
forbids = ["create_booking"]
order = [["find_booking", "cancel_booking"]]
```

| Key | Meaning | Finding if broken |
|---|---|---|
| `keywords` | recognise the intent from the caller's first two turns | – |
| `requires` | every one of these tools must be called | `missing_required_tool` (error) |
| `requires_any` | at least one tool of each group | `missing_required_tool` (error) |
| `prefers` | should be called; not fatal | `missing_preferred_tool` (warning) |
| `forbids` | must not be called | `forbidden_tool` (error) |
| `order` | `[a, b]`: if `b` is called, `a` must have been called before | `wrong_order` (error) |

**How the intent is found:** the conversation's own `intent` label if it has one; otherwise the
first intent, in file order, with a keyword in the caller's first two turns; otherwise the intent
whose `requires` were all called. Put specific intents before general ones: `cancel` before
`book`, since "cancel my appointment" also contains "appointment". A replay always uses the intent
found in the recording, so both are judged by the same rules.

A conversation that matches no intent gets `unclear_intent` (review) when the contract defines
intents.

## Tools

```toml
[tools.create_booking]
required_args = ["name", "date", "time", "phone|caller_id"]
writes = true
```

- `required_args`: arguments that must be present and not empty. `a|b` accepts either.
  → `missing_args` (error).
- `writes = true`: the tool changes something (a booking, a refund, a ticket). Calling it in a
  conversation whose intent doesn't list it in `requires`, `requires_any` or `prefers` is an
  `unexpected_write` (error). With intents defined, a write in a conversation with no clear
  intent is also flagged.

Any tool called that isn't in the `[agent] tools` file or the contract is an `unknown_tool`
(warning).

## Ending the call

```toml
[end_call]
tool = "end_call"
goodbye = ["bye", "thanks", "that's all", "cheers"]
also_ends = ["transfer_call"]
```

If the caller's last turn contains a goodbye word and `tool` was never called: `missing_end_call`
(warning). A replay stops after `tool` or any tool in `also_ends`.

## Grounding

```toml
[grounding]
kinds = ["time", "money"]
allow = ["9 am", "6 pm"]
severity = "error"
```

Every time and amount of money the agent says must have come from a tool result, from the caller,
from the system prompt or from `allow`; otherwise `ungrounded_value`. Spoken times are read with
every plausible meaning ("5:30" is 05:30 or 17:30) and pass if any of them is known. Numbers in
tool results count as amounts (`{"price": 95}` grounds "$95"). Only English phrasing is
recognised for now.

## Nothing to judge

```toml
[unscorable]
min_user_words = 2
noise = ["[inaudible]", "[silence]", "..."]
```

A conversation is `unscorable`, kept out of every rate and never replayed, when the caller never
spoke, only produced noise, said fewer than `min_user_words` words, or its metadata has
`"unscorable": "reason"` (test calls, internal numbers).

## Other checks

| Code | Severity | When |
|---|---|---|
| `no_agent_response` | error | the caller spoke and the agent never said anything or called anything |
| `silent_turn` | warning | an agent turn with no text and no tool call (dead air on a call) |
| `repeated_call` | warning | the same tool with the same arguments more than `[loops] max_same_call` times (default 2) |
| `tool_loop` | warning | in a replay, still calling tools after `--max-tool-rounds` in one turn |
| `unmocked_tool` | info | in a replay, a tool result that came from neither the recording nor `[mocks]` |

## Status and score

A conversation's status is its worst finding: any error → `fail`, else any warning → `warn`,
else any review → `review`, else `pass`. The score is 100 minus a penalty per finding, capped so
it agrees with the status (fail ≤ 59, warn ≤ 84, review ≤ 89). Change the penalties with:

```toml
[score]
error = 25
warning = 10
review = 5
```

## For replays

```toml
[agent]
prompt = "prompt.md"          # the system prompt; {caller} and {now} are filled in per conversation
tools = "tools.json"          # OpenAI tool definitions
greeting = ""                 # used when the recording doesn't start with the agent speaking

[mocks.check_availability]    # the result a tool gets when the recording has none for it
slots = ["09:30", "11:00"]
```

See [replay.md](replay.md).
