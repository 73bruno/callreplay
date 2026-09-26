# Changelog

## 0.2.0 — 2026-09-25

- `callreplay init` drafts a contract from recorded calls: one intent per kind of call with the
  tools those calls always used, their order and the words callers used for them; the arguments
  each tool always got; which tools change data, hang up or transfer; and the times and prices the
  agent says without a tool. It writes a `tools.json` too when there is none, and checks the calls
  against the draft.
- Results are grouped by cause: the terminal, `summary.md` and the report show what broke and what
  got fixed once per cause, with one call as an example (the caller's words and the tools recorded
  against the tools replayed).
- Tool sequences lined up, recorded against replayed, in the terminal, the summary and the report,
  so a skipped or extra tool is a visible gap.
- A progress bar while replaying; only the calls whose result changed are printed.
- In GitHub Actions the summary goes to the job summary by itself (`CALLREPLAY_STEP_SUMMARY=0` turns
  it off).
- The report opens with what broke and what got fixed, lists what wasn't judged and why, and shows
  the tools of each call under its row.
- `tools.json` and `results.json` next to the calls are no longer read as conversations.

## 0.1.0 — 2026-09-24

First release.

- Evaluate recorded conversations against a TOML contract: required, preferred and forbidden
  tools per intent, call order, required arguments, writes outside their intent, grounding of
  times and prices, missing hang-up, repeated calls, silent turns.
- Unscorable conversations and replay errors are reported apart from agent failures.
- Replay recorded caller turns against OpenAI-compatible APIs, Anthropic, local models or a
  Python function, with tool results from the recording and contract mocks.
- Reports: HTML (side by side), Markdown summary, JSON, CSV; exit codes for CI.
- Importers for ElevenLabs Agents, LiveKit Agents and OpenAI message logs.
- A synthetic example (a dental clinic, 42 calls) and `callreplay demo`.
