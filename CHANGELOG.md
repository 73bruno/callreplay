# Changelog

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
