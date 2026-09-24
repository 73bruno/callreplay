# Contributing

```bash
git clone https://github.com/73bruno/callreplay && cd callreplay
pip install -e ".[dev]"
pytest && ruff check .
```

Everything runs offline in a few seconds. Keep the package free of runtime dependencies.

Most useful right now:

- **Importers** for other voice and chat platforms (`callreplay/formats.py` + a test with a real,
  anonymised record in `tests/test_formats.py`).
- **Checks** that catch real failures you have seen (`callreplay/checks.py`, one function, a
  stable finding code, documented in `docs/contracts.md`).
- **Grounding** for other languages and value kinds (`callreplay/grounding.py`).

Never include real conversations in issues or pull requests; anonymise them or rewrite them.

If you change the example data, regenerate it with `python scripts/make_demo.py` and update the
counts the tests expect.
