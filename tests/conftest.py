import pytest


@pytest.fixture(autouse=True)
def _no_github_summary(monkeypatch):
    """Under GitHub Actions every test that runs the CLI would write to the job's summary."""
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)
    monkeypatch.delenv('CALLREPLAY_STEP_SUMMARY', raising=False)
