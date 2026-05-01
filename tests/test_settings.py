import pytest
from sumospace.settings import SumoSettings

def test_explicit_kwarg_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SUMO_PROVIDER", "ollama")
    settings = SumoSettings(provider="hf")   # Explicit kwarg wins
    assert settings.provider == "hf"

def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("SUMO_MAX_RETRIES", "15")
    settings = SumoSettings()
    assert settings.max_retries == 15

def test_env_file_loads(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("SUMO_PROVIDER=gemini\nSUMO_VERBOSE=false\n")
    settings = SumoSettings(_env_file=str(env_file))
    assert settings.provider == "gemini"
    assert settings.verbose is False
