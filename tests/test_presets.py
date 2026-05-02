import pytest
from sumospace.settings import SumoSettings

def test_for_chat_disables_committee_and_rag():
    settings = SumoSettings.for_chat()
    
    assert settings.committee_enabled is False
    assert settings.rag_enabled is False
    assert settings.memory_enabled is True
    assert settings.execution_enabled is False

def test_for_chat_with_context():
    settings = SumoSettings.for_chat_with_context()
    
    assert settings.committee_enabled is False
    assert settings.rag_enabled is True
    assert settings.memory_enabled is True
    assert settings.execution_enabled is False

def test_for_chat_stateless():
    settings = SumoSettings.for_chat_stateless()
    
    assert settings.committee_enabled is False
    assert settings.rag_enabled is False
    assert settings.memory_enabled is False
    assert settings.execution_enabled is False

def test_for_review_disables_execution():
    settings = SumoSettings.for_review()
    
    assert settings.committee_enabled is True
    assert settings.committee_mode == "full"
    assert settings.execution_enabled is False

def test_for_coding_enables_sandbox():
    settings = SumoSettings.for_coding()
    
    assert settings.committee_enabled is True
    assert settings.committee_mode == "full"
    assert settings.rag_enabled is True
    assert settings.rag_top_k_final == 8
    assert settings.shell_sandbox is True

def test_for_research_enables_plan_only():
    settings = SumoSettings.for_research()
    
    assert settings.committee_enabled is True
    assert settings.committee_mode == "plan_only"
    assert settings.rag_enabled is True
    assert settings.execution_enabled is True
    assert settings.shell_sandbox is True

def test_preset_kwargs_override_preset_defaults():
    # Test that we can pass arbitrary kwargs to override the preset's opinionated defaults
    settings = SumoSettings.for_chat_stateless(rag_enabled=True, model="gpt-4")
    
    assert settings.committee_enabled is False
    assert settings.memory_enabled is False
    assert settings.execution_enabled is False
    
    # These were overridden by kwargs
    assert settings.rag_enabled is True
    assert settings.model == "gpt-4"
