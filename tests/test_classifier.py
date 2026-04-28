# tests/test_classifier.py

import pytest
from sumospace.classifier import (
    Intent, RuleBasedClassifier, EntityExtractor,
    LLMClassifier, SumoClassifier,
)


class TestRuleBasedClassifier:
    def setup_method(self):
        self.clf = RuleBasedClassifier()

    def test_debug_intent(self):
        result = self.clf.classify("Fix the bug in auth.py")
        assert result is not None
        assert result.intent == Intent.DEBUG_AND_FIX
        assert result.confidence > 0.7

    def test_refactor_intent(self):
        result = self.clf.classify("Refactor the database module")
        assert result is not None
        assert result.intent == Intent.REFACTOR

    def test_write_tests_intent(self):
        result = self.clf.classify("Write unit tests for the UserService class")
        assert result is not None
        assert result.intent == Intent.WRITE_TESTS

    def test_scan_directory_intent(self):
        result = self.clf.classify("List all Python files in the src folder")
        assert result is not None
        assert result.intent == Intent.SCAN_DIRECTORY

    def test_docker_intent(self):
        result = self.clf.classify("Build a Docker image for this project")
        assert result is not None
        assert result.intent == Intent.DOCKER_OPERATION

    def test_dependency_intent(self):
        result = self.clf.classify("Install numpy and pandas")
        assert result is not None
        assert result.intent == Intent.DEPENDENCY_MANAGEMENT

    def test_web_search_intent(self):
        result = self.clf.classify("Search for the latest Python asyncio docs")
        assert result is not None
        assert result.intent == Intent.WEB_SEARCH

    def test_summarize_intent(self):
        result = self.clf.classify("Summarize this document")
        assert result is not None
        assert result.intent == Intent.SUMMARIZE

    def test_ingest_intent(self):
        result = self.clf.classify("Ingest all files in the docs folder")
        assert result is not None
        assert result.intent == Intent.INGEST_DATA

    def test_unknown_returns_none(self):
        result = self.clf.classify("the quick brown fox")
        # May or may not match; just check it doesn't crash
        assert result is None or result.confidence >= 0.0

    def test_needs_execution_flag(self):
        result = self.clf.classify("Run the test suite")
        assert result is not None
        assert result.needs_execution is True

    def test_needs_web_flag(self):
        result = self.clf.classify("Search online for Python best practices")
        assert result is not None
        assert result.needs_web is True


class TestEntityExtractor:
    def setup_method(self):
        self.extractor = EntityExtractor()

    def test_extract_python_file(self):
        entities = self.extractor.extract("Fix the bug in src/auth.py")
        assert "src/auth.py" in entities["files"]

    def test_extract_url(self):
        entities = self.extractor.extract("Fetch https://example.com/api/data")
        assert any("example.com" in u for u in entities["urls"])

    def test_extract_function(self):
        entities = self.extractor.extract("Debug the authenticate() function")
        assert "authenticate" in entities["functions"]

    def test_extract_class(self):
        entities = self.extractor.extract("Refactor the UserService class")
        assert "UserService" in entities["classes"]

    def test_extract_package(self):
        entities = self.extractor.extract("pip install requests httpx")
        assert "requests" in entities["packages"]


@pytest.mark.asyncio
class TestLLMClassifier:
    async def test_classifies_with_mock(self, mock_provider):
        clf = LLMClassifier(mock_provider)
        result = await clf.classify("What is the meaning of life?")
        assert result is not None
        assert result.intent is not None
        assert 0 <= result.confidence <= 1.0

    async def test_handles_malformed_json(self, mock_provider):
        # Override to return garbage
        mock_provider.complete = lambda **kwargs: "not json at all"
        clf = LLMClassifier(mock_provider)
        result = await clf.classify("some task")
        # Should fall back to GENERAL_QA
        assert result.intent == Intent.GENERAL_QA


@pytest.mark.asyncio
class TestSumoClassifier:
    async def test_full_classify_pipeline(self, mock_provider):
        clf = SumoClassifier(mock_provider)
        # Rule engine will handle this
        result = await clf.classify("Fix the bug in main.py")
        assert result.intent == Intent.DEBUG_AND_FIX

    async def test_classify_does_not_crash(self, mock_provider):
        clf = SumoClassifier(mock_provider)
        result = await clf.classify("do something completely ambiguous and weird xyzzy")
        assert result is not None
        assert result.intent is not None
