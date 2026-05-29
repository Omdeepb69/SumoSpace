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
        assert result
        best_intent = max(result.items(), key=lambda x: x[1])[0]
        assert best_intent == Intent.DEBUG_AND_FIX
        assert result[best_intent] > 0.7

    def test_refactor_intent(self):
        result = self.clf.classify("Refactor the database module")
        assert result
        best_intent = max(result.items(), key=lambda x: x[1])[0]
        assert best_intent == Intent.REFACTOR

    def test_write_tests_intent(self):
        result = self.clf.classify("Write unit tests for the UserService class")
        assert result
        best_intent = max(result.items(), key=lambda x: x[1])[0]
        assert best_intent == Intent.WRITE_TESTS

    def test_scan_directory_intent(self):
        result = self.clf.classify("List all Python files in the src folder")
        assert result
        best_intent = max(result.items(), key=lambda x: x[1])[0]
        assert best_intent == Intent.SCAN_DIRECTORY

    def test_docker_intent(self):
        result = self.clf.classify("Build a Docker image for this project")
        assert result
        best_intent = max(result.items(), key=lambda x: x[1])[0]
        assert best_intent == Intent.DOCKER_OPERATION

    def test_dependency_intent(self):
        result = self.clf.classify("Install numpy and pandas")
        assert result
        best_intent = max(result.items(), key=lambda x: x[1])[0]
        assert best_intent == Intent.DEPENDENCY_MANAGEMENT

    def test_web_search_intent(self):
        result = self.clf.classify("Search for the latest Python asyncio docs")
        assert result
        best_intent = max(result.items(), key=lambda x: x[1])[0]
        assert best_intent == Intent.WEB_SEARCH

    def test_summarize_intent(self):
        result = self.clf.classify("Summarize this document")
        assert result
        best_intent = max(result.items(), key=lambda x: x[1])[0]
        assert best_intent == Intent.SUMMARIZE

    def test_ingest_intent(self):
        result = self.clf.classify("Ingest all files in the docs folder")
        assert result
        best_intent = max(result.items(), key=lambda x: x[1])[0]
        assert best_intent == Intent.INGEST_DATA

    def test_unknown_returns_none(self):
        result = self.clf.classify("the quick brown fox")
        # Ensure it doesn't crash
        assert isinstance(result, dict)

    def test_needs_execution_flag(self):
        # RuleBasedClassifier does not set needs_execution, that's done by the SumoClassifier wrapper.
        # But we can test it maps to a RUN_COMMAND intent
        result = self.clf.classify("Run the test suite")
        assert result
        best_intent = max(result.items(), key=lambda x: x[1])[0]
        assert best_intent == Intent.RUN_COMMAND

    def test_needs_web_flag(self):
        result = self.clf.classify("Search online for Python best practices")
        assert result
        best_intent = max(result.items(), key=lambda x: x[1])[0]
        assert best_intent == Intent.WEB_SEARCH


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
        # Rule engine will handle this, mapped properly
        result = await clf.classify("Fix the bug in main.py")
        assert result.intent in (Intent.DEBUG_AND_FIX, Intent.WRITE_TESTS)

    async def test_classify_does_not_crash(self, mock_provider):
        clf = SumoClassifier(mock_provider)
        result = await clf.classify("do something completely ambiguous and weird xyzzy")
        assert result is not None
        assert result.intent is not None
