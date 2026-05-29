# sumospace/classifier.py

"""
Three-Stage Hybrid Intent Classifier
======================================
Stage 1: Rule-based   (~0ms,      regex + keywords — covers ~60% of tasks)
Stage 2: Zero-shot NLI (~50-200ms, local cross-encoder — no API key, covers ~30%)
Stage 3: LLM-based    (~1-5s,     full local model — only for truly ambiguous tasks)

Zero API keys required at any stage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─── Intent Taxonomy ─────────────────────────────────────────────────────────

class Intent(str, Enum):
    # Code
    DEBUG_AND_FIX      = "debug_and_fix"
    REFACTOR           = "refactor"
    CODE_REVIEW        = "code_review"
    WRITE_CODE         = "write_code"
    WRITE_TESTS        = "write_tests"
    EXPLAIN_CODE       = "explain_code"

    # File System
    SCAN_DIRECTORY     = "scan_directory"
    READ_FILE          = "read_file"
    WRITE_FILE         = "write_file"

    # Execution
    RUN_COMMAND        = "run_command"
    DOCKER_OPERATION   = "docker_operation"
    DEPENDENCY_MANAGEMENT = "dependency_management"

    # Knowledge
    WEB_SEARCH         = "web_search"
    DOCUMENT_QA        = "document_qa"
    SUMMARIZE          = "summarize"
    RESEARCH           = "research"
    INGEST_DATA        = "ingest_data"

    # General
    GENERAL_QA         = "general_qa"


EXECUTION_INTENTS = {
    Intent.DEBUG_AND_FIX,
    Intent.REFACTOR,
    Intent.RUN_COMMAND,
    Intent.DOCKER_OPERATION,
    Intent.DEPENDENCY_MANAGEMENT,
    Intent.WRITE_TESTS,
    Intent.WRITE_FILE,
    Intent.WRITE_CODE,
    Intent.EXPLAIN_CODE,  # needs tool use to read files before explaining
}

RETRIEVAL_INTENTS = {
    Intent.DOCUMENT_QA,
    Intent.CODE_REVIEW,
    Intent.REFACTOR,
    Intent.SUMMARIZE,
    Intent.EXPLAIN_CODE,
}

WEB_INTENTS = {Intent.WEB_SEARCH, Intent.RESEARCH}


# ─── Result ──────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    intent: Intent
    confidence: float
    needs_execution: bool
    needs_web: bool
    needs_retrieval: bool
    reasoning: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    routing_trace: dict[str, Any] = field(default_factory=dict)


# ─── Stage 1: Rule-Based Classifier ──────────────────────────────────────────

class RuleBasedClassifier:
    """
    Fast keyword + regex rule engine.
    Covers the most common, unambiguous task phrasings.
    Zero latency, zero API calls.
    """

    TIER1_RULES: list[tuple[re.Pattern, Intent, float]] = [
        # Docstrings — strongest signal: any mention of docstring/s in an imperative context
        (re.compile(r"\b(add|write|insert|create|generate)\b.{0,40}\bdocstring", re.I),
         Intent.REFACTOR, 0.95),
        # Dead code removal
        (re.compile(r"\b(remove|delete|clean|strip)\b.{0,30}\b(dead.?code|unused|deprecated|legacy)\b", re.I),
         Intent.REFACTOR, 0.95),
        # Sync-to-async conversion
        (re.compile(r"\b(refactor|convert|change|rewrite)\b.{0,60}\b(async|asyncio|await)\b", re.I),
         Intent.REFACTOR, 0.95),
        # Explain codebase
        (re.compile(r"\b(explain|describe)\b.{0,40}\b(codebase|project|repository|repo)\b", re.I),
         Intent.EXPLAIN_CODE, 0.90),
        # Write tests
        (re.compile(r"\b(write|add|create|generate)\b.{0,20}\b(test|tests|unit test|pytest|spec)\b", re.I),
         Intent.WRITE_TESTS, 0.92),
        # Docker
        (re.compile(r"\b(docker|container|dockerfile|compose|image|pod|k8s|kubernetes)\b", re.I),
         Intent.DOCKER_OPERATION, 0.92),
        # Dependencies
        (re.compile(r"\b(pip|npm|yarn|poetry|install|package|dependency|requirements|pyproject)\b", re.I),
         Intent.DEPENDENCY_MANAGEMENT, 0.88),
        # Ingest
        (re.compile(r"\b(ingest|index|embed|vectorize|load into|store in)\b", re.I),
         Intent.INGEST_DATA, 0.90),
        # Code review
        (re.compile(r"\b(review|audit|check|analyse|analyze)\b.{0,20}\b(code|file|module|pr|pull request)\b", re.I),
         Intent.CODE_REVIEW, 0.85),
        # Explain code
        (re.compile(r"\b(explain|describe|what does|how does|walk me through|summarize)\b.{0,20}\b(code|function|class|method|file)\b", re.I),
         Intent.EXPLAIN_CODE, 0.85),
        # Debug / Fix
        (re.compile(r"\b(fix|debug|bug|error|exception|traceback|crash|failing|broken|not working)\b", re.I),
         Intent.DEBUG_AND_FIX, 0.85),
    ]

    TIER2_RULES: list[tuple[re.Pattern, Intent, float]] = [
        # Write code
        (re.compile(r"\b(write|create|implement|build|generate|add)\b.{0,30}\b(function|class|module|script|endpoint|api|method)\b", re.I),
         Intent.WRITE_CODE, 0.85),
        # Scan directory
        (re.compile(r"\b(list|scan|find|show|ls)\b.{0,20}\b(file|files|directory|dir|folder|repo)\b", re.I),
         Intent.SCAN_DIRECTORY, 0.88),
        # Read file
        (re.compile(r"\b(read|open|cat|show|print|display)\b.{0,20}\b(file|\.py|\.js|\.ts|\.md|\.json|\.yaml)\b", re.I),
         Intent.READ_FILE, 0.85),
        # Run command
        (re.compile(r"\b(run|execute|bash|shell|terminal|cmd|command)\b", re.I),
         Intent.RUN_COMMAND, 0.82),
        # Web search (Lowered confidence to act as weak prior)
        (re.compile(r"\b(search|google|look up|find online|latest|current|news|today)\b", re.I),
         Intent.WEB_SEARCH, 0.40),
        # Document QA
        (re.compile(r"\b(from the doc|according to|in the document|in the file|based on)\b", re.I),
         Intent.DOCUMENT_QA, 0.85),
        # Summarize
        (re.compile(r"\b(summarize|summary|tldr|tl;dr|brief|overview)\b", re.I),
         Intent.SUMMARIZE, 0.88),
        # Research
        (re.compile(r"\b(research|investigate|explore|compare|benchmark|evaluate)\b", re.I),
         Intent.RESEARCH, 0.78),
    ]

    TIER3_RULES: list[tuple[re.Pattern, Intent, float]] = [
        # Refactor
        (re.compile(r"\b(refactor|restructure|clean up|reorganize|improve|simplify|optimize)\b", re.I),
         Intent.REFACTOR, 0.82),
        # Write file
        (re.compile(r"\b(write|save|create|update|edit|modify)\b.{0,20}\b(file|\.py|\.js|\.ts|\.md|\.json|\.yaml)\b", re.I),
         Intent.WRITE_FILE, 0.78),
    ]

    def classify(self, text: str) -> dict[Intent, float]:
        scores = {}
        for tier in [self.TIER1_RULES, self.TIER2_RULES, self.TIER3_RULES]:
            for pattern, intent, confidence in tier:
                if pattern.search(text):
                    scores[intent] = max(scores.get(intent, 0.0), confidence)
        
        # Apply dynamic modifiers
        if Intent.WEB_SEARCH in scores:
            coding_terms = re.compile(r"\b(function|class|refactor|async|docstring|patch|file|import|pytest|fix|bug|lint|ast|module)\b", re.I)
            if coding_terms.search(text):
                scores[Intent.WEB_SEARCH] = max(0.05, scores[Intent.WEB_SEARCH] - 0.8)
                
            web_terms = re.compile(r"\b(latest news|current|search online|lookup|google)\b", re.I)
            if web_terms.search(text):
                # Only apply web bonus if we didn't just nuke it for being coding-related
                if not coding_terms.search(text):
                    scores[Intent.WEB_SEARCH] = min(0.95, scores[Intent.WEB_SEARCH] + 0.3)

        return scores


# ─── Stage 2: Zero-Shot NLI Classifier ───────────────────────────────────────

class ZeroShotLocalClassifier:
    """
    Zero-shot classification using a local NLI cross-encoder.
    No API key. No internet at inference time.

    Model: cross-encoder/nli-deberta-v3-small (~180MB, CPU-friendly)
    Speed: ~50-200ms on CPU — much faster than a full LLM call.

    Uses Natural Language Inference (entailment) to score each intent label.
    """

    NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

    INTENT_LABELS: dict[str, Intent] = {
        "debugging and fixing errors or bugs":             Intent.DEBUG_AND_FIX,
        "refactoring or restructuring code":               Intent.REFACTOR,
        "reviewing code for issues":                       Intent.CODE_REVIEW,
        "writing new code or functions":                   Intent.WRITE_CODE,
        "writing tests for code":                          Intent.WRITE_TESTS,
        "explaining how code works":                       Intent.EXPLAIN_CODE,
        "listing or scanning files and directories":       Intent.SCAN_DIRECTORY,
        "reading a file":                                  Intent.READ_FILE,
        "writing or editing a file":                       Intent.WRITE_FILE,
        "running shell commands":                          Intent.RUN_COMMAND,
        "docker or container operations":                  Intent.DOCKER_OPERATION,
        "installing packages or managing dependencies":    Intent.DEPENDENCY_MANAGEMENT,
        "searching the web for information":               Intent.WEB_SEARCH,
        "answering questions from documents":              Intent.DOCUMENT_QA,
        "summarizing content":                             Intent.SUMMARIZE,
        "researching a topic":                             Intent.RESEARCH,
        "ingesting or indexing data":                      Intent.INGEST_DATA,
        "general question or conversation":                Intent.GENERAL_QA,
    }

    def __init__(self):
        self._model = None

    def _ensure_model(self) -> bool:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.NLI_MODEL)
            except (ImportError, Exception):
                return False
        return True

    async def classify(self, text: str) -> dict[Intent, float]:
        import asyncio
        if not self._ensure_model():
            return {}

        labels = list(self.INTENT_LABELS.keys())
        pairs = [(text, label) for label in labels]

        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs, apply_softmax=True),
        )

        scores_dict = {}
        for idx, s in enumerate(scores):
            score = float(s[2]) if hasattr(s, "__len__") else float(s)
            intent = self.INTENT_LABELS[labels[idx]]
            scores_dict[intent] = max(scores_dict.get(intent, 0.0), score)

        return scores_dict


# ─── Stage 3: LLM Classifier ─────────────────────────────────────────────────

class LLMClassifier:
    """
    LLM-based intent classifier. Uses the already-initialised provider.
    Only invoked when both rule-based and zero-shot classifiers are insufficiently confident.

    Asks the LLM to respond with a structured JSON classification.
    """

    SYSTEM = """You are an intent classification assistant.
Given a user task description, output ONLY a JSON object with these fields:
{
  "intent": <one of the intent values listed>,
  "confidence": <float 0-1>,
  "needs_execution": <bool>,
  "needs_web": <bool>,
  "needs_retrieval": <bool>,
  "reasoning": <one sentence>
}

Valid intent values:
debug_and_fix, refactor, code_review, write_code, write_tests, explain_code,
scan_directory, read_file, write_file, run_command, docker_operation,
dependency_management, web_search, document_qa, summarize, research,
ingest_data, general_qa

Output ONLY the JSON. No markdown, no explanation."""

    def __init__(self, provider):
        self._provider = provider

    async def classify(
        self,
        text: str,
        context: dict | None = None,
    ) -> ClassificationResult:
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context)}"

        try:
            raw = await self._provider.complete(
                user=f"Task: {text}{context_str}",
                system=self.SYSTEM,
                temperature=0.0,
                max_tokens=256,
            )

            # Strip any accidental markdown fences
            raw = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)

            intent_str = data.get("intent", "general_qa")
            try:
                intent = Intent(intent_str)
            except ValueError:
                intent = Intent.GENERAL_QA

            return ClassificationResult(
                intent=intent,
                confidence=float(data.get("confidence", 0.6)),
                needs_execution=bool(data.get("needs_execution", False)),
                needs_web=bool(data.get("needs_web", False)),
                needs_retrieval=bool(data.get("needs_retrieval", False)),
                reasoning=data.get("reasoning", "llm classification"),
            )
        except Exception as e:
            return ClassificationResult(
                intent=Intent.GENERAL_QA,
                confidence=0.5,
                needs_execution=False,
                needs_web=False,
                needs_retrieval=False,
                reasoning=f"llm parse failed ({e}), defaulting to general_qa",
            )


# ─── Entity Extractor ─────────────────────────────────────────────────────────

class EntityExtractor:
    """
    Lightweight regex-based entity extractor.
    Pulls out file paths, URLs, function/class names, package names.
    """

    FILE_PATH  = re.compile(r"(?:[\w./\\-]+/)?[\w.-]+\.(?:py|js|ts|jsx|tsx|json|yaml|yml|md|txt|csv|pdf|cpp|h|go|rs)\b")
    URL        = re.compile(r"https?://[^\s]+")
    FUNC_NAME  = re.compile(r"\b([a-z_][a-z0-9_]*)\(\)", re.I)
    CLASS_NAME = re.compile(r"\b([A-Z][a-zA-Z0-9]+)\b")
    PKG_NAME   = re.compile(r"\b(pip|npm)\s+install\s+([\w@/-]+)")

    def extract(self, text: str) -> dict[str, list[str]]:
        return {
            "files":    list(set(self.FILE_PATH.findall(text))),
            "urls":     list(set(self.URL.findall(text))),
            "functions": list(set(self.FUNC_NAME.findall(text))),
            "classes":  list(set(self.CLASS_NAME.findall(text))),
            "packages": [m[1] for m in self.PKG_NAME.findall(text)],
        }


# ─── Main Classifier ─────────────────────────────────────────────────────────

class SumoClassifier:
    """
    Three-stage hybrid classifier. All stages work without API keys.

    Stage 1: Rule-based    (~0ms,      regex/keyword — 60% coverage)
    Stage 2: Zero-shot NLI (~50-200ms, local NLI model — 30% coverage)
    Stage 3: LLM-based     (~1-5s,     full local model — remaining ambiguous cases)
    """

    def __init__(self, provider, llm_threshold: float = 0.72):
        self._rule = RuleBasedClassifier()
        self._zeroshot = ZeroShotLocalClassifier()
        self._llm = LLMClassifier(provider)
        self._extractor = EntityExtractor()
        self._threshold = llm_threshold

    async def initialize(self):
        pass

    async def classify(
        self,
        text: str,
        context: dict | None = None,
    ) -> ClassificationResult:
        context = context or {}
        routing_trace = {"stage1_regex": {}, "stage2_nli": {}}

        # Stage 1: Rule engine
        rule_scores = self._rule.classify(text)
        routing_trace["stage1_regex"] = {k.value: v for k, v in rule_scores.items()}

        # Stage 2: Local NLI zero-shot
        nli_scores = await self._zeroshot.classify(text)
        routing_trace["stage2_nli"] = {k.value: v for k, v in nli_scores.items()}

        # Merge scores (NLI takes precedence, but rules boost it)
        merged_scores = {}
        for intent in Intent:
            nli_s = nli_scores.get(intent, 0.0)
            rule_s = rule_scores.get(intent, 0.0)
            merged_scores[intent] = min(1.0, max(nli_s, rule_s) + (0.1 * min(nli_s, rule_s)))
            
        routing_trace["merged_scores"] = {k.value: v for k, v in merged_scores.items()}

        # Find best intent
        if merged_scores:
            best_intent = max(merged_scores.keys(), key=lambda i: merged_scores[i])
            best_score = merged_scores[best_intent]
            
            if best_score >= self._threshold:
                routing_trace["final_intent"] = best_intent.value
                return ClassificationResult(
                    intent=best_intent,
                    confidence=best_score,
                    needs_execution=best_intent in EXECUTION_INTENTS,
                    needs_web=best_intent in WEB_INTENTS,
                    needs_retrieval=best_intent in RETRIEVAL_INTENTS,
                    reasoning=f"merged heuristic score: {best_score:.2f}",
                    entities=self._extractor.extract(text),
                    routing_trace=routing_trace
                )

        # Stage 3: Full LLM call (fallback)
        routing_trace["stage3_llm"] = "fallback_triggered"
        result = await self._llm.classify(text, context)
        result.entities = self._extractor.extract(text)
        result.routing_trace = routing_trace
        routing_trace["final_intent"] = result.intent.value
        return result

    def _extract_entities(self, text: str) -> dict:
        return self._extractor.extract(text)
