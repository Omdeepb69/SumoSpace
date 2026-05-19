import pytest
import json
from pydantic import ValidationError
from sumospace.schemas import ExecutionPlan, ExecutionStep, CritiqueVerdict, ResolverOutput, dereference_schema
from sumospace.providers.base import BaseProvider, ProviderCapabilities
from sumospace.providers import ProviderRouter


class MockStructuredProvider(BaseProvider):
    name = "mock_structured"
    
    def __init__(self):
        self.capabilities = ProviderCapabilities(structured_output=True)
    
    async def complete(self, *args, **kwargs):
        return '{"reasoning": "legacy", "steps": []}'
        
    async def complete_structured(self, *args, **kwargs):
        return '{"protocol_version": "1.0", "reasoning": "structured", "steps": [], "estimated_duration_s": 10, "risks": []}'


class MockXMLProvider(BaseProvider):
    name = "mock_xml"
    
    def __init__(self):
        self.capabilities = ProviderCapabilities(structured_output=False, preferred_fallback="xml")
        
    async def complete(self, user, system, *args, **kwargs):
        return "<response>\n<protocol_version>1.0</protocol_version>\n<reasoning>xml</reasoning>\n<steps>[]</steps>\n</response>"


# ── Helpers ──────────────────────────────────────────────────────────────────

GRAMMAR_FORBIDDEN_KEYS = {"anyOf", "oneOf", "allOf", "$ref"}

def _collect_forbidden(schema, path=""):
    """Recursively scan a JSON schema for keys that break Ollama/llama.cpp grammar."""
    violations = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            current_path = f"{path}.{key}" if path else key
            if key in GRAMMAR_FORBIDDEN_KEYS:
                violations.append(current_path)
            violations.extend(_collect_forbidden(value, current_path))
    elif isinstance(schema, list):
        for i, item in enumerate(schema):
            violations.extend(_collect_forbidden(item, f"{path}[{i}]"))
    return violations


# ── Schema Validation Tests ──────────────────────────────────────────────────

def test_schema_valid_json():
    """Test that a valid JSON can be mapped to ExecutionPlan."""
    valid_json = """
    {
      "protocol_version": "1.0",
      "reasoning": "Valid plan",
      "steps": [
        {"step_number": 1, "tool": "shell", "description": "echo hi", "parameters": {}, "critical": false, "expected_output": ""}
      ],
      "estimated_duration_s": 5
    }
    """
    plan = ExecutionPlan.model_validate_json(valid_json)
    assert plan.reasoning == "Valid plan"
    assert len(plan.steps) == 1


def test_schema_invalid_semantics():
    """Test that syntactically valid JSON with invalid semantics raises ValidationError."""
    invalid_json = """
    {
      "protocol_version": "1.0",
      "reasoning": "Bad plan",
      "steps": "not_a_list"
    }
    """
    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate_json(invalid_json)


def test_resolver_output_has_revision_field():
    """Test that ResolverOutput uses has_revision + revised_steps instead of revised_plan."""
    valid = '{"approved": true, "has_revision": false, "revised_steps": [], "approval_notes": "LGTM"}'
    model = ResolverOutput.model_validate_json(valid)
    assert model.approved is True
    assert model.has_revision is False
    assert model.revised_steps == []
    assert model.approval_notes == "LGTM"


def test_resolver_output_with_revision():
    """Test that ResolverOutput correctly deserializes revised_steps."""
    valid = json.dumps({
        "approved": True,
        "has_revision": True,
        "revised_steps": [
            {"step_number": 1, "tool": "read_file", "description": "Read the target file"}
        ],
        "approval_notes": "Added a read step before write"
    })
    model = ResolverOutput.model_validate_json(valid)
    assert model.has_revision is True
    assert len(model.revised_steps) == 1
    assert model.revised_steps[0].tool == "read_file"


def test_resolver_output_rejection():
    """Test that ResolverOutput correctly handles rejections."""
    valid = '{"approved": false, "rejection_reason": "Blockers are unresolvable"}'
    model = ResolverOutput.model_validate_json(valid)
    assert model.approved is False
    assert model.rejection_reason == "Blockers are unresolvable"


# ── Schema Safety Tests (Ollama Grammar Compatibility) ───────────────────────
# These test the DEREFERENCED schemas — i.e. exactly what gets sent to providers.

def test_schema_is_ollama_safe_execution_plan():
    """Dereferenced ExecutionPlan schema must not contain anyOf/oneOf/allOf/$ref."""
    schema = dereference_schema(ExecutionPlan.model_json_schema())
    violations = _collect_forbidden(schema)
    assert violations == [], f"ExecutionPlan schema contains grammar-unsafe keys: {violations}"


def test_schema_is_ollama_safe_critique_verdict():
    """Dereferenced CritiqueVerdict schema must not contain anyOf/oneOf/allOf/$ref."""
    schema = dereference_schema(CritiqueVerdict.model_json_schema())
    violations = _collect_forbidden(schema)
    assert violations == [], f"CritiqueVerdict schema contains grammar-unsafe keys: {violations}"


def test_schema_is_ollama_safe_resolver_output():
    """Dereferenced ResolverOutput schema must not contain anyOf/oneOf/allOf/$ref."""
    schema = dereference_schema(ResolverOutput.model_json_schema())
    violations = _collect_forbidden(schema)
    assert violations == [], f"ResolverOutput schema contains grammar-unsafe keys: {violations}"


def test_dereference_actually_inlines():
    """Verify that dereference_schema removes $defs and inlines sub-schemas."""
    raw_schema = ExecutionPlan.model_json_schema()
    assert "$defs" in raw_schema, "Pydantic should generate $defs for nested models"
    
    dereferenced = dereference_schema(ExecutionPlan.model_json_schema())
    assert "$defs" not in dereferenced, "$defs should be removed after dereferencing"
    
    # steps.items should now have inline properties, not a $ref
    steps_items = dereferenced["properties"]["steps"]["items"]
    assert "$ref" not in steps_items, "steps.items should be inlined, not a $ref"
    assert "properties" in steps_items, "steps.items should have inline properties"
    assert "step_number" in steps_items["properties"], "Inlined step should have step_number"


# ── Provider Routing Tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_router_strategy():
    """Test that ProviderRouter gets the right strategy."""
    router_hf = ProviderRouter(provider="hf")
    from sumospace.providers.hf import HuggingFaceProvider
    router_hf._provider = HuggingFaceProvider()
    assert router_hf.get_output_strategy() == "xml"
    
    router_ollama = ProviderRouter(provider="ollama")
    from sumospace.providers.ollama import OllamaProvider
    router_ollama._provider = OllamaProvider()
    assert router_ollama.get_output_strategy() == "structured"


@pytest.mark.asyncio
async def test_xml_fallback_parsing():
    """Test that _complete_xml successfully extracts XML tags."""
    provider = MockXMLProvider()
    schema = ExecutionPlan.model_json_schema()
    
    result_str = await provider._complete_xml(user="test", schema=schema)
    result_dict = json.loads(result_str)
    
    assert "protocol_version" in result_dict
    assert result_dict["reasoning"] == "xml"


@pytest.mark.asyncio
async def test_router_complete_structured_routing():
    """Test that the router calls the correct underlying method based on capabilities."""
    router = ProviderRouter(provider="hf")
    router._provider = MockStructuredProvider()
    
    res = await router.complete_structured(user="test")
    data = json.loads(res)
    assert data["reasoning"] == "structured"
    
    router._provider = MockXMLProvider()
    res2 = await router.complete_structured(user="test", schema=ExecutionPlan.model_json_schema())
    data2 = json.loads(res2)
    assert data2["reasoning"] == "xml"
