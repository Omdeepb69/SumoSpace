from typing import Literal, Any
from pydantic import BaseModel, Field


def dereference_schema(schema: dict) -> dict:
    """Inline all $ref references in a JSON schema, removing $defs.
    
    Ollama/llama.cpp grammar engines struggle with $ref and $defs.
    This function produces a fully-inlined, flat schema that is
    safe for constrained decoding.
    """
    defs = schema.pop("$defs", {})
    
    def _resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node["$ref"]  # e.g. "#/$defs/ExecutionStep"
                ref_name = ref_path.split("/")[-1]
                if ref_name in defs:
                    # Return a deep copy of the resolved definition
                    resolved = _resolve(dict(defs[ref_name]))
                    return resolved
                return node
            return {k: _resolve(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [_resolve(item) for item in node]
        return node
    
    return _resolve(schema)

# ─── Data Models ─────────────────────────────────────────────────────────────

class ExecutionStep(BaseModel):
    step_number: int = Field(..., description="The sequential step number, starting from 1.")
    tool: str = Field(..., description="The name of the tool to execute.")
    description: str = Field(..., description="A clear description of what this step accomplishes.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="The parameters to pass to the tool.")
    critical: bool = Field(default=False, description="If True, failure of this step halts the entire plan.")
    expected_output: str = Field(default="", description="What success looks like for this step.")


class ExecutionPlan(BaseModel):
    protocol_version: Literal["1.0"] = Field(default="1.0", description="Protocol version for the plan.")
    reasoning: str = Field(..., description="Brief explanation of the overall approach.")
    steps: list[ExecutionStep] = Field(..., description="The ordered list of steps to execute.")
    estimated_duration_s: int = Field(default=30, description="Estimated duration in seconds to complete the plan.")
    risks: list[str] = Field(default_factory=list, description="Potential risks identified during planning.")


class CritiqueVerdict(BaseModel):
    protocol_version: Literal["1.0"] = Field(default="1.0", description="Protocol version for the critique.")
    verdict: Literal["approve", "revise", "reject"] = Field(..., description="The decision on the plan.")
    reason: str = Field(..., description="One sentence reason for the verdict.")
    risks: list[str] = Field(default_factory=list, description="Potential risks in the plan.")
    blockers: list[str] = Field(default_factory=list, description="Must-fix issues that make the plan unsafe.")
    suggestions: list[str] = Field(default_factory=list, description="Improvements to the plan.")


class ResolverOutput(BaseModel):
    protocol_version: Literal["1.0"] = Field(default="1.0", description="Protocol version for the resolver.")
    approved: bool = Field(..., description="True if the plan is finally approved to execute, False otherwise.")
    has_revision: bool = Field(default=False, description="True if the resolver made revisions to the original plan.")
    revised_steps: list[ExecutionStep] = Field(default_factory=list, description="The revised execution steps, if has_revision is True.")
    approval_notes: str = Field(default="", description="Notes from the resolver regarding the approval.")
    rejection_reason: str = Field(default="", description="Reason for rejecting the plan, if not approved.")
