import pytest
import jsonschema
from unittest.mock import MagicMock, patch
from sumospace.tools import BaseTool, ToolRegistry, ToolResult


class ValidTool(BaseTool):
    name = "valid_tool"
    schema = {
        "type": "object",
        "properties": {
            "param1": {"type": "string"}
        },
        "required": ["param1"]
    }
    async def run(self, param1, **kwargs):
        return ToolResult(tool=self.name, success=True, output=f"param1 is {param1}")


class InvalidTool(BaseTool):
    name = "invalid_tool"
    # missing schema entirely, relies on base class schema


@pytest.mark.asyncio
async def test_validate_params_success():
    tool = ValidTool()
    valid, msg = tool.validate_params({"param1": "test"})
    assert valid is True
    assert msg == ""


@pytest.mark.asyncio
async def test_validate_params_failure():
    tool = ValidTool()
    valid, msg = tool.validate_params({"param1": 123})
    assert valid is False
    assert "123 is not of type 'string'" in msg


def test_describe():
    tool = ValidTool()
    desc = tool.describe()
    assert desc["name"] == "valid_tool"
    assert "schema" in desc
    assert "tags" in desc


@pytest.mark.asyncio
async def test_execute_validates_params():
    registry = ToolRegistry()
    registry.register(ValidTool())
    
    # Valid parameters
    result = await registry.execute("valid_tool", param1="test")
    assert result.success is True
    assert result.output == "param1 is test"
    
    # Invalid parameters
    result = await registry.execute("valid_tool", param1=123)
    assert result.success is False
    assert result.metadata.get("validation_error") is True


@patch("importlib.metadata.entry_points")
def test_discover_plugins(mock_entry_points):
    # Mocking entry point discovery
    class MockEntryPoint:
        name = "mock_tool"
        def load(self):
            return ValidTool
    
    # Support both Python 3.8 and 3.10+ entry_points API
    mock_eps = MagicMock()
    mock_eps.select.return_value = [MockEntryPoint()]
    mock_eps.get.return_value = [MockEntryPoint()]
    mock_entry_points.return_value = mock_eps
    
    registry = ToolRegistry()
    
    assert registry.get("valid_tool") is not None
    assert registry.get("valid_tool").name == "valid_tool"
