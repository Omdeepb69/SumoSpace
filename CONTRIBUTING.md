# Contributing to SumoSpace

Thank you for investing your time in contributing to our project!

## Architecture Overview
SumoSpace is built around an autonomous, deliberative agent architecture:
- **Classifier**: Determines if a task requires web search or codebase RAG.
- **Committee**: A 3-agent group (Planner, Critic, Resolver) that agrees on a safe execution plan before taking action.
- **Kernel**: The main orchestrator that manages lifecycles, memory, scopes, and tools.

## Development Setup
1. Clone the repository: `git clone https://github.com/omdeepb69/sumospace.git`
2. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest`

## Adding a New Provider

To add support for a new LLM provider (e.g. Cohere, Groq):
1. Subclass `BaseProvider` in `sumospace/providers.py`.
2. Implement `initialize()`, `complete()`, and `stream()`.
3. Register your provider in the `PROVIDERS` dict at the top of `providers.py` with a string key (e.g., `"cohere"`).
4. Add any required optional dependencies to `pyproject.toml` (e.g., `cohere = ["cohere>=5.0.0"]`).
5. Add settings fields to `SumoSettings` in `sumospace/settings.py` using the `SUMO_` prefix (e.g., `SUMO_COHERE_API_KEY`).
6. Add a test in `tests/test_providers.py` using the mock pattern from existing tests to verify your implementation without hitting real APIs.
7. Document the addition in `CHANGELOG.md` under the `### Added` section.

## Adding a New Tool

To give the agent new capabilities (e.g. interacting with Jira or AWS):
1. Subclass `BaseTool` in `sumospace/tools.py`.
2. Define the tool's `name`, `description`, `schema`, and `tags` as class attributes.
3. Implement `async def run(self, **kwargs) -> ToolResult`. Ensure you gracefully catch errors and return `ToolResult(success=False, error=str(e))`.
4. Register the tool in `ToolRegistry._register_defaults()` in `sumospace/tools.py`.
5. Add a test in `tests/test_tools.py`.

## Submitting Pull Requests
- Ensure all new features have accompanying unit/integration tests.
- Run `pytest` to verify nothing is broken.
- Provide a clear, descriptive PR title and summary.
