# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-05-02

### Added
- **Real-Time Streaming**: Added `kernel.stream_run()` to yield execution progress step-by-step, ideal for rendering UI updates without blocking until the end of the task.
- **Execution Plan Caching**: Repeated tasks with identical contexts now bypass the committee deliberation phase, resulting in near-instant execution starts and significantly lower token usage.
- **Multi-User Isolation**: Added `ScopeManager` to strictly isolate RAG context, audit logs, and memory between users (`user_id`), sessions (`session_id`), or projects (`project_id`).
- **Telemetry Support**: Full OpenTelemetry (OTLP) tracing added for latency profiling. Enable via `SUMO_TELEMETRY_ENABLED=true`.
- **Intelligent Context Truncation**: When codebase size exceeds token limits, the kernel now predictably sacrifices raw RAG results before discarding your core task instructions or recent memory.

### Fixed
- **Provider Reliability**: If your primary LLM provider (e.g. OpenAI) times out or hits rate limits (429/503), the kernel now gracefully falls back to a pre-initialized secondary provider (e.g. local HuggingFace) mid-run.
- **Malformed Plan Recovery**: Local models occasionally output invalid JSON. The Committee now uses a 3-attempt retry loop with dynamic temperature bumping to self-correct formatting errors.
- **Concurrency Deadlocks**: Fixed race conditions and file lock contention in `AuditLogger` and `MemoryManager` when running multiple concurrent kernels.
- **Token Estimation Accuracy**: Fixed token estimation logic to prevent unexpected context window overflow crashes during large retrievals.
- **Memory Leaks**: Proper cleanup of resources during `kernel.shutdown()`.

### Changed
- The `kernel.run()` method signature was updated to guarantee the return of a standard `ExecutionTrace`, safely capturing critical errors (`ConsensusFailedError`, `ExecutionHaltedError`) rather than throwing raw exceptions to your application.
- Renamed CLI audit subcommands and their `AuditLogger` counterparts for exact 1:1 mapping (e.g., `sumo logs list` now calls `audit.list()`).
- Deprecated `KernelConfig` in favor of the unified `SumoSettings`.

### Security
- Added robust path traversal safeguards to `ScopeManager` to prevent cross-tenant data leaks.
- Added dependency isolation to `UniversalIngestor` to prevent arbitrary code execution during codebase parsing.
