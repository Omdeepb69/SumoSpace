# sumospace/cli.py

"""
Sumo CLI
=========
Usage:
    sumo run "Fix the broken tests in tests/"
    sumo run "Refactor src/auth.py" --provider ollama --model mistral
    sumo run "List all Python files" --dry-run
    sumo ingest ./src
    sumo chat
    sumo info
"""

from typing import Optional
import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="sumo",
    help="SumoSpace — Auto-everything multi-agent execution environment.",
    add_completion=False,
)
logs_app = typer.Typer(help="Explore session audit logs.")
app.add_typer(logs_app, name="logs")
console = Console()


# ─── logs ───────────────────────────────────────────────────────────────────

@logs_app.command("list")
def logs_list(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of sessions to show"),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Workspace root"),
):
    """List recent execution sessions."""
    from sumospace.settings import SumoSettings
    from sumospace.audit import AuditLogger
    
    settings = SumoSettings(workspace=workspace)
    logger = AuditLogger(settings)
    sessions = logger.list(limit=limit)
    
    if not sessions:
        console.print("[yellow]No audit logs found.[/yellow]")
        return
        
    table = Table(title="Recent Sessions")
    table.add_column("Session ID", style="cyan", no_wrap=True)
    table.add_column("Task", style="white")
    table.add_column("Intent", style="dim")
    table.add_column("Result", justify="center")
    table.add_column("Duration", justify="right")
    table.add_column("Timestamp", style="dim")
    
    for s in sessions:
        status = "[green]✓[/green]" if s["success"] else "[red]✗[/red]"
        task = s["task"][:60] + ("..." if len(s["task"]) > 60 else "")
        table.add_row(
            s["session_id"][:12],
            task,
            s["intent"],
            status,
            f"{s['duration_ms']:.0f}ms",
            s["timestamp"].split("T")[0],
        )
    
    console.print(table)

@logs_app.command("show")
def logs_show(
    session_id: str = typer.Argument(..., help="Session ID to show"),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Workspace root"),
):
    """Show detailed trace for a specific session."""
    from sumospace.settings import SumoSettings
    from sumospace.audit import AuditLogger
    import json
    
    settings = SumoSettings(workspace=workspace)
    logger = AuditLogger(settings)
    session = logger.show(session_id)
    
    if not session:
        # Try truncated ID
        sessions = logger.list(limit=100)
        session = next((s for s in sessions if s["session_id"].startswith(session_id)), None)
        
    if not session:
        console.print(f"[red]Session {session_id} not found.[/red]")
        return
        
    console.print(f"[bold cyan]Session Details: {session['session_id']}[/bold cyan]")
    console.print(f"[bold]Task:[/bold] {session['task']}")
    console.print(f"[bold]Result:[/bold] {'[green]Success[/green]' if session['success'] else '[red]Failed[/red]'}")
    console.print(f"[bold]Intent:[/bold] {session['intent']}")
    console.print(f"[bold]Duration:[/bold] {session['duration_ms']:.0f}ms")
    console.print(f"[bold]Timestamp:[/bold] {session['timestamp']}")
    
    if session.get("error"):
        console.print(f"\n[bold red]Error:[/bold red]\n{session['error']}")
        
    table = Table(title="\nExecution Steps", box=None)
    table.add_column("#", style="dim")
    table.add_column("Tool", style="green")
    table.add_column("Description")
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right")
    
    for st in session.get("steps", []):
        status = "[green]✓[/green]" if st["success"] else "[red]✗[/red]"
        table.add_row(
            str(st["step_number"]),
            st["tool"],
            st["description"],
            status,
            f"{st['duration_ms']:.0f}ms",
        )
    console.print(table)
    
    if session.get("final_answer"):
        console.print("\n[bold]Final Answer:[/bold]")
        console.print(session["final_answer"])

@logs_app.command("search")
def logs_search(
    query: str = typer.Argument(..., help="Substring to search in tasks"),
    limit: int = typer.Option(10, "--limit", "-l"),
    workspace: str = typer.Option(".", "--workspace", "-w"),
):
    """Search sessions by task content."""
    from sumospace.settings import SumoSettings
    from sumospace.audit import AuditLogger
    
    settings = SumoSettings(workspace=workspace)
    logger = AuditLogger(settings)
    results = logger.search(query, limit=limit)
    
    if not results:
        console.print(f"[yellow]No sessions found matching: {query}[/yellow]")
        return
        
    table = Table(title=f"Search Results: '{query}'")
    table.add_column("Session ID", style="cyan")
    table.add_column("Task")
    table.add_column("Result", justify="center")
    
    for r in results:
        status = "[green]✓[/green]" if r["success"] else "[red]✗[/red]"
        table.add_row(r["session_id"][:12], r["task"][:80], status)
    
    console.print(table)

@logs_app.command("stats")
def logs_stats(
    workspace: str = typer.Option(".", "--workspace", "-w"),
):
    """Show aggregated session statistics."""
    from sumospace.settings import SumoSettings
    from sumospace.audit import AuditLogger
    
    settings = SumoSettings(workspace=workspace)
    logger = AuditLogger(settings)
    stats = logger.stats()
    
    if not stats:
        console.print("[yellow]No stats index found. Run some tasks first.[/yellow]")
        return
        
    console.print("[bold cyan]Session Statistics[/bold cyan]")
    console.print(f"Total Sessions:     {stats['total_sessions']}")
    success_rate = (stats['successful_sessions'] / stats['total_sessions']) * 100
    console.print(f"Success Rate:       {success_rate:.1f}% ({stats['successful_sessions']}/{stats['total_sessions']})")
    avg_duration = stats['total_duration_ms'] / stats['total_sessions'] / 1000
    console.print(f"Avg Duration:       {avg_duration:.1f}s")
    
    table = Table(title="\nTop Intents", box=None)
    table.add_column("Intent", style="magenta")
    table.add_column("Count", justify="right")
    for intent, count in sorted(stats.get("intent_usage", {}).items(), key=lambda x: x[1], reverse=True):
        table.add_row(intent, str(count))
    console.print(table)
    
    table = Table(title="\nTool Usage", box=None)
    table.add_column("Tool", style="green")
    table.add_column("Success", style="green", justify="right")
    table.add_column("Fail", style="red", justify="right")
    table.add_column("Rate", justify="right")
    
    for tool, usage in sorted(stats.get("tool_usage", {}).items(), key=lambda x: x[1]["success"] + x[1]["fail"], reverse=True):
        total = usage["success"] + usage["fail"]
        rate = (usage["success"] / total) * 100
        table.add_row(tool, str(usage["success"]), str(usage["fail"]), f"{rate:.0f}%")
    console.print(table)

@logs_app.command("export")
def logs_export(
    session_id: str = typer.Argument(..., help="Session ID to export"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    workspace: str = typer.Option(".", "--workspace", "-w"),
):
    """Export a session trace to a Markdown report."""
    from sumospace.settings import SumoSettings
    from sumospace.audit import AuditLogger
    
    settings = SumoSettings(workspace=workspace)
    logger = AuditLogger(settings)
    content = logger.export(session_id)
    
    if not content:
        console.print(f"[red]Session {session_id} not found.[/red]")
        return
        
    out_path = output or Path(f"session_{session_id[:8]}.md")
    out_path.write_text(content, encoding="utf-8")
    console.print(f"[green]✓ Exported report to {out_path}[/green]")


# ─── run ────────────────────────────────────────────────────────────────────

@app.command()
def run(
    task: str = typer.Argument(..., help="Task or instruction to execute"),
    provider: str = typer.Option("hf", "--provider", "-p",
                                 help="Model provider: hf | ollama | auto | gemini | openai | anthropic"),
    model: str = typer.Option("default", "--model", "-m",
                               help="Model name or alias (default resolves per-provider)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only, no actual execution"),
    workspace: str = typer.Option(".", "--workspace", "-w",
                                  help="Root directory for file operations"),
    chroma_path: str = typer.Option(".sumo_db", "--db", help="ChromaDB path"),
    no_consensus: bool = typer.Option(False, "--no-consensus",
                                      help="Skip committee consensus (faster, less safe)"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Verbose output"),
    user_id: str = typer.Option("", "--user-id", help="Scope: validated user identifier"),
    session_id: str = typer.Option("", "--session-id", help="Scope: session identifier"),
    project_id: str = typer.Option("", "--project-id", help="Scope: project identifier"),
    scope_level: str = typer.Option("user", "--scope-level",
                                    help="Isolation level: user | session | project"),
    max_chunks: int = typer.Option(0, "--max-chunks",
                                   help="Quota: max chunks per scope (0 = unlimited)"),
):
    """Execute a task using the full SumoKernel pipeline."""
    from sumospace.settings import SumoSettings
    from sumospace.kernel import SumoKernel

    settings = SumoSettings(
        provider=provider,
        model=model,
        dry_run=dry_run,
        workspace=workspace,
        chroma_base=chroma_path,
        require_consensus=not no_consensus,
        verbose=verbose,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        scope_level=scope_level,
        max_chunks_per_scope=max_chunks or None,
    )

    async def _run():
        async with SumoKernel(settings=settings) as kernel:
            trace = await kernel.run(task)
            if trace.final_answer:
                console.print("\n[bold]Result:[/bold]")
                console.print(trace.final_answer)
            if not trace.success:
                console.print(f"\n[red]Error: {trace.error}[/red]")
                sys.exit(1)

    asyncio.run(_run())


# ─── ingest ─────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest"),
    provider: str = typer.Option("local", "--embed-provider", "-e",
                                 help="Embedding provider: local | google | openai"),
    model: str = typer.Option("BAAI/bge-base-en-v1.5", "--embed-model",
                              help="Embedding model name"),
    chroma_path: str = typer.Option(".sumo_db", "--db", help="ChromaDB path"),
    collection: str = typer.Option("sumospace", "--collection", "-c"),
    user_id: str = typer.Option("", "--user-id", help="Scope: validated user identifier"),
    session_id: str = typer.Option("", "--session-id", help="Scope: session identifier"),
    project_id: str = typer.Option("", "--project-id", help="Scope: project identifier"),
    scope_level: str = typer.Option("user", "--scope-level",
                                    help="Isolation level: user | session | project"),
    max_chunks: int = typer.Option(0, "--max-chunks",
                                   help="Quota: max chunks per scope (0 = unlimited)"),
):
    """Ingest a file or directory into the RAG knowledge base."""
    from sumospace.ingest import UniversalIngestor

    resolved_path = chroma_path
    if user_id:
        from sumospace.scope import ScopeManager
        scope = ScopeManager(chroma_base=chroma_path, level=scope_level)
        resolved_path = scope.resolve(
            user_id=user_id, session_id=session_id, project_id=project_id,
        )

    async def _ingest():
        ingestor = UniversalIngestor(
            chroma_path=resolved_path,
            collection_name=collection,
            embedding_provider=provider,
            embedding_model=model,
            max_chunks=max_chunks or None,
        )
        await ingestor.initialize()

        p = Path(path)
        if p.is_dir():
            results = await ingestor.ingest_directory(str(p))
            total = sum(r.chunks_created for r in results)
            errors = sum(len(r.errors) for r in results)
            console.print(f"\n[green]✓ Ingested {len(results)} files, {total} chunks[/green]")
            if errors:
                console.print(f"[yellow]{errors} files had errors[/yellow]")
        elif p.is_file():
            result = await ingestor.ingest_file(str(p))
            console.print(f"[green]✓ {result.chunks_created} chunks from {path}[/green]")
        else:
            console.print(f"[red]Path not found: {path}[/red]")
            sys.exit(1)

    asyncio.run(_ingest())


# ─── chat ────────────────────────────────────────────────────────────────────

@app.command()
def chat(
    provider: str = typer.Option("hf", "--provider", "-p"),
    model: str = typer.Option("default", "--model", "-m"),
    workspace: str = typer.Option(".", "--workspace", "-w"),
    chroma_path: str = typer.Option(".sumo_db", "--db"),
):
    """Start an interactive chat session (conversational, no tool execution)."""
    from sumospace.settings import SumoSettings
    from sumospace.kernel import SumoKernel

    settings = SumoSettings(
        provider=provider,
        model=model,
        workspace=workspace,
        chroma_base=chroma_path,
        verbose=False,
    )

    async def _chat():
        console.print("[bold cyan]SumoSpace Chat[/bold cyan]  (Ctrl+C to exit)\n")
        async with SumoKernel(settings=settings) as kernel:
            while True:
                try:
                    user_input = typer.prompt("You")
                    if user_input.strip().lower() in {"exit", "quit", "q"}:
                        break
                    response = await kernel.chat(user_input)
                    console.print(f"\n[bold cyan]Sumo:[/bold cyan] {response}\n")
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[dim]Exiting chat.[/dim]")
                    break

    asyncio.run(_chat())


# ─── search ─────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(..., help="Semantic search query"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    chroma_path: str = typer.Option(".sumo_db", "--db"),
    provider: str = typer.Option("local", "--embed-provider"),
    model: str = typer.Option("BAAI/bge-base-en-v1.5", "--embed-model"),
    user_id: str = typer.Option("", "--user-id", help="Scope: validated user identifier"),
    session_id: str = typer.Option("", "--session-id", help="Scope: session identifier"),
    project_id: str = typer.Option("", "--project-id", help="Scope: project identifier"),
    scope_level: str = typer.Option("user", "--scope-level",
                                    help="Isolation level: user | session | project"),
):
    """Semantic search over the ingested knowledge base."""
    from sumospace.ingest import UniversalIngestor

    resolved_path = chroma_path
    if user_id:
        from sumospace.scope import ScopeManager
        scope = ScopeManager(chroma_base=chroma_path, level=scope_level)
        resolved_path = scope.resolve(
            user_id=user_id, session_id=session_id, project_id=project_id,
        )

    async def _search():
        ingestor = UniversalIngestor(
            chroma_path=resolved_path,
            embedding_provider=provider,
            embedding_model=model,
        )
        await ingestor.initialize()
        results = await ingestor.query(query, top_k=top_k)
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "unknown")
            score = r["score"]
            console.print(f"\n[bold][{i}][/bold] {source} (score: {score:.3f})")
            console.print(r["text"][:300])

    asyncio.run(_search())


# ─── info ───────────────────────────────────────────────────────────────────

@app.command()
def info():
    """Show installed capabilities and default configuration."""
    table = Table(title="SumoSpace — Installed Capabilities", show_header=True)
    table.add_column("Feature", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    # Core
    table.add_row("Core", "✓ installed", "Local inference, embeddings, ChromaDB")

    # HF
    try:
        import transformers
        table.add_row("HuggingFace", "✓ available", f"transformers {transformers.__version__}")
    except ImportError:
        table.add_row("HuggingFace", "✗ missing", "pip install sumospace")

    # Ollama
    try:
        import httpx
        import asyncio
        async def _check():
            try:
                async with httpx.AsyncClient(timeout=2) as c:
                    await c.get("http://localhost:11434/api/tags")
                return True
            except Exception:
                return False
        running = asyncio.run(_check())
        table.add_row("Ollama", "✓ running" if running else "⚠ installed (not running)",
                      "http://localhost:11434")
    except ImportError:
        table.add_row("Ollama", "✗ missing", "pip install sumospace")

    # Gemini
    try:
        import google.generativeai
        table.add_row("Gemini", "✓ installed", "pip install sumospace[gemini]")
    except ImportError:
        table.add_row("Gemini", "✗ not installed", "pip install sumospace[gemini]")

    # OpenAI
    try:
        import openai
        table.add_row("OpenAI", "✓ installed", f"openai {openai.__version__}")
    except ImportError:
        table.add_row("OpenAI", "✗ not installed", "pip install sumospace[openai]")

    # Anthropic
    try:
        import anthropic
        table.add_row("Anthropic", "✓ installed", f"anthropic {anthropic.__version__}")
    except ImportError:
        table.add_row("Anthropic", "✗ not installed", "pip install sumospace[anthropic]")

    # Code
    try:
        import tree_sitter
        table.add_row("Code (tree-sitter)", "✓ installed", "AST-aware code parsing")
    except ImportError:
        table.add_row("Code (tree-sitter)", "✗ not installed", "pip install sumospace[code]")

    # PDF
    try:
        import pdfplumber
        table.add_row("PDF", "✓ installed", "pdfplumber")
    except ImportError:
        table.add_row("PDF", "✗ not installed", "pip install sumospace[pdf]")

    # Audio
    try:
        import whisper
        table.add_row("Audio (Whisper)", "✓ installed", "Local ASR, no API key")
    except ImportError:
        table.add_row("Audio (Whisper)", "✗ not installed", "pip install sumospace[audio]")

    # Browser
    try:
        import playwright
        table.add_row("Browser (Playwright)", "✓ installed", "Headless Chromium")
    except ImportError:
        table.add_row("Browser (Playwright)", "✗ not installed", "pip install sumospace[browser]")

    console.print(table)
    console.print("\n[dim]Default provider: hf (HuggingFace Phi-3-mini, no API key)[/dim]")
    console.print("[dim]Default embeddings: BAAI/bge-base-en-v1.5 (local, no API key)[/dim]")


if __name__ == "__main__":
    app()
