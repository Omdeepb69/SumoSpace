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
logs_app      = typer.Typer(help="Explore session audit logs.")
snapshots_app = typer.Typer(help="Manage file snapshots and rollbacks.")
benchmark_app = typer.Typer(help="Run and report agent benchmarks.")
ingest_app    = typer.Typer(help="Ingest content from external sources.")

app.add_typer(logs_app,      name="logs")
app.add_typer(snapshots_app, name="snapshots")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(ingest_app,    name="ingest")
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
    no_committee: bool = typer.Option(False, "--no-committee", help="Skip committee, direct inference only"),
    plan_only: bool = typer.Option(False, "--plan-only", help="Plan but do not execute"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Skip RAG retrieval"),
    preset: str = typer.Option("", "--preset", help="Use a named preset: chat, chat-with-context, chat-stateless, coding, research, review"),
):
    """Execute a task using the full SumoKernel pipeline."""
    from sumospace.settings import SumoSettings
    from sumospace.kernel import SumoKernel

    if preset:
        preset_map = {
            "chat": SumoSettings.for_chat,
            "chat-with-context": SumoSettings.for_chat_with_context,
            "chat-stateless": SumoSettings.for_chat_stateless,
            "coding": SumoSettings.for_coding,
            "research": SumoSettings.for_research,
            "review": SumoSettings.for_review,
        }
        if preset not in preset_map:
            console.print(f"[red]Unknown preset '{preset}'. Choose: {list(preset_map.keys())}[/red]")
            raise typer.Exit(1)
        settings = preset_map[preset](
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
    else:
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
            committee_enabled=not no_committee,
            execution_enabled=not plan_only,
            rag_enabled=not no_rag,
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


@app.command("ingest-all")
def ingest_all(
    path: str = typer.Argument(..., help="Path to file or directory to ingest"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-ingestion of unchanged files"),
):
    """
    Ingest files into the Media RAG pipeline.
    Requires settings.media_enabled = True.
    """
    import asyncio
    from sumospace.kernel import SumoKernel

    async def _run():
        async with SumoKernel() as kernel:
            if not kernel.settings.media_enabled:
                console.print("[red]✗ Media features are disabled. Set media_enabled=True in settings.[/red]")
                return
            
            console.print(f"[cyan]Ingesting path: {path}[/cyan]")
            results = await kernel.ingest_media(path, force=force)
            
            skipped = sum(1 for r in results if r.skipped)
            added = sum(1 for r in results if not r.skipped and r.chunks_added > 0)
            errors = sum(1 for r in results if r.error)
            
            console.print(f"\n[green]✓ Ingestion complete[/green]")
            console.print(f"  Processed: {added} files")
            console.print(f"  Skipped (unchanged): {skipped} files")
            if errors:
                console.print(f"  [red]Errors: {errors} files[/red]")

    asyncio.run(_run())


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Text query or path to image/audio/video file"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
):
    """
    Search the Media RAG pipeline.
    Auto-detects query type (text, image, audio, video).
    Requires settings.media_enabled = True.
    """
    import asyncio
    from sumospace.kernel import SumoKernel

    async def _run():
        async with SumoKernel() as kernel:
            if not kernel.settings.media_enabled:
                console.print("[red]✗ Media features are disabled. Set media_enabled=True in settings.[/red]")
                return
            
            console.print(f"[cyan]Searching for: {query}[/cyan]")
            results = await kernel.search_media(query, top_k=top_k)
            
            if not results:
                console.print("[yellow]No results found.[/yellow]")
                return
                
            console.print("\n[bold]Top Results:[/bold]")
            for r in results:
                console.print(r.preview(max_chars=200))

    asyncio.run(_run())


# ─── snapshots ────────────────────────────────────────────────────────────────

@snapshots_app.command("list")
def snapshots_list(
    workspace: str = typer.Option(".", "--workspace", "-w"),
):
    """List all available snapshots."""
    from sumospace.settings import SumoSettings
    from sumospace.snapshots import SnapshotManager

    mgr = SnapshotManager(SumoSettings(workspace=workspace))
    snaps = mgr.list_snapshots()
    if not snaps:
        console.print("[yellow]No snapshots found.[/yellow]")
        return

    table = Table(title="Snapshots")
    table.add_column("Run ID",   style="cyan", no_wrap=True)
    table.add_column("DateTime", style="dim")
    table.add_column("Files",    justify="right")
    for s in snaps:
        table.add_row(s["run_id"], s.get("datetime", ""), str(s["files_count"]))
    console.print(table)


@snapshots_app.command("show")
def snapshots_show(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
    workspace: str = typer.Option(".", "--workspace", "-w"),
):
    """Show the diff for a specific snapshot."""
    from sumospace.settings import SumoSettings
    from sumospace.snapshots import SnapshotManager

    mgr = SnapshotManager(SumoSettings(workspace=workspace))
    snap = mgr.show_snapshot(run_id)
    if not snap:
        console.print(f"[red]Snapshot '{run_id}' not found.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Snapshot:[/bold] {snap['run_id']}")
    console.print(f"[dim]{snap.get('datetime', '')}[/dim]\n")
    for fe in snap.get("files", []):
        console.print(f"  [cyan]{fe['path']}[/cyan]")
        if fe.get("diff"):
            console.print(fe["diff"][:1000])


@app.command("rollback")
def rollback(
    run_id: str = typer.Argument(..., help="Run ID to rollback"),
    workspace: str = typer.Option(".", "--workspace", "-w"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Rollback all file mutations from a previous run."""
    from sumospace.settings import SumoSettings
    from sumospace.snapshots import SnapshotManager

    mgr = SnapshotManager(SumoSettings(workspace=workspace))
    snap = mgr.show_snapshot(run_id)
    if not snap:
        console.print(f"[red]Snapshot '{run_id}' not found.[/red]")
        raise typer.Exit(1)

    files = [f["path"] for f in snap.get("files", [])]
    console.print(f"[yellow]This will restore {len(files)} file(s) from run {run_id}:[/yellow]")
    for f in files:
        console.print(f"  {f}")

    if not yes:
        confirmed = typer.confirm("Proceed with rollback?")
        if not confirmed:
            console.print("[dim]Rollback cancelled.[/dim]")
            return

    restored = mgr.rollback(run_id)
    for r in restored:
        console.print(f"[green]✓[/green] {r}")
    console.print(f"\n[green]Rollback complete. {len(restored)} file(s) restored.[/green]")


# ─── benchmark ──────────────────────────────────────────────────────────────

@benchmark_app.command("run")
def benchmark_run(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Custom workspace (default: bundled fixtures)"),
    tasks: Optional[str] = typer.Option(None, "--tasks", "-t", help="Comma-separated task IDs"),
    mode: str = typer.Option("full", "--mode", "-m", help="Committee mode: full, plan_only, critique_only, disabled"),
    output: str = typer.Option(".", "--output", "-o", help="Output directory for reports"),
):
    """Run benchmarks against the bundled sample project (or a custom workspace)."""
    from sumospace.settings import SumoSettings
    from sumospace.benchmarks.runner import BenchmarkRunner
    from sumospace.benchmarks.report import BenchmarkReporter

    task_ids = [t.strip() for t in tasks.split(",")] if tasks else None

    async def _run():
        settings = SumoSettings()
        runner = BenchmarkRunner(
            settings=settings,
            workspace=workspace,
            task_ids=task_ids,
            committee_modes=[mode],
        )
        console.print(f"[cyan]Running benchmarks (mode={mode})...[/cyan]")
        results = await runner.run()
        reporter = BenchmarkReporter(results)
        json_path, md_path = reporter.save(output)
        console.print(f"[green]✓ Report saved:[/green] {md_path}")
        console.print(reporter.to_markdown())

    asyncio.run(_run())


@benchmark_app.command("compare")
def benchmark_compare(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    output: str = typer.Option(".", "--output", "-o"),
):
    """Compare all committee modes on benchmark tasks."""
    from sumospace.settings import SumoSettings
    from sumospace.benchmarks.runner import BenchmarkRunner
    from sumospace.benchmarks.report import BenchmarkReporter

    async def _run():
        settings = SumoSettings()
        runner = BenchmarkRunner(
            settings=settings,
            workspace=workspace,
            committee_modes=["full", "plan_only", "critique_only", "disabled"],
        )
        console.print("[cyan]Running full committee mode comparison...[/cyan]")
        results = await runner.run()
        reporter = BenchmarkReporter(results)
        _, md_path = reporter.save(output)
        console.print(f"[green]✓ Comparison report:[/green] {md_path}")
        console.print(reporter.to_markdown())

    asyncio.run(_run())


@benchmark_app.command("report")
def benchmark_report(
    json_path: str = typer.Argument(..., help="Path to a previously saved benchmark JSON"),
):
    """Render a Markdown report from a saved benchmark JSON file."""
    import json
    from sumospace.benchmarks.runner import BenchmarkResult, TaskResult
    from sumospace.benchmarks.report import BenchmarkReporter

    data = json.loads(Path(json_path).read_text())
    results = []
    for run in data.get("runs", []):
        task_results = [
            TaskResult(
                task_id=t["task_id"], task_name=t["task_name"],
                committee_mode=run["committee_mode"],
                passed=t["passed"], validation_reason=t["validation_reason"],
                duration_s=t["duration_s"], retries=t["retries"],
                tool_calls=t["tool_calls"], tool_failures=t["tool_failures"],
                rollback_triggered=t["rollback_triggered"], error=t.get("error", ""),
            )
            for t in run["tasks"]
        ]
        results.append(BenchmarkResult(
            run_id=run["run_id"],
            timestamp=0,
            committee_mode=run["committee_mode"],
            workspace="",
            task_results=task_results,
        ))
    reporter = BenchmarkReporter(results)
    console.print(reporter.to_markdown())


# ─── ingest (external loaders) ────────────────────────────────────────────────

@ingest_app.command("github")
def ingest_github(
    repo_url: str = typer.Argument(..., help="GitHub repository URL"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b"),
    workspace: str = typer.Option(".", "--workspace", "-w"),
):
    """Clone and ingest a GitHub repository."""
    from sumospace.settings import SumoSettings
    from sumospace.ingest import UniversalIngestor
    from sumospace.loaders.github import GitHubLoader

    async def _run():
        settings = SumoSettings(workspace=workspace)
        ingestor = UniversalIngestor(chroma_path=settings.chroma_base)
        await ingestor.initialize()
        loader = GitHubLoader(branch=branch)
        console.print(f"[cyan]Cloning {repo_url}...[/cyan]")
        count = await loader.load_into(repo_url, ingestor)
        console.print(f"[green]✓ Ingested {count} chunks from {repo_url}[/green]")

    asyncio.run(_run())


@ingest_app.command("youtube")
def ingest_youtube(
    url: str = typer.Argument(..., help="YouTube video URL"),
    workspace: str = typer.Option(".", "--workspace", "-w"),
):
    """Fetch and ingest a YouTube video transcript."""
    from sumospace.settings import SumoSettings
    from sumospace.ingest import UniversalIngestor
    from sumospace.loaders.youtube import YouTubeLoader

    async def _run():
        settings = SumoSettings(workspace=workspace)
        ingestor = UniversalIngestor(chroma_path=settings.chroma_base)
        await ingestor.initialize()
        loader = YouTubeLoader()
        console.print(f"[cyan]Fetching transcript for {url}...[/cyan]")
        chunks = await loader.load(url)
        if not chunks:
            console.print("[yellow]No transcript found.[/yellow]")
            return
        errors: list[str] = []
        await ingestor._embed_and_store(chunks, errors)
        console.print(f"[green]✓ Ingested {len(chunks)} chunks[/green]")
        if errors:
            console.print(f"[red]Errors: {errors}[/red]")

    asyncio.run(_run())


@ingest_app.command("web")
def ingest_web(
    url: str = typer.Argument(..., help="Web page or root URL to crawl"),
    crawl: bool = typer.Option(False, "--crawl", "-c", help="Crawl linked pages from root URL"),
    max_pages: int = typer.Option(10, "--max-pages", "-n"),
    workspace: str = typer.Option(".", "--workspace", "-w"),
):
    """Fetch and ingest a web page (or crawl a site)."""
    from sumospace.settings import SumoSettings
    from sumospace.ingest import UniversalIngestor
    from sumospace.loaders.web import WebLoader

    async def _run():
        settings = SumoSettings(workspace=workspace)
        ingestor = UniversalIngestor(chroma_path=settings.chroma_base)
        await ingestor.initialize()
        loader = WebLoader()
        if crawl:
            console.print(f"[cyan]Crawling up to {max_pages} pages from {url}...[/cyan]")
            chunks = await loader.crawl(url, max_pages=max_pages)
        else:
            console.print(f"[cyan]Fetching {url}...[/cyan]")
            chunks = await loader.load(url)
        errors: list[str] = []
        await ingestor._embed_and_store(chunks, errors)
        console.print(f"[green]✓ Ingested {len(chunks)} chunks[/green]")

    asyncio.run(_run())


# ─── watch ──────────────────────────────────────────────────────────────────

@app.command("watch")
def watch(
    path: str = typer.Argument(".", help="Directory to watch for file changes"),
    workspace: str = typer.Option(".", "--workspace", "-w"),
    debounce: float = typer.Option(2.0, "--debounce", "-d", help="Seconds to wait after last change before re-ingesting"),
):
    """
    Watch a directory for file changes and automatically re-ingest modified files.
    Press Ctrl+C to stop.
    """
    import time
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    from sumospace.settings import SumoSettings
    from sumospace.ingest import UniversalIngestor

    settings = SumoSettings(workspace=workspace)
    pending: dict[str, float] = {}  # path -> timestamp of last event

    class _Handler(FileSystemEventHandler):
        IGNORE = {"__pycache__", ".git", ".sumo_db", ".pytest_cache"}

        def on_modified(self, event):
            if event.is_directory:
                return
            if any(ex in event.src_path for ex in self.IGNORE):
                return
            pending[event.src_path] = time.monotonic()

        on_created = on_modified

    observer = Observer()
    observer.schedule(_Handler(), path, recursive=True)
    observer.start()
    console.print(f"[green]Watching[/green] [cyan]{path}[/cyan] — press Ctrl+C to stop")

    async def _ingest_pending():
        ingestor = UniversalIngestor(chroma_path=settings.chroma_base)
        await ingestor.initialize()
        while True:
            await asyncio.sleep(0.5)
            now = time.monotonic()
            ready = [p for p, t in list(pending.items()) if now - t >= debounce]
            for fpath in ready:
                del pending[fpath]
                try:
                    result = await ingestor.ingest_file(fpath, force=True)
                    if result.chunks_created > 0:
                        console.print(
                            f"[green]✓[/green] Re-ingested [cyan]{fpath}[/cyan] "
                            f"([dim]{result.chunks_created} chunks[/dim])"
                        )
                except Exception as e:
                    console.print(f"[red]✗ {fpath}: {e}[/red]")

    try:
        asyncio.run(_ingest_pending())
    except KeyboardInterrupt:
        observer.stop()
        console.print("\n[dim]Watch stopped.[/dim]")
    observer.join()


if __name__ == "__main__":
    app()
