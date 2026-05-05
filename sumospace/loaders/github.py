# sumospace/loaders/github.py
"""
GitHub Repository Loader
========================
Clones a GitHub repository into a temp directory and feeds it
into SumoSpace's UniversalIngestor.

Usage:
    loader = GitHubLoader()
    chunks = await loader.load("https://github.com/owner/repo")
    # or: ingestor = loader.get_ingestor(settings)
    # await ingestor.ingest_directory(loader.clone_path)

Requirements: git must be installed on the system.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from sumospace.ingest import Chunk, TextLoader, RecursiveTextSplitter


class GitHubLoader:
    """
    Clone a GitHub (or any git) repository and return chunks.

    Args:
        branch:      Branch to clone. None = default branch.
        depth:       Shallow clone depth. 1 = latest commit only (fast).
        extensions:  File extensions to ingest. None = all supported types.
        exclude:     Directory names to skip.
    """

    DEFAULT_EXCLUDE = {
        "__pycache__", ".git", "node_modules", ".venv", "venv",
        "dist", "build", ".pytest_cache", "*.egg-info",
    }

    def __init__(
        self,
        branch: str | None = None,
        depth: int = 1,
        extensions: set[str] | None = None,
        exclude: set[str] | None = None,
    ):
        self.branch = branch
        self.depth = depth
        self.extensions = extensions
        self.exclude = exclude or self.DEFAULT_EXCLUDE
        self._clone_dir: str | None = None

    async def load(self, repo_url: str) -> list[Chunk]:
        """
        Clone repo_url and return Chunks ready for embedding.
        The temp directory is cleaned up after this call.
        """
        tmp = tempfile.mkdtemp(prefix="sumo_github_")
        try:
            await self._clone(repo_url, tmp)
            chunks = await self._ingest_dir(Path(tmp), repo_url)
            return chunks
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def load_into(self, repo_url: str, ingestor) -> int:
        """
        Clone repo_url and ingest directly into a UniversalIngestor.
        Returns total chunks ingested.
        """
        tmp = tempfile.mkdtemp(prefix="sumo_github_")
        try:
            await self._clone(repo_url, tmp)
            results = await ingestor.ingest_directory(
                tmp,
                extensions=self.extensions,
                exclude_patterns=list(self.exclude),
            )
            return sum(r.chunks_created for r in results)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def _clone(self, repo_url: str, dest: str):
        cmd = ["git", "clone", f"--depth={self.depth}"]
        if self.branch:
            cmd += ["--branch", self.branch]
        cmd += [repo_url, dest]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"git clone failed (exit {proc.returncode}): {stderr.decode()}"
            )

    async def _ingest_dir(self, path: Path, source_url: str) -> list[Chunk]:
        from sumospace.ingest import (
            PythonASTLoader, TextLoader, JSONLoader, CSVLoader, PDFLoader
        )

        loaders = [PythonASTLoader(), JSONLoader(), CSVLoader(), PDFLoader(), TextLoader()]

        chunks: list[Chunk] = []
        for file in sorted(path.rglob("*")):
            if file.is_dir():
                continue
            if any(ex in file.parts for ex in self.exclude):
                continue
            if self.extensions and file.suffix.lower() not in self.extensions:
                continue

            loader = next((l for l in loaders if l.can_handle(file)), loaders[-1])
            try:
                file_chunks = await loader.load(file)
                for c in file_chunks:
                    c.metadata["github_url"] = source_url
                    c.metadata["repo_path"] = str(file.relative_to(path))
                chunks.extend(file_chunks)
            except Exception:
                continue

        return chunks
