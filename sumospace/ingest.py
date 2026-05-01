# sumospace/ingest.py

"""
Universal Ingestion Engine
===========================
Digests any file type into ChromaDB-ready chunks with rich metadata.
Supports: Python/JS/C++ (AST), PDF, plain text, markdown, JSON, CSV.
Optional: Audio (Whisper), Video (frames + vision), Images (captioning).

Default embedding: BAAI/bge-base-en-v1.5 via sentence-transformers (local, no API key).
"""

from __future__ import annotations

import asyncio
import ast
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import chromadb
from chromadb.config import Settings
from rich.console import Console

console = Console()


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single ingestion unit with content and provenance metadata."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def id(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


@dataclass
class IngestionResult:
    """Result returned after ingesting a document or directory."""
    source: str
    chunks_created: int
    loader_used: str
    duration_ms: float
    errors: list[str] = field(default_factory=list)


# ─── Embedding Providers ──────────────────────────────────────────────────────

class EmbeddingProvider:
    """Base class for embedding backends."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Sentence-Transformers — fully offline, no API key, no internet at inference.

    Default model: BAAI/bge-base-en-v1.5
      - 768 dimensions
      - ~440MB download (cached after first use in ~/.cache/huggingface)
      - Excellent for semantic search

    Alternative models (all free, no key):
      - "BAAI/bge-small-en-v1.5"                   (~130MB, faster)
      - "BAAI/bge-large-en-v1.5"                   (~1.3GB, more accurate)
      - "sentence-transformers/all-MiniLM-L6-v2"   (~90MB, very fast, CPU-friendly)
    """
    name = "local"

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self._model_name = model_name
        self._model = None
        self._use_fallback = False
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        except (ImportError, Exception):
            self._use_fallback = True

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._use_fallback:
            return await self._fallback_embed(texts)
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(None, self._model.encode, texts)
        return vectors.tolist()

    async def _fallback_embed(self, texts: list[str]) -> list[list[float]]:
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            fn = DefaultEmbeddingFunction()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, fn, texts)
            return [list(v) for v in result]
        except Exception as e:
            from sumospace.exceptions import IngestError
            raise IngestError(
                f"All embedding backends failed: {e}\n"
                "Install sentence-transformers: pip install sumospace\n"
                "Or set embedding_provider='google'/'openai' with the matching API key."
            ) from e


class GoogleEmbeddingProvider(EmbeddingProvider):
    """
    Google Generative AI Embeddings.
    Requires: pip install sumospace[gemini] + GOOGLE_API_KEY.
    """

    def __init__(self, api_key: str | None = None):
        import google.generativeai as genai
        key = api_key or os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=key)
        self._genai = genai
        self._model = "models/text-embedding-004"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        results = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]
            embeddings = await loop.run_in_executor(
                None,
                lambda b=batch: [
                    self._genai.embed_content(
                        model=self._model,
                        content=t,
                        task_type="retrieval_document",
                    )["embedding"]
                    for t in b
                ],
            )
            results.extend(embeddings)
        return results


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI Embeddings.
    Requires: pip install sumospace[openai] + OPENAI_API_KEY.
    """

    def __init__(self, api_key: str | None = None):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model="text-embedding-3-large",
            input=texts,
        )
        return [item.embedding for item in response.data]


# ─── Text Splitters ───────────────────────────────────────────────────────────

class RecursiveTextSplitter:
    """
    Recursive character splitter with overlap.
    Priority: paragraphs → sentences → words → characters.
    """
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        return list(self._split_recursive(text, self.SEPARATORS))

    def _split_recursive(self, text: str, separators: list[str]) -> Iterator[str]:
        if len(text) <= self.chunk_size:
            yield text
            return

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else []

        parts = (
            text.split(sep)
            if sep
            else [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        )
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    yield from self._split_recursive(current, remaining_seps)
                    overlap_text = current[-self.overlap:] if self.overlap > 0 and len(current) > self.overlap else (current if self.overlap > 0 else "")
                    current = overlap_text + (sep if overlap_text else "") + part
                else:
                    yield from self._split_recursive(part, remaining_seps)
                    current = ""
        if current:
            yield from self._split_recursive(current, remaining_seps)


# ─── Loaders ─────────────────────────────────────────────────────────────────

class BaseLoader:
    name: str = "base"

    async def load(self, path: Path) -> list[Chunk]:
        raise NotImplementedError

    def can_handle(self, path: Path) -> bool:
        raise NotImplementedError


class TextLoader(BaseLoader):
    name = "text"
    EXTENSIONS = {".txt", ".md", ".rst", ".log", ".yaml", ".yml", ".toml", ".ini", ".cfg"}

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in self.EXTENSIONS

    async def load(self, path: Path) -> list[Chunk]:
        text = path.read_text(encoding="utf-8", errors="replace")
        splitter = RecursiveTextSplitter()
        return [
            Chunk(
                text=chunk,
                metadata={"source": str(path), "loader": self.name,
                           "type": "text", "file": path.name},
            )
            for chunk in splitter.split(text) if chunk.strip()
        ]


class JSONLoader(BaseLoader):
    name = "json"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in {".json", ".jsonl"}

    async def load(self, path: Path) -> list[Chunk]:
        import json
        text = path.read_text()
        try:
            data = json.loads(text)
            serialized = json.dumps(data, indent=2)
        except json.JSONDecodeError:
            serialized = text
        splitter = RecursiveTextSplitter(chunk_size=1024)
        return [
            Chunk(text=c, metadata={"source": str(path), "loader": self.name, "type": "json"})
            for c in splitter.split(serialized) if c.strip()
        ]


class CSVLoader(BaseLoader):
    name = "csv"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".csv"

    async def load(self, path: Path) -> list[Chunk]:
        import csv
        chunks = []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text = " | ".join(f"{k}: {v}" for k, v in row.items())
                chunks.append(Chunk(
                    text=text,
                    metadata={"source": str(path), "loader": self.name,
                               "row": i, "type": "csv"},
                ))
        return chunks


class PythonASTLoader(BaseLoader):
    """
    AST-aware Python loader.
    Extracts: module docstring, class definitions, function signatures + docstrings,
    imports, and decorators. Preserves structural context.
    """
    name = "python_ast"

    def can_handle(self, path: Path) -> bool:
        return path.suffix == ".py"

    async def load(self, path: Path) -> list[Chunk]:
        import ast
        source = path.read_text(encoding="utf-8", errors="replace")
        chunks = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return await TextLoader().load(path)

        if (module_doc := ast.get_docstring(tree)):
            chunks.append(Chunk(
                text=f"[Module: {path.name}] {module_doc}",
                metadata={"source": str(path), "loader": self.name,
                           "type": "module_doc", "file": path.name},
            ))

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.extend(self._extract_function(node, path, source))
            elif isinstance(node, ast.ClassDef):
                chunks.extend(self._extract_class(node, path, source))

        imports = [
            ast.unparse(n) for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        if imports:
            chunks.append(Chunk(
                text=f"[Imports: {path.name}]\n" + "\n".join(imports),
                metadata={"source": str(path), "loader": self.name,
                           "type": "imports", "file": path.name},
            ))

        return chunks

    def _extract_function(self, node, path, source) -> list[Chunk]:
        chunks = []
        try:
            func_source = ast.get_source_segment(source, node) or ""
            docstring = ast.get_docstring(node) or ""
            args = [a.arg for a in node.args.args]
            decorators = [ast.unparse(d) for d in node.decorator_list]
            summary = (
                f"[Function: {node.name} in {path.name}]\n"
                f"Decorators: {', '.join(decorators) or 'none'}\n"
                f"Args: {', '.join(args)}\n"
                f"Docstring: {docstring}\n"
                f"Source (lines {node.lineno}-{node.end_lineno}):\n{func_source[:800]}"
            )
            chunks.append(Chunk(
                text=summary,
                metadata={
                    "source": str(path), "loader": self.name, "type": "function",
                    "name": node.name, "line": node.lineno, "file": path.name,
                },
            ))
        except Exception:
            pass
        return chunks

    def _extract_class(self, node, path, source) -> list[Chunk]:
        chunks = []
        try:
            docstring = ast.get_docstring(node) or ""
            methods = [
                n.name for n in ast.walk(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            bases = [ast.unparse(b) for b in node.bases]
            summary = (
                f"[Class: {node.name} in {path.name}]\n"
                f"Bases: {', '.join(bases) or 'none'}\n"
                f"Methods: {', '.join(methods)}\n"
                f"Docstring: {docstring}"
            )
            chunks.append(Chunk(
                text=summary,
                metadata={
                    "source": str(path), "loader": self.name, "type": "class",
                    "name": node.name, "line": node.lineno, "file": path.name,
                },
            ))
        except Exception:
            pass
        return chunks


class TreeSitterLoader(BaseLoader):
    """
    Tree-sitter based loader for JS, TS, C++.
    Requires: pip install sumospace[code]
    """
    name = "tree_sitter"
    LANGUAGE_MAP = {
        ".js": "javascript", ".ts": "javascript",
        ".jsx": "javascript", ".tsx": "javascript",
        ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
        ".c": "cpp", ".h": "cpp", ".hpp": "cpp",
    }

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in self.LANGUAGE_MAP

    async def load(self, path: Path) -> list[Chunk]:
        try:
            from tree_sitter import Language, Parser
        except ImportError:
            console.print("[yellow]tree-sitter not installed. pip install sumospace[code][/yellow]")
            return await TextLoader().load(path)

        source = path.read_bytes()
        lang_name = self.LANGUAGE_MAP[path.suffix.lower()]
        try:
            import importlib
            lang_mod = importlib.import_module(f"tree_sitter_{lang_name}")
            language = Language(lang_mod.language())
            parser = Parser(language)
            tree = parser.parse(source)
            root = tree.root_node
        except Exception:
            return await TextLoader().load(path)

        chunks: list[Chunk] = []
        self._walk_node(root, source, path, chunks)
        return chunks or await TextLoader().load(path)

    def _walk_node(self, node, source: bytes, path: Path, chunks: list):
        interesting = {
            "function_declaration", "method_definition", "class_declaration",
            "function_definition", "export_statement",
        }
        if node.type in interesting:
            text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            if len(text) > 20:
                chunks.append(Chunk(
                    text=f"[{node.type} in {path.name}]\n{text[:1000]}",
                    metadata={
                        "source": str(path), "loader": self.name,
                        "type": node.type, "file": path.name,
                        "line": node.start_point[0],
                    },
                ))
        for child in node.children:
            self._walk_node(child, source, path, chunks)


class PDFLoader(BaseLoader):
    """
    Semantic PDF loader.
    Requires: pip install sumospace[pdf]
    Strategy: pdfplumber for structured tables + text extraction.
    """
    name = "pdf"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    async def load(self, path: Path) -> list[Chunk]:
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pip install sumospace[pdf]")

        chunks = []
        splitter = RecursiveTextSplitter(chunk_size=600, overlap=80)

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    table_text = "\n".join(
                        " | ".join(str(cell or "") for cell in row)
                        for row in table if row
                    )
                    if table_text.strip():
                        chunks.append(Chunk(
                            text=f"[Table on page {page_num + 1} of {path.name}]\n{table_text}",
                            metadata={"source": str(path), "loader": self.name,
                                      "type": "table", "page": page_num + 1},
                        ))

                text = page.extract_text() or ""
                for chunk in splitter.split(text):
                    if chunk.strip():
                        chunks.append(Chunk(
                            text=chunk,
                            metadata={"source": str(path), "loader": self.name,
                                      "type": "text", "page": page_num + 1},
                        ))

        return chunks


# ─── Main Ingestion Engine ────────────────────────────────────────────────────

# Module-level lock registry: keyed by chroma_path to prevent concurrent upsert races
_collection_locks: dict[str, asyncio.Lock] = {}


def _get_collection_lock(chroma_path: str) -> asyncio.Lock:
    """Return a per-collection-path asyncio.Lock (created lazily)."""
    if chroma_path not in _collection_locks:
        _collection_locks[chroma_path] = asyncio.Lock()
    return _collection_locks[chroma_path]


class UniversalIngestor:
    """
    Digests any file or directory into ChromaDB.

    Usage:
        ingestor = UniversalIngestor(chroma_path=".sumo_db")
        await ingestor.initialize()
        await ingestor.ingest_directory("./src")
        await ingestor.ingest_file("./docs/README.md")
    """

    LOADERS: list[type[BaseLoader]] = [
        PythonASTLoader,
        TreeSitterLoader,
        PDFLoader,
        JSONLoader,
        CSVLoader,
        TextLoader,  # fallback — must be last
    ]

    def __init__(
        self,
        chroma_path: str = ".sumo_db",
        collection_name: str = "sumospace",
        embedding_provider: str = "local",        # local is default — zero API key
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 50,
        max_chunks: int | None = None,
    ):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_chunks = max_chunks

        self._client: chromadb.ClientAPI | None = None
        self._collection = None
        self._embedder: EmbeddingProvider | None = None
        self._loaders = [cls() for cls in self.LOADERS]
        self._lock = _get_collection_lock(chroma_path)

    async def initialize(self):
        """Set up ChromaDB client and embedding provider."""
        self._client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Embedder selection — local is the default, cloud providers are explicit opt-in
        if self.embedding_provider in ("local", "sentence-transformers", "st"):
            self._embedder = LocalEmbeddingProvider(model_name=self.embedding_model)
        elif self.embedding_provider == "google":
            self._embedder = GoogleEmbeddingProvider()
        elif self.embedding_provider == "openai":
            self._embedder = OpenAIEmbeddingProvider()
        else:
            self._embedder = LocalEmbeddingProvider(model_name=self.embedding_model)

    async def ingest_file(self, path: str | Path) -> IngestionResult:
        """Ingest a single file."""
        import time
        path = Path(path)
        start = time.monotonic()

        loader = self._get_loader(path)
        chunks = await loader.load(path)

        errors: list[str] = []
        await self._embed_and_store(chunks, errors)

        return IngestionResult(
            source=str(path),
            chunks_created=len(chunks),
            loader_used=loader.name,
            duration_ms=(time.monotonic() - start) * 1000,
            errors=errors,
        )

    async def ingest_directory(
        self,
        directory: str | Path,
        extensions: set[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> list[IngestionResult]:
        """
        Recursively ingest all files in a directory.

        Args:
            directory:        Root directory to scan.
            extensions:       Whitelist of file extensions (e.g., {".py", ".md"}).
            exclude_patterns: Patterns to exclude (e.g., ["__pycache__", ".git"]).
        """
        exclude_patterns = exclude_patterns or [
            "__pycache__", ".git", "node_modules", ".venv", "venv",
            "dist", "build", ".pytest_cache", ".mypy_cache",
        ]

        results = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in exclude_patterns]

            for filename in files:
                filepath = Path(root) / filename

                if exclude_patterns and any(p in str(filepath) for p in exclude_patterns):
                    continue
                if extensions and filepath.suffix.lower() not in extensions:
                    continue

                try:
                    result = await self.ingest_file(filepath)
                    results.append(result)
                    console.print(
                        f"[green]✓[/green] {filepath} "
                        f"([cyan]{result.chunks_created} chunks[/cyan])"
                    )
                except Exception as e:
                    results.append(IngestionResult(
                        source=str(filepath), chunks_created=0,
                        loader_used="error", duration_ms=0,
                        errors=[str(e)],
                    ))

        return results

    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over ingested content."""
        embeddings = await self._embedder.embed([query_text])
        query_kwargs: dict = {
            "query_embeddings": embeddings,
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where
        results = self._collection.query(**query_kwargs)
        return [
            {
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,  # cosine distance → similarity
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    async def _embed_and_store(self, chunks: list[Chunk], errors: list[str]):
        """Embed chunks in batches and upsert into ChromaDB with locking and quota."""
        from sumospace.exceptions import QuotaExceededError

        async with self._lock:
            # Quota check before upsert
            if self.max_chunks is not None:
                current = self._collection.count()
                if current + len(chunks) > self.max_chunks:
                    raise QuotaExceededError(
                        current=current,
                        attempted=len(chunks),
                        limit=self.max_chunks,
                    )

            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                try:
                    texts = [c.text for c in batch]
                    embeddings = await self._embedder.embed(texts)
                    self._collection.upsert(
                        ids=[c.id for c in batch],
                        documents=texts,
                        embeddings=embeddings,
                        metadatas=[c.metadata for c in batch],
                    )
                except Exception as e:
                    errors.append(str(e))

    def _get_loader(self, path: Path) -> BaseLoader:
        for loader in self._loaders:
            if loader.can_handle(path):
                return loader
        return self._loaders[-1]  # TextLoader fallback
