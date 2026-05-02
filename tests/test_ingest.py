# tests/test_ingest.py

import pytest
from pathlib import Path
from sumospace.ingest import (
    RecursiveTextSplitter,
    TextLoader,
    JSONLoader,
    CSVLoader,
    PythonASTLoader,
    UniversalIngestor,
    Chunk,
)


class TestRecursiveTextSplitter:
    def setup_method(self):
        self.splitter = RecursiveTextSplitter(chunk_size=100, overlap=20)

    def test_short_text_not_split(self):
        text = "Hello world. This is short."
        chunks = self.splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_split(self):
        text = "word " * 100   # 500 chars
        chunks = self.splitter.split(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_empty_text(self):
        chunks = self.splitter.split("")
        assert chunks == [""] or chunks == []

    def test_paragraph_split_preferred(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = RecursiveTextSplitter(chunk_size=30, overlap=0).split(text)
        # Should split on paragraphs
        assert len(chunks) >= 2


class TestTextLoader:
    def setup_method(self):
        self.loader = TextLoader()

    def test_can_handle_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert self.loader.can_handle(f) is True

    def test_can_handle_md(self, tmp_path):
        f = tmp_path / "README.md"
        f.write_text("# Hi")
        assert self.loader.can_handle(f) is True

    def test_cannot_handle_py(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("pass")
        assert self.loader.can_handle(f) is False

    @pytest.mark.asyncio
    async def test_load_returns_chunks(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("This is a test document.\n" * 20)
        chunks = await self.loader.load(f)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.text.strip()
            assert chunk.metadata["source"] == str(f)

    @pytest.mark.asyncio
    async def test_load_with_unicode(self, tmp_path):
        f = tmp_path / "unicode.txt"
        f.write_text("こんにちは世界 — Hello World — مرحبا")
        chunks = await self.loader.load(f)
        assert len(chunks) >= 1


class TestJSONLoader:
    def setup_method(self):
        self.loader = JSONLoader()

    def test_can_handle_json(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text("{}")
        assert self.loader.can_handle(f) is True

    @pytest.mark.asyncio
    async def test_load_valid_json(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text('{"name": "test", "version": "1.0", "description": "A test config"}')
        chunks = await self.loader.load(f)
        assert len(chunks) >= 1
        assert "name" in chunks[0].text or "test" in chunks[0].text

    @pytest.mark.asyncio
    async def test_load_malformed_json_fallback(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not valid json at all")
        chunks = await self.loader.load(f)
        assert len(chunks) >= 1  # Should not crash


class TestCSVLoader:
    def setup_method(self):
        self.loader = CSVLoader()

    @pytest.mark.asyncio
    async def test_load_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
        chunks = await self.loader.load(f)
        assert len(chunks) == 2  # 2 data rows
        assert "Alice" in chunks[0].text
        assert "age" in chunks[0].text.lower() or "30" in chunks[0].text

    @pytest.mark.asyncio
    async def test_csv_metadata(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\nval1,val2\n")
        chunks = await self.loader.load(f)
        assert chunks[0].metadata["row"] == 0
        assert chunks[0].metadata["type"] == "csv"


class TestPythonASTLoader:
    def setup_method(self):
        self.loader = PythonASTLoader()

    def test_can_handle_py(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("pass")
        assert self.loader.can_handle(f) is True

    def test_cannot_handle_js(self, tmp_path):
        f = tmp_path / "app.js"
        f.write_text("const x = 1;")
        assert self.loader.can_handle(f) is False

    @pytest.mark.asyncio
    async def test_load_function(self, tmp_path):
        f = tmp_path / "funcs.py"
        f.write_text(
            '"""Module docstring."""\n\n'
            'def add(a: int, b: int) -> int:\n'
            '    """Add two numbers."""\n'
            '    return a + b\n'
        )
        chunks = await self.loader.load(f)
        texts = [c.text for c in chunks]
        assert any("add" in t for t in texts)
        assert any("Module docstring" in t for t in texts)

    @pytest.mark.asyncio
    async def test_load_class(self, tmp_path):
        f = tmp_path / "cls.py"
        f.write_text(
            'class MyService:\n'
            '    """A service class."""\n\n'
            '    def __init__(self):\n'
            '        self.x = 0\n\n'
            '    def process(self) -> None:\n'
            '        pass\n'
        )
        chunks = await self.loader.load(f)
        texts = [c.text for c in chunks]
        assert any("MyService" in t for t in texts)
        assert any("process" in t for t in texts)

    @pytest.mark.asyncio
    async def test_syntax_error_fallback(self, tmp_path):
        f = tmp_path / "broken.py"
        f.write_text("def broken(\n    # missing closing paren and body\n")
        chunks = await self.loader.load(f)
        # Should not crash; falls back to TextLoader
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_imports_extracted(self, tmp_path):
        f = tmp_path / "imports.py"
        f.write_text(
            "import os\nimport sys\nfrom pathlib import Path\n\ndef noop(): pass\n"
        )
        chunks = await self.loader.load(f)
        texts = " ".join(c.text for c in chunks)
        assert "import os" in texts or "Imports" in texts


from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_embedder():
    with patch("chromadb.utils.embedding_functions.DefaultEmbeddingFunction") as mock:
        def fake_embed(texts):
            return [[0.1] * 384 for _ in texts]
        instance = MagicMock()
        instance.side_effect = fake_embed
        mock.return_value = instance
        yield instance

@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_embedder")
class TestUniversalIngestor:
    async def test_initialize(self, tmp_chroma):
        ingestor = UniversalIngestor(
            chroma_path=tmp_chroma,
            embedding_provider="local",
        )
        await ingestor.initialize()
        assert ingestor._client is not None
        assert ingestor._collection is not None
        assert ingestor._embedder is not None

    async def test_ingest_text_file(self, tmp_path, tmp_chroma):
        f = tmp_path / "hello.txt"
        f.write_text("The quick brown fox jumps over the lazy dog. " * 10)
        ingestor = UniversalIngestor(chroma_path=tmp_chroma)
        await ingestor.initialize()
        result = await ingestor.ingest_file(f)
        assert result.chunks_created >= 1
        assert result.source == str(f)
        assert not result.errors

    async def test_ingest_python_file(self, tmp_path, tmp_chroma):
        f = tmp_path / "module.py"
        f.write_text(
            '"""Module."""\n\ndef hello():\n    """Say hello."""\n    return "hi"\n'
        )
        ingestor = UniversalIngestor(chroma_path=tmp_chroma)
        await ingestor.initialize()
        result = await ingestor.ingest_file(f)
        assert result.chunks_created >= 1
        assert result.loader_used == "python_ast"

    async def test_ingest_directory(self, tmp_workspace, tmp_chroma):
        ingestor = UniversalIngestor(chroma_path=tmp_chroma)
        await ingestor.initialize()
        results = await ingestor.ingest_directory(str(tmp_workspace))
        total = sum(r.chunks_created for r in results)
        assert total >= 1
        assert len(results) >= 3  # main.py, utils.py, README.md at minimum

    async def test_query_after_ingest(self, tmp_path, tmp_chroma):
        f = tmp_path / "doc.txt"
        f.write_text(
            "The authenticate function handles user login and session management. "
            "It validates credentials against the database and returns a JWT token."
        )
        ingestor = UniversalIngestor(chroma_path=tmp_chroma)
        await ingestor.initialize()
        await ingestor.ingest_file(f)
        results = await ingestor.query("user authentication and login", top_k=3)
        assert len(results) >= 1
        assert "score" in results[0]
        assert results[0]["score"] > 0

    async def test_ingest_nonexistent_file(self, tmp_chroma):
        ingestor = UniversalIngestor(chroma_path=tmp_chroma)
        await ingestor.initialize()
        result = await ingestor.ingest_file("/nonexistent/path/file.txt")
        assert result.chunks_created == 0
        assert len(result.errors) > 0

    async def test_chunk_deduplication(self, tmp_path, tmp_chroma):
        """Ingest the same file twice — chunk IDs are content-based, so no duplicates."""
        f = tmp_path / "data.txt"
        f.write_text("Unique content that should only appear once in the store.")
        ingestor = UniversalIngestor(chroma_path=tmp_chroma)
        await ingestor.initialize()
        r1 = await ingestor.ingest_file(f)
        r2 = await ingestor.ingest_file(f)
        results = await ingestor.query("unique content", top_k=10)
        # Content-based IDs mean upsert deduplicates automatically
        texts = [r["text"] for r in results]
        # Should not have duplicate text entries
        assert len(set(texts)) == len(texts)

    async def test_incremental_ingest_skips_unchanged(self, tmp_path, tmp_chroma):
        """Verify that ingesting the same file twice skips the second time."""
        f = tmp_path / "incremental.txt"
        f.write_text("This content will be hashed.")
        ingestor = UniversalIngestor(chroma_path=tmp_chroma)
        await ingestor.initialize()
        
        # First ingest
        r1 = await ingestor.ingest_file(f)
        assert r1.chunks_created > 0
        assert r1.loader_used != "skipped (unchanged)"
        
        # Second ingest (unchanged)
        r2 = await ingestor.ingest_file(f)
        assert r2.chunks_created == 0
        assert r2.loader_used == "skipped (unchanged)"

    async def test_incremental_ingest_force(self, tmp_path, tmp_chroma):
        """Verify that force=True re-ingests even if unchanged."""
        f = tmp_path / "force.txt"
        f.write_text("Force re-ingest content.")
        ingestor = UniversalIngestor(chroma_path=tmp_chroma)
        await ingestor.initialize()
        
        await ingestor.ingest_file(f)
        
        # Second ingest with force=True
        r2 = await ingestor.ingest_file(f, force=True)
        assert r2.chunks_created > 0
        assert r2.loader_used != "skipped (unchanged)"
