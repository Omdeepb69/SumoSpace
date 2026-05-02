# tests/test_rag.py

import pytest
from sumospace.ingest import UniversalIngestor
from sumospace.rag import RAGPipeline, CrossEncoderReranker, RetrievedChunk, RAGResult


@pytest.fixture
async def populated_ingestor(tmp_path, tmp_chroma):
    """Ingestor pre-populated with some test documents."""
    docs = {
        "auth.py": (
            '"""Authentication module."""\n\n'
            'def authenticate(username: str, password: str) -> str:\n'
            '    """Validate credentials and return JWT token."""\n'
            '    # Hash password and check against DB\n'
            '    return "jwt_token_here"\n'
        ),
        "database.py": (
            '"""Database connection and ORM utilities."""\n\n'
            'class DatabaseManager:\n'
            '    """Manages PostgreSQL connections with connection pooling."""\n\n'
            '    def connect(self, url: str) -> None:\n'
            '        """Open a pooled connection to the database."""\n'
            '        pass\n'
        ),
        "README.md": (
            "# MyApp\n\n"
            "A web application with JWT authentication and PostgreSQL storage.\n\n"
            "## Setup\n\nRun `pip install -r requirements.txt` to install dependencies.\n"
        ),
    }
    for name, content in docs.items():
        (tmp_path / name).write_text(content)

    ingestor = UniversalIngestor(chroma_path=tmp_chroma)
    await ingestor.initialize()
    for name in docs:
        await ingestor.ingest_file(tmp_path / name)

    return ingestor


@pytest.mark.asyncio
class TestCrossEncoderReranker:
    async def test_rerank_basic(self):
        reranker = CrossEncoderReranker()
        chunks = [
            RetrievedChunk(
                text="JWT authentication validates user credentials.",
                metadata={}, vector_score=0.7,
            ),
            RetrievedChunk(
                text="Database connection pooling improves performance.",
                metadata={}, vector_score=0.65,
            ),
            RetrievedChunk(
                text="User login uses JWT tokens for session management.",
                metadata={}, vector_score=0.6,
            ),
        ]
        reranked = await reranker.rerank("JWT user authentication", chunks)
        assert len(reranked) == 3
        # All chunks should have rerank scores
        for c in reranked:
            assert c.rerank_score is not None
        # JWT-relevant chunks should score higher
        assert reranked[0].rerank_score >= reranked[-1].rerank_score

    async def test_rerank_empty_list(self):
        reranker = CrossEncoderReranker()
        result = await reranker.rerank("query", [])
        assert result == []

    async def test_rerank_single_chunk(self):
        reranker = CrossEncoderReranker()
        chunk = RetrievedChunk(text="some text", metadata={}, vector_score=0.8)
        result = await reranker.rerank("query", [chunk])
        assert len(result) == 1
        assert result[0].rerank_score is not None


@pytest.mark.asyncio
class TestRAGPipeline:
    async def test_initialize(self, tmp_chroma, populated_ingestor):
        pipeline = RAGPipeline(ingestor=populated_ingestor)
        await pipeline.initialize()
        assert pipeline._reranker is not None

    async def test_retrieve_relevant_chunks(self, populated_ingestor):
        pipeline = RAGPipeline(ingestor=populated_ingestor, top_k_final=3)
        await pipeline.initialize()
        result = await pipeline.retrieve("How does JWT authentication work?")
        assert isinstance(result, RAGResult)
        assert len(result.chunks) >= 1
        assert result.context
        assert result.query == "How does JWT authentication work?"

    async def test_retrieve_with_reranking(self, populated_ingestor):
        pipeline = RAGPipeline(ingestor=populated_ingestor, use_reranker=True)
        await pipeline.initialize()
        result = await pipeline.retrieve("JWT token authentication")
        assert result.used_reranker is True

    async def test_retrieve_without_reranking(self, populated_ingestor):
        pipeline = RAGPipeline(ingestor=populated_ingestor, use_reranker=False)
        await pipeline.initialize()
        result = await pipeline.retrieve("database connection")
        assert result.used_reranker is False

    async def test_retrieve_returns_scored_chunks(self, populated_ingestor):
        pipeline = RAGPipeline(ingestor=populated_ingestor)
        await pipeline.initialize()
        result = await pipeline.retrieve("PostgreSQL database")
        for chunk in result.chunks:
            assert hasattr(chunk, "final_score")
            assert isinstance(chunk.final_score, float)

    async def test_context_not_exceeds_max_chars(self, populated_ingestor):
        pipeline = RAGPipeline(
            ingestor=populated_ingestor,
            max_context_chars=500,
        )
        await pipeline.initialize()
        result = await pipeline.retrieve("anything")
        assert len(result.context) <= 600  # small buffer for truncation marker

    async def test_build_prompt(self, populated_ingestor):
        pipeline = RAGPipeline(ingestor=populated_ingestor)
        await pipeline.initialize()
        rag_result = await pipeline.retrieve("authentication")
        prompt = pipeline.build_prompt(
            query="How does auth work?",
            rag_result=rag_result,
            task_description="Answer using the codebase context.",
        )
        assert "RETRIEVED CONTEXT" in prompt
        assert "How does auth work?" in prompt
        assert "Answer using the codebase context." in prompt

    async def test_force_no_rerank(self, populated_ingestor):
        pipeline = RAGPipeline(ingestor=populated_ingestor, use_reranker=True)
        await pipeline.initialize()
        result = await pipeline.retrieve("database", force_no_rerank=True)
        assert result.used_reranker is False

    async def test_total_candidates_tracked(self, populated_ingestor):
        pipeline = RAGPipeline(
            ingestor=populated_ingestor,
            top_k_candidates=10,
            top_k_final=2,
        )
        await pipeline.initialize()
        result = await pipeline.retrieve("any query")
        assert result.total_candidates >= 0


class TestRetrievedChunk:
    def test_final_score_uses_rerank_when_available(self):
        chunk = RetrievedChunk(text="t", metadata={}, vector_score=0.7, rerank_score=0.9)
        assert chunk.final_score == 0.9

    def test_final_score_falls_back_to_vector(self):
        chunk = RetrievedChunk(text="t", metadata={}, vector_score=0.7)
        assert chunk.final_score == 0.7

    def test_to_context_block_includes_source(self):
        chunk = RetrievedChunk(
            text="Some text content",
            metadata={"source": "/path/to/file.py"},
            vector_score=0.8,
            rerank_score=0.95,
        )
        block = chunk.to_context_block()
        assert "/path/to/file.py" in block
        assert "Some text content" in block
        assert "0.95" in block

    def test_to_context_block_with_page(self):
        chunk = RetrievedChunk(
            text="PDF content",
            metadata={"source": "doc.pdf", "page": 3},
            vector_score=0.75,
        )
        block = chunk.to_context_block()
        assert "page 3" in block
