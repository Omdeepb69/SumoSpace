# tests/test_loaders.py
"""Tests for strategic content loaders."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sumospace.loaders.youtube import YouTubeLoader, _extract_video_id
from sumospace.loaders.web import WebLoader


# ── YouTube ────────────────────────────────────────────────────────────────

def test_extract_video_id_standard():
    assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_short():
    assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_direct():
    assert _extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_invalid():
    with pytest.raises(ValueError):
        _extract_video_id("https://example.com/notayoutube")


@pytest.mark.asyncio
async def test_youtube_loader_returns_chunks():
    fake_transcript = [
        {"text": "Hello world", "start": 0.0, "duration": 2.0},
        {"text": "This is a test", "start": 2.0, "duration": 2.0},
    ]
    # fetch() is now an instance method in youtube-transcript-api>=0.6
    # We mock the class so instantiation returns a mock with fetch() configured.
    mock_instance = MagicMock()
    mock_instance.fetch.return_value = iter(fake_transcript)
    with patch("youtube_transcript_api.YouTubeTranscriptApi", return_value=mock_instance):
        loader = YouTubeLoader(chunk_size=100)
        chunks = await loader.load("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    assert len(chunks) > 0
    assert chunks[0].metadata["loader"] == "youtube"
    assert chunks[0].metadata["video_id"] == "dQw4w9WgXcQ"
    assert "Hello world" in chunks[0].text


@pytest.mark.asyncio
async def test_youtube_loader_missing_dep():
    with patch.dict("sys.modules", {"youtube_transcript_api": None}):
        loader = YouTubeLoader()
        with pytest.raises(ImportError, match="youtube-transcript-api"):
            await loader.load("https://www.youtube.com/watch?v=test")


# ── Web ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_web_loader_returns_chunks():
    fake_html = """
    <html>
    <head><title>Test Page</title></head>
    <body>
      <p>This is the main content of the test page. It has enough text to form a chunk.</p>
      <script>var x = 1;</script>
    </body>
    </html>
    """
    mock_response = MagicMock()
    mock_response.text = fake_html
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        loader = WebLoader(chunk_size=200)
        chunks = await loader.load("https://example.com")

    assert len(chunks) > 0
    assert chunks[0].metadata["source"] == "https://example.com"
    assert chunks[0].metadata["loader"] == "web"
    assert "Test Page" in chunks[0].metadata["title"]
    # Script content should be removed
    assert "var x = 1" not in chunks[0].text


@pytest.mark.asyncio
async def test_web_loader_missing_dep():
    with patch.dict("sys.modules", {"httpx": None, "bs4": None}):
        loader = WebLoader()
        with pytest.raises(ImportError, match="httpx"):
            await loader.load("https://example.com")


# ── GitHub ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_github_loader_clone_failure():
    from sumospace.loaders.github import GitHubLoader
    loader = GitHubLoader()

    async def _bad_clone(url, dest):
        raise RuntimeError("git clone failed (exit 128): Repository not found")

    with patch.object(loader, "_clone", side_effect=RuntimeError("git clone failed")):
        with pytest.raises(RuntimeError, match="git clone failed"):
            await loader.load("https://github.com/invalid/repo")


@pytest.mark.asyncio
async def test_github_loader_processes_files(tmp_path):
    """GitHub loader should process Python files from a local directory."""
    from sumospace.loaders.github import GitHubLoader

    # Write a sample file to a temp dir
    (tmp_path / "sample.py").write_text("def hello():\n    return 'world'\n")

    loader = GitHubLoader()

    async def _fake_clone(url, dest):
        import shutil
        shutil.copytree(str(tmp_path), dest, dirs_exist_ok=True)

    with patch.object(loader, "_clone", side_effect=_fake_clone):
        chunks = await loader.load("https://github.com/fake/repo")

    assert len(chunks) > 0
    assert any("hello" in c.text for c in chunks)
    assert chunks[0].metadata.get("github_url") == "https://github.com/fake/repo"
