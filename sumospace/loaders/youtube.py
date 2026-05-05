# sumospace/loaders/youtube.py
"""
YouTube Transcript Loader
=========================
Fetches the transcript of a YouTube video and returns Chunks.

Requirements:
    pip install sumospace[loaders]
    (installs youtube-transcript-api)

Usage:
    loader = YouTubeLoader()
    chunks = await loader.load("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
"""
from __future__ import annotations

import re
from sumospace.ingest import Chunk, RecursiveTextSplitter


def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Assume url is already a video ID
    if re.match(r"^[A-Za-z0-9_-]{11}$", url):
        return url
    raise ValueError(f"Could not extract video ID from: {url}")


class YouTubeLoader:
    """
    Fetch and chunk a YouTube video transcript.

    Args:
        languages:    Preferred transcript languages in order. Default: ["en"].
        chunk_size:   Characters per chunk. Default: 600.
        overlap:      Overlap between chunks. Default: 80.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        chunk_size: int = 600,
        overlap: int = 80,
    ):
        self.languages = languages or ["en"]
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def load(self, url: str) -> list[Chunk]:
        """
        Fetch transcript for the given YouTube URL and return Chunks.

        Raises:
            ImportError: if youtube-transcript-api is not installed.
            ValueError:  if video ID cannot be extracted.
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "youtube-transcript-api is not installed. "
                "Run: pip install sumospace[loaders]"
            )

        video_id = _extract_video_id(url)

        # Fetch transcript (blocking call — run in executor in production)
        import asyncio
        loop = asyncio.get_event_loop()

        def _fetch():
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id, languages=self.languages)
            # FetchedTranscript is iterable, yields snippet dicts
            return [s for s in transcript]

        transcript_entries = await loop.run_in_executor(None, _fetch)

        # Reassemble into full text with timestamps
        full_text = " ".join(entry["text"] for entry in transcript_entries)

        # Chunk
        splitter = RecursiveTextSplitter(
            chunk_size=self.chunk_size, overlap=self.overlap
        )
        text_chunks = splitter.split(full_text)

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        return [
            Chunk(
                text=chunk,
                metadata={
                    "source": video_url,
                    "loader": "youtube",
                    "type": "transcript",
                    "video_id": video_id,
                    "chunk_index": i,
                },
            )
            for i, chunk in enumerate(text_chunks)
            if chunk.strip()
        ]
