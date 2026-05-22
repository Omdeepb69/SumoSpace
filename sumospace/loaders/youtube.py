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
            # Handle both old (0.x) and new (1.0+) youtube-transcript-api versions
            try:
                # New API (1.0+): instance method returning FetchedTranscript
                api = YouTubeTranscriptApi()
                transcript = api.fetch(video_id, languages=self.languages)
                # FetchedTranscriptSnippet has .text attribute, not dict access
                return [{"text": snippet.text} for snippet in transcript]
            except (TypeError, AttributeError):
                pass
            # Old API (0.x): class method returning list of dicts
            return YouTubeTranscriptApi.get_transcript(video_id, languages=self.languages)

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

    async def download_media(self, url: str, output_dir: str = ".", extract_audio: bool = False, max_duration: int | None = None) -> str:
        """
        Download video or audio from YouTube using yt-dlp.
        Returns the path to the downloaded file.
        """
        try:
            import yt_dlp
        except ImportError:
            raise ImportError(
                "yt-dlp is not installed. "
                "Run: pip install sumospace[loaders]"
            )

        video_id = _extract_video_id(url)
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        import asyncio
        loop = asyncio.get_event_loop()

        def _download():
            import os
            os.makedirs(output_dir, exist_ok=True)
            ydl_opts = {
                'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
                'quiet': True,
                'no_warnings': True,
            }
            if extract_audio:
                ydl_opts['format'] = 'bestaudio/best'
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }]
            else:
                ydl_opts['format'] = 'worst[ext=mp4]' # Use worst format for fast benchmark downloads

            if max_duration:
                def filter_func(info, *args, **kwargs):
                    if info.get('duration', 0) > max_duration:
                        return 'Video too long'
                    return None
                ydl_opts['match_filter'] = filter_func

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                if extract_audio:
                    return f"{output_dir}/{info['id']}.mp3"
                else:
                    return f"{output_dir}/{info['id']}.{info['ext']}"

        return await loop.run_in_executor(None, _download)
