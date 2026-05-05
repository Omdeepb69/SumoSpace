# sumospace/loaders/__init__.py
"""
Strategic Content Loaders
==========================
Five high-value loaders that normalize external content into
SumoSpace's existing ingest pipeline.

All loaders produce list[Chunk] compatible with UniversalIngestor.

Available:
    GitHubLoader    — clone and ingest a GitHub repository
    YouTubeLoader   — fetch video transcript
    WebLoader       — crawl and ingest web pages

Coming in v0.3:
    NotionLoader    — Notion workspace pages (requires OAuth)
"""
from sumospace.loaders.github import GitHubLoader
from sumospace.loaders.youtube import YouTubeLoader
from sumospace.loaders.web import WebLoader

__all__ = ["GitHubLoader", "YouTubeLoader", "WebLoader"]
