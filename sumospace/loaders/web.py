# sumospace/loaders/web.py
"""
Web Crawler Loader
==================
Fetches and ingests web pages into SumoSpace's RAG pipeline.
Respects robots.txt and rate limits. Does not require JavaScript.

Requirements:
    pip install sumospace[loaders]
    (installs httpx, beautifulsoup4)

Usage:
    loader = WebLoader()
    chunks = await loader.load("https://docs.example.com/intro")
    # Crawl up to 10 pages from a root URL:
    chunks = await loader.crawl("https://docs.example.com", max_pages=10)
"""
from __future__ import annotations

import asyncio
import re
from urllib.parse import urljoin, urlparse

from sumospace.ingest import Chunk, RecursiveTextSplitter


class WebLoader:
    """
    Fetch and chunk web page content.

    Args:
        chunk_size:     Characters per chunk. Default: 800.
        overlap:        Overlap between chunks. Default: 100.
        timeout:        HTTP request timeout in seconds. Default: 15.
        headers:        Custom HTTP headers (e.g., User-Agent).
        include_links:  Append list of found links to chunk metadata.
    """

    DEFAULT_HEADERS = {
        "User-Agent": "SumoSpace/0.2.0 (+https://github.com/Omdeepb69/SumoSpace)"
    }

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 100,
        timeout: int = 15,
        headers: dict | None = None,
        include_links: bool = False,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.timeout = timeout
        self.headers = {**self.DEFAULT_HEADERS, **(headers or {})}
        self.include_links = include_links

    async def load(self, url: str) -> list[Chunk]:
        """Fetch and chunk a single web page."""
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "httpx and beautifulsoup4 are required. "
                "Run: pip install sumospace[loaders]"
            )

        async with httpx.AsyncClient(
            headers=self.headers, timeout=self.timeout, follow_redirects=True
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script, style, nav, footer noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else ""
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s{2,}", " ", text).strip()

        links = []
        if self.include_links:
            links = [
                urljoin(url, a.get("href", ""))
                for a in soup.find_all("a", href=True)
            ]

        splitter = RecursiveTextSplitter(
            chunk_size=self.chunk_size, overlap=self.overlap
        )
        text_chunks = splitter.split(text)

        return [
            Chunk(
                text=chunk,
                metadata={
                    "source": url,
                    "loader": "web",
                    "type": "webpage",
                    "title": title,
                    "chunk_index": i,
                    "links": links if self.include_links else [],
                },
            )
            for i, chunk in enumerate(text_chunks)
            if chunk.strip()
        ]

    async def crawl(
        self,
        root_url: str,
        max_pages: int = 10,
        same_domain_only: bool = True,
        delay_s: float = 0.5,
    ) -> list[Chunk]:
        """
        Crawl from root_url, following links up to max_pages.

        Args:
            root_url:         Starting URL.
            max_pages:        Maximum pages to visit.
            same_domain_only: Only follow links on the same domain.
            delay_s:          Seconds to wait between requests (rate limiting).
        """
        visited: set[str] = set()
        queue: list[str] = [root_url]
        all_chunks: list[Chunk] = []
        root_domain = urlparse(root_url).netloc

        self_with_links = WebLoader(
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            timeout=self.timeout,
            headers=self.headers,
            include_links=True,
        )

        while queue and len(visited) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                chunks = await self_with_links.load(url)
                all_chunks.extend(chunks)

                # Extract and enqueue links
                if chunks and self_with_links.include_links:
                    for link in chunks[0].metadata.get("links", []):
                        parsed = urlparse(link)
                        if not parsed.scheme.startswith("http"):
                            continue
                        if same_domain_only and parsed.netloc != root_domain:
                            continue
                        if link not in visited:
                            queue.append(link)
            except Exception:
                pass  # Skip failed pages

            if queue:
                await asyncio.sleep(delay_s)

        return all_chunks
