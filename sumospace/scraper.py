# sumospace/scraper.py

"""
Async Web Scraper
==================
Fetches and cleans web pages for knowledge retrieval.
- Respects robots.txt
- Rate limiting + exponential backoff
- Converts HTML → clean Markdown
- Optional: Playwright for JS-rendered pages
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class ScrapedPage:
    url: str
    title: str
    text: str
    markdown: str
    links: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str = ""


# ─── Core Scraper ─────────────────────────────────────────────────────────────

class WebScraper:
    """
    Async web scraper with robots.txt compliance, rate limiting, and retry logic.
    No API key required.
    """

    HEADERS = {
        "User-Agent": "SumoSpace/0.1 (research agent; +https://github.com/sumospace/sumospace)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(
        self,
        timeout: int = 20,
        max_retries: int = 3,
        rate_limit: float = 1.0,     # seconds between requests to same domain
        respect_robots: bool = True,
        max_content_length: int = 2_000_000,  # 2MB cap
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self.respect_robots = respect_robots
        self.max_content_length = max_content_length

        self._last_request: dict[str, float] = {}
        self._robots_cache: dict[str, RobotFileParser] = {}
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            headers=self.HEADERS,
            timeout=self.timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def fetch(self, url: str) -> ScrapedPage:
        """Fetch and parse a single URL."""
        if self.respect_robots and not await self._is_allowed(url):
            return ScrapedPage(
                url=url, title="", text="", markdown="",
                success=False, error="Blocked by robots.txt",
            )

        await self._rate_limit_wait(url)

        for attempt in range(self.max_retries):
            try:
                resp = await self._client.get(url)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 5 * (attempt + 1)))
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code >= 400:
                    return ScrapedPage(
                        url=url, title="", text="", markdown="",
                        success=False, error=f"HTTP {resp.status_code}",
                    )
                content_type = resp.headers.get("content-type", "")
                if "html" not in content_type and "text" not in content_type:
                    return ScrapedPage(
                        url=url, title="", text="", markdown="",
                        success=False, error=f"Unsupported content type: {content_type}",
                    )
                html = resp.text[:self.max_content_length]
                return self._parse_html(html, url)
            except httpx.TimeoutException:
                if attempt == self.max_retries - 1:
                    return ScrapedPage(
                        url=url, title="", text="", markdown="",
                        success=False, error="Request timed out",
                    )
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                return ScrapedPage(
                    url=url, title="", text="", markdown="",
                    success=False, error=str(e),
                )
        return ScrapedPage(url=url, title="", text="", markdown="",
                           success=False, error="Max retries exceeded")

    async def fetch_many(self, urls: list[str], concurrency: int = 3) -> list[ScrapedPage]:
        """Fetch multiple URLs concurrently with a semaphore."""
        sem = asyncio.Semaphore(concurrency)

        async def _fetch_one(url):
            async with sem:
                return await self.fetch(url)

        return await asyncio.gather(*[_fetch_one(u) for u in urls])

    def _parse_html(self, html: str, base_url: str) -> ScrapedPage:
        """Extract title, text, markdown, and links from raw HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # Fallback: basic regex strip
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return ScrapedPage(url=base_url, title="", text=text, markdown=text)

        soup = BeautifulSoup(html, "lxml" if _has_lxml() else "html.parser")

        # Remove boilerplate
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "iframe", "noscript", "svg"]):
            tag.decompose()

        title = ""
        if soup.title:
            title = soup.title.string or ""
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        # Extract links
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http"):
                links.append(href)
            elif href.startswith("/"):
                links.append(urljoin(base_url, href))

        # Clean text
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # Convert to markdown if available
        try:
            from markdownify import markdownify as md
            markdown = md(str(soup.find("body") or soup), heading_style="ATX")
            markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
        except ImportError:
            markdown = text

        return ScrapedPage(
            url=base_url,
            title=title.strip(),
            text=text[:50000],
            markdown=markdown[:50000],
            links=links[:100],
            metadata={"content_length": len(html)},
        )

    async def _is_allowed(self, url: str) -> bool:
        """Check robots.txt for the given URL."""
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root not in self._robots_cache:
            rp = RobotFileParser()
            rp.set_url(f"{root}/robots.txt")
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{root}/robots.txt")
                    if resp.status_code == 200:
                        rp.parse(resp.text.splitlines())
            except Exception:
                pass
            self._robots_cache[root] = rp
        return self._robots_cache[root].can_fetch(self.HEADERS["User-Agent"], url)

    async def _rate_limit_wait(self, url: str):
        """Wait to respect per-domain rate limits."""
        domain = urlparse(url).netloc
        last = self._last_request.get(domain, 0)
        elapsed = time.monotonic() - last
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self._last_request[domain] = time.monotonic()


def _has_lxml() -> bool:
    try:
        import lxml
        return True
    except ImportError:
        return False


# ─── Convenience functions ────────────────────────────────────────────────────

async def scrape(url: str) -> ScrapedPage:
    """One-shot fetch and parse a URL."""
    async with WebScraper() as scraper:
        return await scraper.fetch(url)


async def scrape_many(urls: list[str], concurrency: int = 3) -> list[ScrapedPage]:
    """Fetch multiple URLs."""
    async with WebScraper() as scraper:
        return await scraper.fetch_many(urls, concurrency=concurrency)
