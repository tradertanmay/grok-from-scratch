import asyncio
import html
import ipaddress
import os
import re
import socket
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx

from .base import Tool

# A custom bot token in the UA makes DuckDuckGo answer with a 202 bot challenge.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
WIKI_USER_AGENT = "GrokBot/0.1 (https://github.com/tradertanmay/grok-from-scratch)"
_SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}


MAX_REDIRECTS = 5


def _allow_private() -> bool:
    return os.environ.get("GROK_ALLOW_PRIVATE_FETCH", "").lower() in {"1", "true", "yes"}


async def check_public_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts resolving to private/loopback/link-local IPs (SSRF)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only http(s) URLs are allowed")
    if _allow_private():
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    infos = await asyncio.get_running_loop().getaddrinfo(
        parsed.hostname, port, type=socket.SOCK_STREAM
    )
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise ValueError(f"Refusing to fetch non-public address {ip} ({parsed.hostname})")


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(raw: str) -> tuple[str, str]:
    p = _TextExtractor()
    p.feed(raw)
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return p.title.strip(), text.strip()


def _ddg_real_url(href: str) -> str:
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg")
        if uddg:
            return unquote(uddg[0])
    return href


_DDG_RESULT = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'(?:.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>)?',
    re.S,
)


def parse_ddg_html(raw: str, max_results: int) -> list[dict[str, str]]:
    results = []
    for m in _DDG_RESULT.finditer(raw):
        strip = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()  # noqa: E731
        results.append(
            {
                "title": strip(m["title"]),
                "url": _ddg_real_url(m["href"]),
                "snippet": strip(m["snippet"]),
            }
        )
        if len(results) >= max_results:
            break
    return results


def format_results(results: list[dict[str, str]]) -> str:
    if not results:
        return "No results."
    return "\n\n".join(
        f"[{i}] {r['title']}\n{r['url']}\n{r['snippet']}" for i, r in enumerate(results, 1)
    )


async def _wikipedia_search(
    client: httpx.AsyncClient, query: str, max_results: int
) -> list[dict[str, str]]:
    r = await client.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results,
        },
        headers={"User-Agent": WIKI_USER_AGENT},
    )
    r.raise_for_status()
    return [
        {
            "title": f"{x['title']} (Wikipedia)",
            "url": "https://en.wikipedia.org/wiki/" + x["title"].replace(" ", "_"),
            "snippet": html.unescape(re.sub(r"<[^>]+>", "", x.get("snippet", ""))),
        }
        for x in r.json().get("query", {}).get("search", [])
    ]


def web_search_tool(searxng_url: str = "") -> Tool:
    async def web_search(query: str, max_results: int = 5) -> str:
        max_results = max(1, min(int(max_results), 10))
        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient(timeout=20, headers=headers, follow_redirects=True) as c:
            if searxng_url:
                r = await c.get(f"{searxng_url}/search", params={"q": query, "format": "json"})
                r.raise_for_status()
                results = [
                    {
                        "title": x.get("title", ""),
                        "url": x.get("url", ""),
                        "snippet": x.get("content", ""),
                    }
                    for x in r.json().get("results", [])[:max_results]
                ]
            else:
                r = await c.post("https://html.duckduckgo.com/html/", data={"q": query})
                r.raise_for_status()
                results = parse_ddg_html(r.text, max_results)
                if not results and r.status_code == 202:
                    results = await _wikipedia_search(c, query, max_results)
                    if not results:
                        raise RuntimeError(
                            "DuckDuckGo returned a bot challenge (HTTP 202) for this IP and "
                            "Wikipedia had no results; set SEARXNG_URL to use SearXNG instead"
                        )
        return format_results(results)

    return Tool(
        name="web_search",
        description="Search the web for current information. Returns titles, URLs and snippets.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "1-10, default 5"},
            },
            "required": ["query"],
        },
        func=web_search,
    )


def fetch_url_tool(max_chars: int = 10_000) -> Tool:
    async def fetch_url(url: str) -> str:
        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient(timeout=20, headers=headers) as c:
            for _ in range(MAX_REDIRECTS + 1):
                await check_public_url(url)
                r = await c.get(url)
                if not r.is_redirect:
                    break
                url = urljoin(str(r.url), r.headers["location"])
            else:
                raise ValueError(f"Too many redirects (>{MAX_REDIRECTS})")
            r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "html" in ctype:
            title, text = html_to_text(r.text)
        else:
            title, text = "", r.text
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return f"Title: {title}\nURL: {r.url}\n\n{text}" if title else f"URL: {r.url}\n\n{text}"

    return Tool(
        name="fetch_url",
        description="Fetch a web page and return its readable text content.",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "http(s) URL"}},
            "required": ["url"],
        },
        func=fetch_url,
    )
