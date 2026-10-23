import pytest

from grok_bot.tools.python_exec import run_python
from grok_bot.tools.web import _ddg_real_url, html_to_text, parse_ddg_html

DDG_SAMPLE = """
<div class="result"><a rel="nofollow" class="result__a"
 href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fx%3D1&amp;rut=abc">
Example <b>A</b></a>
<a class="result__snippet" href="#">Snippet &amp; more</a></div>
<div class="result"><a rel="nofollow" class="result__a" href="https://example.org/b">B</a></div>
"""


def test_parse_ddg_html():
    results = parse_ddg_html(DDG_SAMPLE, 5)
    assert results[0] == {
        "title": "Example A",
        "url": "https://example.com/a?x=1",
        "snippet": "Snippet & more",
    }
    assert results[1]["url"] == "https://example.org/b"
    assert len(parse_ddg_html(DDG_SAMPLE, 1)) == 1


def test_ddg_real_url_passthrough():
    assert _ddg_real_url("https://x.com") == "https://x.com"


def test_html_to_text_strips_scripts():
    title, text = html_to_text(
        "<html><head><title>T</title><script>evil()</script></head>"
        "<body><p>Hello</p><p>World</p><style>.a{}</style></body></html>"
    )
    assert title == "T"
    assert "evil" not in text and "Hello" in text and "World" in text


@pytest.mark.anyio
async def test_run_python_ok_and_error():
    assert (await run_python("print(6*7)")).strip() == "42"
    out = await run_python("raise ValueError('x')")
    assert "ValueError" in out and "[exit code 1]" in out


@pytest.mark.anyio
async def test_run_python_timeout():
    assert "timed out" in await run_python("while True: pass", timeout=1)


@pytest.mark.anyio
async def test_web_search_reports_ddg_challenge(monkeypatch):
    import httpx

    from grok_bot.tools.web import web_search_tool

    def handler(request):
        if "wikipedia" in request.url.host:
            return httpx.Response(200, json={"query": {"search": []}})
        return httpx.Response(202, text="<html>challenge</html>")

    real = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        return real(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)
    with pytest.raises(RuntimeError, match="SEARXNG_URL"):
        await web_search_tool().func(query="x")


@pytest.mark.anyio
async def test_web_search_falls_back_to_wikipedia(monkeypatch):
    import httpx

    from grok_bot.tools.web import web_search_tool

    def handler(request):
        if "wikipedia" in request.url.host:
            hit = {"title": "Grok (chatbot)", "snippet": "<b>Grok</b> is a chatbot"}
            return httpx.Response(200, json={"query": {"search": [hit]}})
        return httpx.Response(202, text="<html>challenge</html>")

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **k: real(*a, transport=httpx.MockTransport(handler), **k),
    )
    out = await web_search_tool().func(query="grok")
    assert "https://en.wikipedia.org/wiki/Grok_(chatbot)" in out and "Grok is a chatbot" in out


@pytest.mark.anyio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost:8000/x",
        "http://10.0.0.5/",
        "http://169.254.169.254/",
        "http://[::1]/",
        "file:///etc/passwd",
        "ftp://example.com/",
    ],
)
async def test_check_public_url_blocks_private(url, monkeypatch):
    from grok_bot.tools.web import check_public_url

    monkeypatch.delenv("GROK_ALLOW_PRIVATE_FETCH", raising=False)
    with pytest.raises(ValueError):
        await check_public_url(url)


@pytest.mark.anyio
async def test_check_public_url_override(monkeypatch):
    from grok_bot.tools.web import check_public_url

    monkeypatch.setenv("GROK_ALLOW_PRIVATE_FETCH", "1")
    await check_public_url("http://127.0.0.1/")


@pytest.mark.anyio
async def test_fetch_url_blocks_redirect_to_private(monkeypatch):
    import httpx

    from grok_bot.tools import web

    async def fake_check(url):
        if "127.0.0.1" in url:
            raise ValueError("Refusing to fetch non-public address")

    def handler(request):
        return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})

    real = httpx.AsyncClient
    monkeypatch.setattr(web, "check_public_url", fake_check)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **k: real(*a, transport=httpx.MockTransport(handler), **k),
    )
    with pytest.raises(ValueError, match="non-public"):
        await web.fetch_url_tool().func(url="https://example.com/")


def test_docker_command_is_locked_down():
    from grok_bot.tools.python_exec import docker_command

    cmd = docker_command("n", 256)
    for flag in ["--network", "none", "--read-only", "--cap-drop", "ALL", "--memory", "256m"]:
        assert flag in cmd


@pytest.mark.anyio
async def test_sandbox_backend_env(monkeypatch):
    from grok_bot.tools.python_exec import sandbox_backend

    monkeypatch.setenv("GROK_SANDBOX", "subprocess")
    assert sandbox_backend() == "subprocess"
    monkeypatch.setenv("GROK_SANDBOX", "bogus")
    with pytest.raises(ValueError):
        sandbox_backend()


def test_sandbox_off_removes_python_exec(monkeypatch):
    from grok_bot.tools import default_registry

    monkeypatch.setenv("GROK_SANDBOX", "off")
    assert [t.name for t in default_registry()] == ["web_search", "fetch_url"]
    monkeypatch.setenv("GROK_SANDBOX", "subprocess")
    assert "python_exec" in [t.name for t in default_registry()]
