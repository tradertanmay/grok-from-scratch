import json

import pytest
from fastapi.testclient import TestClient

from grok_bot.app import create_app
from grok_bot.config import Settings
from grok_bot.embeddings import SafeEmbedder
from grok_bot.files import Files, UnsupportedFile, chunk_text, classify, extract_text
from grok_bot.store import Store
from grok_bot.tools import ToolRegistry
from tests.conftest import FakeLLM, text_turn
from tests.test_memory import ConceptEmbedder

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
PW = "correct horse"


def make_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def test_classify():
    assert classify("a.PDF", "") == ("pdf", "application/pdf")
    assert classify("x.png", "application/octet-stream") == ("image", "image/png")
    assert classify("notes", "text/markdown; charset=utf-8") == ("text", "text/markdown")
    assert classify("main.py", "")[0] == "text"
    with pytest.raises(UnsupportedFile):
        classify("setup.exe", "application/x-msdownload")


def test_extract_text_pdf_and_binary():
    assert "Quarterly revenue grew" in extract_text("pdf", make_pdf("Quarterly revenue grew 40%"))
    assert extract_text("text", "héllo".encode()) == "héllo"
    with pytest.raises(UnsupportedFile):
        extract_text("text", b"\x00\x01binary")
    with pytest.raises(UnsupportedFile):
        extract_text("pdf", b"not a pdf")


def test_chunk_text_covers_text_with_overlap():
    paras = [f"Paragraph {i}. " + "word " * 60 for i in range(30)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, size=1200, overlap=200)
    assert len(chunks) > 5 and all(len(c) <= 1200 for c in chunks)
    for p in paras:
        assert any(p.strip() in c for c in chunks)
    assert chunk_text("short") == ["short"] and chunk_text("   ") == []
    assert len(chunk_text("x" * 5000, size=1000, overlap=100)) == 6


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "f.db"))


@pytest.mark.anyio
async def test_embedding_retrieval_picks_relevant_chunk(store):
    files = Files(store, SafeEmbedder(ConceptEmbedder()), k=1)
    filler = "\n\n".join("Lorem ipsum dolor sit amet. " * 40 for _ in range(15))
    body = filler + "\n\nThe team lunch is pizza on Friday.\n\n" + filler
    f = await files.upload("local", "notes.txt", "text/plain", body.encode())
    assert f["chunks"] > 10 and f["chars"] > 16_000
    block, images = await files.context("local", [f["id"]], None, "what food is served?")
    assert "pizza on Friday" in block and images == []
    assert block.count("### notes.txt (part") == 1


@pytest.mark.anyio
async def test_keyword_fallback_and_first_chunk_fallback(store):
    files = Files(store)
    filler = "\n\n".join("Lorem ipsum dolor sit amet. " * 40 for _ in range(15))
    body = "Intro section.\n\n" + filler + "\n\nThe wifi password is hunter2.\n\n" + filler
    f = await files.upload("local", "office.md", "", body.encode())
    block, _ = await files.context("local", [f["id"]], None, "what is the wifi password")
    assert "hunter2" in block
    block, _ = await files.context("local", [f["id"]], None, "zzz")
    assert "Intro section." in block


def make_client(tmp_path, turns=(), **settings):
    llm = FakeLLM([text_turn(t) for t in turns])
    app = create_app(
        Settings(llm_model="fake", embed_model="", **settings),
        llm=llm,
        store=Store(str(tmp_path / "t.db")),
        tools=ToolRegistry([]),
    )
    return TestClient(app), llm


def upload(client, name, data, mime="text/plain", **form):
    return client.post("/api/files", files={"file": (name, data, mime)}, data=form)


def test_upload_list_get_delete_and_privacy(tmp_path):
    client, _ = make_client(tmp_path)
    client.post("/api/auth/signup", json={"username": "alice", "password": PW})
    r = upload(client, "report.pdf", make_pdf("Revenue grew 40%"), "application/pdf")
    assert r.status_code == 200
    f = r.json()
    assert f["name"] == "report.pdf" and f["mime"] == "application/pdf" and f["chunks"] == 1
    assert client.get("/api/files").json()["files"][0]["id"] == f["id"]
    assert client.get(f"/api/files/{f['id']}").json()["name"] == "report.pdf"
    assert upload(client, "x.exe", b"MZ", "application/x-msdownload").status_code == 415
    assert upload(client, "empty.txt", b"   ").status_code == 415
    assert upload(client, "a.txt", b"hi", session_id="nope").status_code == 404

    bob = TestClient(client.app)
    bob.post("/api/auth/signup", json={"username": "bob", "password": PW})
    assert bob.get("/api/files").json() == {"files": []}
    assert bob.get(f"/api/files/{f['id']}").status_code == 404
    assert bob.delete(f"/api/files/{f['id']}").status_code == 404
    bob_chat = bob.post("/api/chat", json={"message": "hi", "file_ids": [f["id"]]})
    assert bob_chat.status_code == 404

    assert client.delete(f"/api/files/{f['id']}").json() == {"deleted": f["id"]}
    assert client.get("/api/files").json() == {"files": []}


def test_upload_size_limit(tmp_path):
    client, _ = make_client(tmp_path, auth_enabled=False, max_upload_mb=0)
    assert upload(client, "a.txt", b"hello").status_code == 413


def test_chat_uses_file_and_keeps_it_for_the_session(tmp_path):
    client, llm = make_client(tmp_path, ["It grew 40%.", "Still 40%."], auth_enabled=False)
    f = upload(client, "report.pdf", make_pdf("Revenue grew 40%"), "application/pdf").json()
    r = client.post("/api/chat", json={"message": "how much growth?", "file_ids": [f["id"]]})
    sid = json.loads(r.text.split("data:", 2)[1].split("\n")[0])["session_id"]
    system = llm.calls[0]["messages"][0]["content"]
    assert "### report.pdf" in system and "Revenue grew 40%" in system
    assert client.get(f"/api/files?session_id={sid}").json()["files"][0]["id"] == f["id"]

    client.post("/api/chat", json={"session_id": sid, "message": "and again?"})
    assert "Revenue grew 40%" in llm.calls[1]["messages"][0]["content"]
    stored = client.get(f"/api/sessions/{sid}").json()["messages"][0]
    assert stored["content"] == "how much growth?"


def test_images_go_to_vision_model_as_image_parts(tmp_path):
    client, llm = make_client(
        tmp_path, ["a cat", "a png"], auth_enabled=False, vision_model="mac/gemma4:12b"
    )
    url = "data:image/png;base64,iVBORw0KGgo="
    client.post("/api/chat", json={"message": "what is this?", "images": [url]})
    call = llm.calls[0]
    assert call["model"] == "mac/gemma4:12b"
    content = call["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1] == {"type": "image_url", "image_url": {"url": url}}

    f = upload(client, "pic.png", PNG, "image/png").json()
    assert f["chunks"] == 0 and f["mime"] == "image/png"
    client.post("/api/chat", json={"message": "and this?", "file_ids": [f["id"]]})
    part = llm.calls[1]["messages"][-1]["content"][1]
    assert part["image_url"]["url"].startswith("data:image/png;base64,iVBORw0KGgo")
    assert "messages" not in client.get(f"/api/files/{f['id']}").json()

    bad = client.post("/api/chat", json={"message": "x", "images": ["https://evil/x.png"]})
    assert bad.status_code == 422


def test_text_only_chat_keeps_default_model(tmp_path):
    client, llm = make_client(tmp_path, ["hi"], auth_enabled=False, vision_model="mac/v")
    client.post("/api/chat", json={"message": "hello"})
    assert llm.calls[0]["model"] == "fake"
    assert llm.calls[0]["messages"][-1]["content"] == "hello"
