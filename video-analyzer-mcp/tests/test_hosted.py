import asyncio
import os
import time
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from video_analyzer_mcp import analyze_video as av
from video_analyzer_mcp import hosted


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["VIDEO_ANALYZER_FILES_DIR"] = str(tmp_path_factory.mktemp("files"))
    app = hosted.build_app()
    with TestClient(app) as c:
        yield c


INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "t", "version": "1"}}}
MCP_HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


# --- token gate -------------------------------------------------------------

def test_mcp_requires_token(client):
    assert client.post("/mcp", json=INIT, headers=MCP_HEADERS).status_code == 401
    bad = {**MCP_HEADERS, "x-mcprush-token": "nope"}
    assert client.post("/mcp", json=INIT, headers=bad).status_code == 401


def test_mcp_accepts_token_and_public_host(client):
    ok = {**MCP_HEADERS, "x-mcprush-token": os.environ["MCPRUSH_TOKEN"], "host": "mcp.example.com"}
    assert client.post("/mcp", json=INIT, headers=ok).status_code == 200


def test_rotation_token_accepted(monkeypatch):
    monkeypatch.setenv("MCPRUSH_TOKEN_NEXT", "mgw_next")
    assert hosted._token_ok("mgw_next")
    assert hosted._token_ok(os.environ["MCPRUSH_TOKEN"])
    assert not hosted._token_ok("")


def test_healthz_is_public(client):
    assert client.get("/healthz").text == "ok"


def test_check_deps_not_offered_when_hosted(client):
    ok = {**MCP_HEADERS, "x-mcprush-token": os.environ["MCPRUSH_TOKEN"]}
    r = client.post("/mcp", headers=ok,
                    json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = set(r.text.split('"name":"')[i].split('"')[0] for i in range(1, r.text.count('"name":"') + 1))
    assert {"video_analyze", "video_transcribe", "video_fingerprint", "video_extract_frames"} <= names
    assert "video_check_deps" not in names


# --- input policy -----------------------------------------------------------

@pytest.mark.parametrize("url", [
    "/etc/passwd",
    "file:///etc/passwd",
    "http://localhost:8000/x",
    "http://127.0.0.1/x",
    "http://169.254.169.254/latest",
    "http://10.0.0.5/v.mp4",
])
def test_rejects_non_public_inputs(url):
    with pytest.raises(ValueError):
        hosted.validate_remote_url(url)


def test_rejects_cookies_and_output_dir():
    with pytest.raises(ValueError, match="cookies"):
        hosted.check_hosted_inputs("https://example.com/v.mp4", cookies_file="/x")
    with pytest.raises(ValueError, match="output_dir"):
        hosted.check_hosted_inputs("https://example.com/v.mp4", output_dir="/tmp/x")


def test_duration_cap(monkeypatch):
    monkeypatch.setenv("VIDEO_ANALYZER_MAX_SECONDS", "300")
    hosted.check_duration(299)
    with pytest.raises(ValueError, match="limit"):
        hosted.check_duration(301)


# --- file links -------------------------------------------------------------

def test_bundle_files_served_until_expiry(client, monkeypatch):
    b = hosted.new_bundle()
    (b.path / "frames").mkdir()
    (b.path / "frames" / "shot_0001.jpg").write_bytes(b"jpeg")
    assert b.url == f"https://files.example.test/files/{b.id}"
    assert client.get(f"/files/{b.id}/frames/shot_0001.jpg").content == b"jpeg"

    b.zip_dir("frames", "frames.zip")
    assert client.get(f"/files/{b.id}/frames.zip").status_code == 200

    old = time.time() - 100 * 3600
    os.utime(b.path, (old, old))
    assert client.get(f"/files/{b.id}/frames/shot_0001.jpg").status_code == 404


def test_bundle_rejects_bad_ids_and_traversal(client):
    b = hosted.new_bundle()
    (b.path / "a.txt").write_text("x")
    assert client.get("/files/short/a.txt").status_code == 404
    assert hosted.resolve_bundle_file(b.id, "../../etc/passwd") is None
    assert hosted.resolve_bundle_file(b.id, "a.txt") is not None


# --- cache --------------------------------------------------------------------

def test_cached_call_reuses_result():
    calls = []

    async def compute():
        calls.append(1)
        return ["result"]

    params = SimpleNamespace(model_dump_json=lambda: '{"input_path":"https://e.com/v"}')

    async def go():
        a = await hosted.cached_call("t", params, compute)
        b = await hosted.cached_call("t", params, compute)
        return a, b

    a, b = asyncio.run(go())
    assert a == b == ["result"] and len(calls) == 1


# --- pipeline helpers -------------------------------------------------------

def test_sample_shots_is_even_and_capped():
    shots = [av.Shot(number=i, start_time=i, end_time=i + 1) for i in range(1, 101)]
    picked = av.sample_shots(shots, 10)
    assert len(picked) == 10 and picked[0].number == 1 and picked[-1].number == 91
    assert av.sample_shots(shots, None) == shots


def test_ytdlp_error_keeps_error_line():
    stderr = "WARNING: blah\nERROR: [youtube] x: Sign in to confirm you're not a bot\n"
    assert av._ytdlp_error(stderr).startswith("ERROR: [youtube]")


def test_frames_links_rewritten():
    from video_analyzer_mcp.server import _frames_url_rewrite

    md = "| 1 | 0:00 | ![Shot 1](../frames/shot_0001.jpg) |"
    assert "https://f.test/files/abc/frames/shot_0001.jpg" in _frames_url_rewrite(md, "https://f.test/files/abc")


def test_progress_sends_total_not_message():
    from video_analyzer_mcp.server import _Progress

    sent = []

    async def report_progress(progress, total=None, message=None):
        sent.append((progress, total, message))

    asyncio.run(_Progress(SimpleNamespace(report_progress=report_progress)).report(0.3, "Transcribing"))
    assert sent == [(0.3, 1.0, "Transcribing")]


def test_progress_run_raises_original_error():
    from video_analyzer_mcp.server import _Progress

    async def report_progress(*a, **k):
        pass

    def boom():
        raise RuntimeError("Download failed: Sign in to confirm you're not a bot")

    with pytest.raises(RuntimeError, match="Download failed"):
        asyncio.run(_Progress(SimpleNamespace(report_progress=report_progress)).run(boom))


class _FakeMessages:
    def __init__(self, stop="end_turn"):
        self.kwargs = None
        self.stop = stop

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(stop_reason=self.stop,
                               content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=" a prompt ")])


def _fake_client(stop="end_turn"):
    plain, beta = _FakeMessages(stop), _FakeMessages(stop)
    return SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta)), plain, beta


def test_claude_call_uses_fallback_on_sonnet(monkeypatch):
    monkeypatch.setenv("VIDEO_ANALYZER_MODEL", "claude-sonnet-5-5")
    c, plain, beta = _fake_client()
    assert av._claude_text(c, "hi", 100) == "a prompt"
    assert beta.kwargs["fallbacks"] == "default" and plain.kwargs is None
    assert beta.kwargs["output_config"] == {"effort": "low"}


def test_claude_call_plain_on_haiku_and_refusal_is_empty(monkeypatch):
    monkeypatch.setenv("VIDEO_ANALYZER_MODEL", "claude-haiku-5-5")
    c, plain, beta = _fake_client(stop="refusal")
    assert av._claude_text(c, "hi", 100) == ""
    assert beta.kwargs is None and plain.kwargs["model"] == "claude-haiku-5-5"


def test_contact_sheet(tmp_path):
    from PIL import Image

    shots = []
    for i in range(1, 7):
        f = tmp_path / f"shot_{i:04d}.jpg"
        Image.new("RGB", (1280, 720), (i * 30, 80, 120)).save(f)
        shots.append(av.Shot(number=i, start_time=i * 2.0, end_time=i * 2.0 + 2, frame_path=str(f)))
    out = av.make_contact_sheet(shots, str(tmp_path / "sheet.jpg"))
    im = Image.open(out)
    assert im.width == 4 * (320 + 8) + 8 and im.height > 2 * 180
