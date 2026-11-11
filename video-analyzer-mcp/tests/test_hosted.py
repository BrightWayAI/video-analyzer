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


# --- background jobs --------------------------------------------------------

def _params(url="https://e.com/v"):
    return SimpleNamespace(model_dump_json=lambda: '{"input_path":"%s"}' % url)


def test_job_result_is_reused_without_recomputing(monkeypatch):
    hosted._jobs.clear()
    calls = []

    async def compute(job):
        calls.append(1)
        return "result"

    async def go():
        return (await hosted.run_job("t", _params(), compute),
                await hosted.run_job("t", _params(), compute))

    assert asyncio.run(go()) == ("result", "result") and len(calls) == 1


def test_slow_job_answers_still_working_then_result(monkeypatch):
    hosted._jobs.clear()
    monkeypatch.setenv("VIDEO_ANALYZER_WAIT_SECONDS", "0.05")

    async def compute(job):
        job.update(0.5, "Analyzing frames with Claude")
        await asyncio.sleep(0.2)
        return "done"

    async def go():
        first = await hosted.run_job("video_analyze", _params(), compute)
        await asyncio.sleep(0.3)
        second = await hosted.run_job("video_analyze", _params(), compute)
        return first, second

    first, second = asyncio.run(go())
    assert first.isError and "STILL WORKING" in first.content[0].text
    assert "Analyzing frames with Claude, about 50% done" in first.content[0].text
    assert "video_analyze again with exactly the same arguments" in first.content[0].text
    assert second == "done"


def test_failed_job_reported_once_then_retried(monkeypatch):
    hosted._jobs.clear()
    attempts = []

    async def compute(job):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("Download failed: boom")
        return "ok"

    async def go():
        with pytest.raises(RuntimeError, match="boom"):
            await hosted.run_job("t", _params(), compute)
        return await hosted.run_job("t", _params(), compute)

    assert asyncio.run(go()) == "ok" and len(attempts) == 2


def test_too_many_running_jobs_refused(monkeypatch):
    hosted._jobs.clear()
    monkeypatch.setenv("VIDEO_ANALYZER_WAIT_SECONDS", "0.01")
    monkeypatch.setenv("VIDEO_ANALYZER_MAX_JOBS", "1")

    async def compute(job):
        await asyncio.sleep(1)

    async def go():
        await hosted.run_job("t", _params("https://e.com/a"), compute)
        with pytest.raises(RuntimeError, match="busy"):
            await hosted.run_job("t", _params("https://e.com/b"), compute)
        for j in hosted._jobs.values():
            j.task.cancel()

    asyncio.run(go())


def test_tool_call_through_app_returns_json_still_working(client, monkeypatch):
    """End to end: a slow tool answers isError 'still working' as plain JSON
    within the wait, and the repeat call returns the finished result."""
    from video_analyzer_mcp import server

    hosted._jobs.clear()
    monkeypatch.setenv("VIDEO_ANALYZER_WAIT_SECONDS", "0.2")
    monkeypatch.setattr(hosted, "validate_remote_url", lambda url: None)

    async def slow_transcribe(params, progress):
        await progress.report(0.3, "Transcribing audio")
        await asyncio.sleep(0.6)
        return '{"status": "success"}'

    monkeypatch.setattr(server, "_transcribe", slow_transcribe)
    call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "video_transcribe",
                       "arguments": {"params": {"input_path": "https://e.com/v.mp4"}}}}
    headers = {**MCP_HEADERS, "x-mcprush-token": os.environ["MCPRUSH_TOKEN"]}

    t0 = time.time()
    r = client.post("/mcp", json=call, headers=headers)
    assert time.time() - t0 < 0.6
    assert r.headers["content-type"].startswith("application/json")
    result = r.json()["result"]
    assert result["isError"] is True and "STILL WORKING" in result["content"][0]["text"]

    time.sleep(0.6)
    result = client.post("/mcp", json=call, headers=headers).json()["result"]
    assert result["isError"] is False
    assert result["content"][0]["text"] == '{"status": "success"}'


# --- response size ----------------------------------------------------------

def test_fit_response_drops_image_then_truncates():
    small = hosted.fit_response(["summary", "storyboard"], b"\xff" * 1000)
    assert [c.type for c in small.content] == ["text", "text", "image"]

    no_image = hosted.fit_response(["summary", "x" * 100_000], b"\xff" * 120_000)
    assert [c.type for c in no_image.content] == ["text", "text"]

    cut = hosted.fit_response(["summary", "y" * 400_000], None, docs_hint="See the docx.")
    assert len(cut.model_dump_json().encode()) <= hosted.MAX_RESPONSE_BYTES
    assert cut.content[0].text == "summary" and "See the docx." in cut.content[1].text


def test_shrink_jpeg_fits_budget(tmp_path):
    from PIL import Image as PILImage
    import random

    path = tmp_path / "sheet.jpg"
    img = PILImage.new("RGB", (1280, 2000))
    img.putdata([(random.randrange(256), random.randrange(256), random.randrange(256))
                 for _ in range(1280 * 2000)])
    img.save(path, "JPEG", quality=90)
    out = hosted.shrink_jpeg(str(path), max_bytes=90_000)
    assert out is None or len(out) <= 90_000


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


def test_extract_frame_steps_back_past_end(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, capture_output=True):
        calls.append(cmd)
        out = cmd[-2]
        ok = len(calls) == 2  # first seek lands past the last frame
        if ok:
            open(out, "wb").write(b"jpg")
        return SimpleNamespace(returncode=0 if ok else 234, args=cmd, stdout=b"", stderr=b"")

    monkeypatch.setattr(av.subprocess, "run", fake_run)
    out = tmp_path / "f.jpg"
    assert av.extract_frame_at_time("v.mp4", 32.495, str(out)) == str(out)
    assert calls[1][1:3] == ["-ss", "31.495"]


def test_extract_frame_raises_when_all_fail(monkeypatch):
    monkeypatch.setattr(av.subprocess, "run",
                        lambda cmd, capture_output=True: SimpleNamespace(returncode=1, args=cmd, stdout=b"", stderr=b"x"))
    with pytest.raises(av.subprocess.CalledProcessError):
        av.extract_frame_at_time("v.mp4", 5.0, "/nonexistent/f.jpg")


def test_ffmpeg_failure_hint_is_not_missing_install():
    import subprocess
    from video_analyzer_mcp.server import _error_suggestion

    failed = subprocess.CalledProcessError(234, ["ffmpeg", "-i", "x"])
    assert "not installed" not in _error_suggestion(failed)
    assert "not installed" in _error_suggestion(FileNotFoundError("[Errno 2] No such file or directory: 'ffmpeg'"))


def test_whisper_hallucinations_dropped():
    # Scores Whisper gave the music-only Big Buck Bunny trailer
    assert av._likely_not_speech({"no_speech_prob": 0.48, "avg_logprob": -2.58, "compression_ratio": 0.81})
    assert av._likely_not_speech({"no_speech_prob": 0.72, "avg_logprob": -0.93, "compression_ratio": 0.27})
    # Ordinary narration stays
    assert not av._likely_not_speech({"no_speech_prob": 0.02, "avg_logprob": -0.25, "compression_ratio": 1.5})
    assert not av._likely_not_speech({"no_speech_prob": 0.6, "avg_logprob": -0.3, "compression_ratio": 1.4})


def _fake_ytdlp(monkeypatch, outcomes):
    """outcomes: list of (returncode, stderr) for successive yt-dlp runs."""
    calls = []

    def run(cmd, capture_output=True, text=True):
        rc, err = outcomes[len(calls)]
        calls.append(cmd)
        return SimpleNamespace(returncode=rc, stdout="My Title\n" if rc == 0 else "", stderr=err)

    monkeypatch.setattr(av.subprocess, "run", run)
    return calls


def test_youtube_goes_straight_to_proxy(tmp_path, monkeypatch):
    monkeypatch.setenv("YTDLP_PROXY", "http://user:secret@proxy:8000")
    calls = _fake_ytdlp(monkeypatch, [(0, "")])
    (tmp_path / "video_download.mp4").write_bytes(b"x")
    _, title = av.download_video("https://www.youtube.com/watch?v=abc", str(tmp_path))
    assert title == "My Title"
    assert len(calls) == 1 and "--proxy" in calls[0]
    assert "res:480,ext:mp4:m4a" in calls[0]


def test_other_sites_fall_back_to_proxy_only_when_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("YTDLP_PROXY", "http://user:secret@proxy:8000")
    (tmp_path / "video_download.mp4").write_bytes(b"x")
    calls = _fake_ytdlp(monkeypatch, [(1, "ERROR: [TikTok] x: HTTP Error 403: Forbidden"), (0, "")])
    av.download_video("https://www.tiktok.com/@a/video/1", str(tmp_path))
    assert "--proxy" not in calls[0] and "--proxy" in calls[1]

    calls = _fake_ytdlp(monkeypatch, [(1, "ERROR: Unsupported URL: https://example.com")])
    with pytest.raises(RuntimeError, match="Unsupported URL"):
        av.download_video("https://example.com/v", str(tmp_path))
    assert len(calls) == 1


def test_proxy_secret_scrubbed_from_errors(tmp_path, monkeypatch):
    proxy = "http://user:secret@proxy:8000"
    monkeypatch.setenv("YTDLP_PROXY", proxy)
    _fake_ytdlp(monkeypatch, [(1, f"ERROR: Unable to connect to proxy {proxy}")])
    with pytest.raises(RuntimeError) as e:
        av.download_video("https://youtu.be/abc", str(tmp_path))
    assert "secret" not in str(e.value)


def test_no_proxy_means_direct(tmp_path, monkeypatch):
    monkeypatch.delenv("YTDLP_PROXY", raising=False)
    (tmp_path / "video_download.mp4").write_bytes(b"x")
    calls = _fake_ytdlp(monkeypatch, [(0, "")])
    av.download_video("https://www.instagram.com/reel/abc/", str(tmp_path))
    assert "--proxy" not in calls[0] and "res:720,ext:mp4:m4a" in calls[0]


def test_oxylabs_proxy_gets_sticky_session():
    out = av._sticky_proxy("http://myuser:pa55@pr.oxylabs.io:7777")
    assert out.startswith("http://customer-myuser-sessid-") and out.endswith("-sesstime-10:pa55@pr.oxylabs.io:7777")
    assert av._sticky_proxy("http://myuser:pa55@pr.oxylabs.io:7777") != out  # new session per download
    already = "http://customer-u-sessid-abc-sesstime-10:p@pr.oxylabs.io:7777"
    assert av._sticky_proxy(already) == already
    other = "http://u:p@geo.iproyal.com:12321"
    assert av._sticky_proxy(other) == other


def test_on_screen_text_split_from_description():
    desc, text = av.split_on_screen_text(
        'Live-action footage. Woman at desk.\nON-SCREEN TEXT: "POV: you just got paid" / @money.tips')
    assert desc == "Live-action footage. Woman at desk."
    assert text == '"POV: you just got paid" / @money.tips'
    assert av.split_on_screen_text("3D render. Meadow.\nON-SCREEN TEXT: none") == ("3D render. Meadow.", "")
    assert av.split_on_screen_text("Flat 2D.\n**On-screen text:** 40% of Americans")[1] == "40% of Americans"
    assert av.split_on_screen_text("No marker.") == ("No marker.", "")


def test_pacing_and_hook_stats():
    spans = [(0, 0.8), (0.8, 1.5), (1.5, 4), (4, 9), (9, 10.5)]
    shots = [av.Shot(number=i + 1, start_time=a, end_time=b, visual_description="Live-action footage. Dog.",
                     on_screen_text="POV: my dog" if i == 0 else "") for i, (a, b) in enumerate(spans)]
    transcript = [av.TranscriptSegment(start_time=0.4, end_time=3.0, text="Mostly my kids fell in love with my nose"),
                  av.TranscriptSegment(start_time=6, end_time=9, text="and that is the story")]
    p = av.pacing_stats(shots, transcript, 10.5)
    assert p["cuts_per_minute"] == 22.9 and p["first_cut_at_seconds"] == 0.8
    assert p["cuts_per_10s_window"] == [4, 0] and p["pace"] == "Fast"
    assert p["hook"]["shots"] == 3 and p["hook"]["words_spoken"] == 9
    assert p["hook"]["on_screen_text"] == "POV: my dog"
    md = "\n".join(av.pacing_markdown(p))
    assert "## Pacing & Hook" in md and "22.9" in md and "Mostly my kids" in md
    assert av.pacing_stats([], [], 10) == {}


def test_docx_includes_pacing_and_on_screen_text(tmp_path):
    from docx import Document

    shots = [av.Shot(number=1, start_time=0, end_time=2, visual_description="Text card.", on_screen_text='"ONE BIG RABBIT"')]
    meta = av.VideoMetadata(title="t", duration=2.0, width=640, height=360, orientation="horizontal",
                            word_count=0, image_count=1)
    out = tmp_path / "s.docx"
    av.generate_docx_output(shots, meta, str(out), None, av.pacing_stats(shots, [], 2.0))
    text = "\n".join(c.text for t in Document(str(out)).tables for r in t.rows for c in r.cells)
    assert "Cuts per minute" in text and "ONE BIG RABBIT" in text


def test_pdf_storyboard_renders(tmp_path):
    from PIL import Image as PILImage
    from video_analyzer_mcp.pdf_storyboard import generate_pdf_storyboard

    shots = []
    for i in range(8):
        f = tmp_path / f"shot_{i}.jpg"
        PILImage.new("RGB", (320, 180), (40 * i, 90, 160)).save(f)
        shots.append(av.Shot(number=i + 1, start_time=i * 2, end_time=i * 2 + 2, frame_path=str(f),
                             visual_description="Live-action footage. A dog on a boat → smiling.",
                             on_screen_text='"POV"' if i == 0 else "", dialogue="hello there"))
    meta = av.VideoMetadata(title="Test – video", duration=16.0, width=320, height=180,
                            orientation="horizontal", word_count=2, image_count=8)
    out = generate_pdf_storyboard(shots, meta, str(tmp_path / "s.pdf"), None, av.pacing_stats(shots, [], 16.0))
    data = open(out, "rb").read()
    assert data.startswith(b"%PDF") and data.count(b"/Type /Page\n") + data.count(b"/Type /Page ") >= 3


def _fake_run(tmp_path, name, cuts, rendering):
    from PIL import Image as PILImage

    shots = [av.Shot(number=i + 1, start_time=i * 2.0, end_time=i * 2.0 + 2, visual_description=f"{rendering} shot {i}")
             for i in range(cuts)]
    meta = av.VideoMetadata(title=name, duration=cuts * 2.0, width=720, height=1280, orientation="vertical",
                            word_count=0, image_count=cuts, story_structure="Montage")
    result = av.PipelineResult(shots, meta, {"rendering_class": rendering, "tonal_positioning": "Gen Z Social"},
                               [], cuts, av.pacing_stats(shots, [], cuts * 2.0))
    sheet = tmp_path / f"{name}.jpg"
    PILImage.new("RGB", (640, 900), (200, 80, 80)).save(sheet)
    return {"result": result, "bundle": SimpleNamespace(url=f"https://x/files/{name}"), "markdown": "", "sheet": str(sheet)}


def test_compare_table_marks_differences(tmp_path):
    from video_analyzer_mcp import server

    a = server._video_summary(_fake_run(tmp_path, "a", 5, "Live Action (filmed footage)"))
    b = server._video_summary(_fake_run(tmp_path, "b", 5, "Flat 2D"))
    table = server.compare_table(a, b)
    rendering = next(l for l in table.splitlines() if "**Rendering**" in l)
    tone = next(l for l in table.splitlines() if "**Tone**" in l)
    assert rendering.endswith("| ≠ |") and tone.endswith("|  |")


def test_compare_job_end_to_end(tmp_path, monkeypatch):
    from video_analyzer_mcp import server

    hosted._jobs.clear()
    runs = {"https://e.com/a": _fake_run(tmp_path, "a", 4, "Flat 2D"),
            "https://e.com/b": _fake_run(tmp_path, "b", 8, "Live Action (filmed footage)")}

    async def fake_pipeline(url, interval, progress, tool):
        await progress.report(0.5, "Analyzing frames with Claude")
        return runs[url]

    monkeypatch.setattr(server, "_hosted_pipeline", fake_pipeline)
    monkeypatch.setattr(server, "_check_deps", lambda: None)
    monkeypatch.setattr(server, "compare_narrative", lambda *a: "### Biggest differences\n- B cuts twice as often")
    monkeypatch.setattr(hosted, "validate_remote_url", lambda url: None)

    params = server.CompareVideosInput(video_a="https://e.com/a", video_b="https://e.com/b")
    result = asyncio.run(hosted.run_job("video_compare", params, lambda job: server._compare_hosted(params, job)))
    summary = __import__("json").loads(result.content[0].text)
    assert summary["video_a"]["title"] == "a" and summary["video_b"]["shots"] == 8
    assert "Rendering" in summary["differs_on"] and "Shots" in summary["differs_on"]
    assert "B cuts twice as often" in result.content[1].text
    assert result.content[-1].type == "image"
