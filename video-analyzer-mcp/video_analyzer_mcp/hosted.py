"""
Hosted (remote HTTP) mode for the Video Analyzer MCP server.

Run with the `video-analyzer-mcp-http` entry point. In this mode the server:

- serves MCP over streamable HTTP at /mcp
- refuses every request to /mcp that lacks a valid `x-mcprush-token` header
  (MCPRUSH_TOKEN, plus MCPRUSH_TOKEN_NEXT during a rotation)
- accepts only public http(s) video URLs: no local paths, no cookies files,
  no custom output directories
- caps video length (VIDEO_ANALYZER_MAX_SECONDS, default 600)
- runs at most VIDEO_ANALYZER_MAX_CONCURRENT heavy jobs at once (default 2)
- returns results inline instead of as paths on the server's disk

Environment:
    MCPRUSH_TOKEN                  required; the gateway token from mcprush
    MCPRUSH_TOKEN_NEXT             optional; second token accepted during rotation
    ANTHROPIC_API_KEY              required; billed to the operator
    PORT                           optional; default 8000
    VIDEO_ANALYZER_MAX_SECONDS     optional; default 600
    VIDEO_ANALYZER_MAX_CONCURRENT  optional; default 2
"""

import asyncio
import contextlib
import hashlib
import hmac
import ipaddress
import logging
import os
import re
import secrets
import shutil
import socket
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

HOSTED_ENV = "VIDEO_ANALYZER_HOSTED"
HOSTED_WHISPER_MODELS = {"base", "small"}


def is_hosted() -> bool:
    return os.environ.get(HOSTED_ENV) == "1"


def max_duration() -> float:
    return float(os.environ.get("VIDEO_ANALYZER_MAX_SECONDS", "600"))


# ---------------------------------------------------------------------------
# Input policy
# ---------------------------------------------------------------------------

def validate_remote_url(url: str) -> None:
    """Allow only http(s) URLs whose host resolves to public addresses."""
    parsed = urlparse(url if "://" in url else "https://" + url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("The hosted server accepts only http(s) video URLs, not file paths.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise ValueError(f"Could not resolve host: {parsed.hostname}")
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise ValueError("That URL points at a private or local address.")


def check_hosted_inputs(input_path: str, cookies_file=None, output_dir=None) -> None:
    """Raise ValueError if a tool input isn't allowed on the hosted server."""
    if not is_hosted():
        return
    validate_remote_url(input_path)
    if cookies_file:
        raise ValueError("cookies_file isn't supported on the hosted server.")
    if output_dir:
        raise ValueError("output_dir isn't supported on the hosted server.")


def check_duration(duration: float) -> None:
    if is_hosted() and duration > max_duration():
        raise ValueError(
            f"Video is {duration:.0f}s long; the hosted limit is {max_duration():.0f}s."
        )


def max_frames() -> int:
    """Most frames sent to Claude per video on the hosted server."""
    return int(os.environ.get("VIDEO_ANALYZER_MAX_FRAMES", "60"))


def file_ttl_hours() -> int:
    return int(os.environ.get("VIDEO_ANALYZER_FILE_TTL_HOURS", "72"))


# ---------------------------------------------------------------------------
# Expiring file links: each job's outputs live in a "bundle" directory served
# at /files/<bundle-id>/..., where the id is an unguessable random token.
# ---------------------------------------------------------------------------

_BUNDLE_ID = re.compile(r"^[A-Za-z0-9_-]{20,64}$")


def files_root() -> Path:
    return Path(os.environ.get("VIDEO_ANALYZER_FILES_DIR", "/tmp/video-analyzer-files"))


def public_base_url() -> str | None:
    """Where buyers can fetch files from, e.g. https://videomcp.example.com."""
    url = os.environ.get("PUBLIC_BASE_URL")
    if not url and os.environ.get("RAILWAY_PUBLIC_DOMAIN"):
        url = "https://" + os.environ["RAILWAY_PUBLIC_DOMAIN"]
    return url.rstrip("/") if url else None


class Bundle:
    def __init__(self, bundle_id: str):
        self.id = bundle_id
        self.path = files_root() / bundle_id

    @property
    def url(self) -> str | None:
        base = public_base_url()
        return f"{base}/files/{self.id}" if base else None

    def zip_dir(self, subdir: str, zip_name: str) -> Path:
        src = self.path / subdir
        out = self.path / zip_name
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(src.glob("*")):
                if f.is_file():
                    zf.write(f, f.name)
        return out

    def discard(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def sweep_bundles() -> None:
    """Delete bundles older than the TTL."""
    root = files_root()
    if not root.is_dir():
        return
    cutoff = time.time() - file_ttl_hours() * 3600
    for d in root.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def new_bundle() -> Bundle:
    sweep_bundles()
    bundle = Bundle(secrets.token_urlsafe(24))
    bundle.path.mkdir(parents=True, exist_ok=True)
    return bundle


def resolve_bundle_file(bundle_id: str, rel_path: str) -> Path | None:
    """The file a /files URL names, or None if invalid, missing or expired."""
    if not _BUNDLE_ID.match(bundle_id):
        return None
    base = (files_root() / bundle_id).resolve()
    target = (base / rel_path).resolve()
    if base not in target.parents or not target.is_file():
        return None
    if base.stat().st_mtime < time.time() - file_ttl_hours() * 3600:
        return None
    return target


async def serve_file(request):
    from starlette.responses import FileResponse, PlainTextResponse

    target = resolve_bundle_file(request.path_params["bundle"], request.path_params["path"])
    if target is None:
        return PlainTextResponse("Not found or expired.", status_code=404)
    return FileResponse(target, headers={"Cache-Control": "private, max-age=3600"})


async def healthz(request):
    from starlette.responses import PlainTextResponse

    return PlainTextResponse("ok")


# ---------------------------------------------------------------------------
# Background jobs. The mcprush gateway gives a call 44 seconds end to end, and
# a video takes minutes. So a call starts the work as a background job keyed
# by (tool, arguments) and waits up to wait_seconds(); if it isn't done, it
# answers isError "still working, call again with the same arguments". mcprush
# doesn't count isError answers, nor a repeat of a counted call within 15
# minutes, so the buyer pays for one call. A finished job's result is kept
# (VIDEO_ANALYZER_CACHE_HOURS) so a repeat returns it instead of paying twice.
# ---------------------------------------------------------------------------

def wait_seconds() -> float:
    return float(os.environ.get("VIDEO_ANALYZER_WAIT_SECONDS", "35"))


def _cache_ttl() -> float:
    return float(os.environ.get("VIDEO_ANALYZER_CACHE_HOURS", "24")) * 3600


def _max_jobs() -> int:
    return int(os.environ.get("VIDEO_ANALYZER_MAX_JOBS", "12"))


class Job:
    def __init__(self, tool: str):
        self.tool = tool
        self.started = time.time()
        self.finished: float | None = None
        self.frac = 0.0
        self.msg = "Queued"
        self.task: asyncio.Task | None = None

    def update(self, frac: float, msg: str) -> None:
        self.frac, self.msg = frac, msg

    @property
    def done(self) -> bool:
        return self.task is not None and self.task.done()


_jobs: "OrderedDict[str, Job]" = OrderedDict()


def _job_key(tool: str, params) -> str:
    return hashlib.sha256(f"{tool}:{params.model_dump_json()}".encode()).hexdigest()


def _prune_jobs() -> None:
    ttl = min(_cache_ttl(), file_ttl_hours() * 3600)
    for key, job in list(_jobs.items()):
        if job.finished and time.time() - job.finished > ttl:
            del _jobs[key]
    finished = [k for k, j in _jobs.items() if j.finished]
    while len(_jobs) > 200 and finished:
        del _jobs[finished.pop(0)]


async def _run_to_end(job: Job, compute) -> object:
    try:
        return await compute(job)
    finally:
        job.finished = time.time()


def _pending_result(job: Job):
    from mcp.types import CallToolResult, TextContent

    elapsed = int(time.time() - job.started)
    pct = f", about {int(job.frac * 100)}% done" if job.frac > 0 else ""
    text = (
        f"STILL WORKING, not a failure. {job.msg}{pct}, {elapsed // 60}m{elapsed % 60:02d}s so far. "
        f"Call {job.tool} again with exactly the same arguments in about 30 seconds to get the "
        "result. Calls made while it is still running are not counted against your plan."
    )
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=True)


async def run_job(tool: str, params, compute):
    """Start or join the job for (tool, params); return its result or a
    'still working' isError answer after waiting at most wait_seconds()."""
    _prune_jobs()
    key = _job_key(tool, params)
    job = _jobs.get(key)
    if job is None:
        if sum(1 for j in _jobs.values() if not j.done) >= _max_jobs():
            raise RuntimeError("The server is busy with other videos. Try again in a few minutes.")
        job = Job(tool)
        job.task = asyncio.create_task(_run_to_end(job, compute))
        _jobs[key] = job
        log.info("job start tool=%s key=%s", tool, key[:8])
    else:
        log.info("job join tool=%s key=%s done=%s", tool, key[:8], job.done)

    if not job.done:
        await asyncio.wait({job.task}, timeout=wait_seconds())
    if not job.done:
        return _pending_result(job)
    if job.task.exception() is not None:
        _jobs.pop(key, None)  # a failed job is reported once; a retry starts over
        raise job.task.exception()
    return job.task.result()


# ---------------------------------------------------------------------------
# Response size: mcprush refuses answers over 256 KiB (the buyer gets a 500).
# ---------------------------------------------------------------------------

MAX_RESPONSE_BYTES = 230_000


def fit_response(texts: list[str], image: bytes | None = None, docs_hint: str = ""):
    """Build a CallToolResult under MAX_RESPONSE_BYTES: drop the image first,
    then shorten the last text block, pointing at the full files."""
    import base64

    from mcp.types import CallToolResult, ImageContent, TextContent

    def build(texts, image):
        content = [TextContent(type="text", text=t) for t in texts]
        if image:
            content.append(ImageContent(type="image", data=base64.b64encode(image).decode(),
                                        mimeType="image/jpeg"))
        return CallToolResult(content=content, isError=False)

    def size(result) -> int:
        return len(result.model_dump_json(by_alias=True, exclude_none=True).encode())

    result = build(texts, image)
    if size(result) <= MAX_RESPONSE_BYTES:
        return result
    result = build(texts, None)
    over = size(result) - MAX_RESPONSE_BYTES
    if over <= 0:
        return result
    texts = list(texts)
    note = f"\n\n[Cut to fit the response size limit.{' ' + docs_hint if docs_hint else ''}]"
    keep = max(0, len(texts[-1]) - over - len(note) - 2000)
    texts[-1] = texts[-1][:keep] + note
    return build(texts, None)


def shrink_jpeg(path: str, max_width: int = 1100, max_bytes: int = 90_000) -> bytes | None:
    """A smaller copy of a JPEG for inline display, or None if it can't fit."""
    import io

    from PIL import Image as PILImage

    img = PILImage.open(path).convert("RGB")
    # Cap the longest side: a sheet of vertical (TikTok/Reels) shots is tall.
    scale = min(1.0, max_width / max(img.width, img.height))
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    for quality in (75, 65, 55, 50, 45, 45):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality, optimize=True)
        if buf.tell() <= max_bytes:
            return buf.getvalue()
        if max(img.width, img.height) < 500:
            break
        img = img.resize((int(img.width * 0.85), int(img.height * 0.85)))
    return None


# ---------------------------------------------------------------------------
# Logging: one line per step with timings, so Railway's logs show where a
# slow or failed job spent its time.
# ---------------------------------------------------------------------------

log = logging.getLogger("video_analyzer")


class _JobLog:
    def __init__(self, tool: str, url: str):
        self.id = secrets.token_hex(4)
        self.tool = tool
        self.host = urlparse(url if "://" in url else "https://" + url).hostname or "local"
        self.t0 = self.last = time.monotonic()

    def step(self, name: str, **fields):
        now = time.monotonic()
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        log.info("job=%s tool=%s step=%s took=%.1fs %s", self.id, self.tool, name, now - self.last, extra)
        self.last = now


@contextlib.asynccontextmanager
async def job_log(tool: str, url: str):
    job = _JobLog(tool, url)
    log.info("job=%s tool=%s start host=%s", job.id, tool, job.host)
    try:
        yield job
    except Exception as e:
        log.warning("job=%s tool=%s failed after %.1fs: %s", job.id, tool, time.monotonic() - job.t0, e)
        raise
    log.info("job=%s tool=%s done total=%.1fs", job.id, tool, time.monotonic() - job.t0)


def _setup_observability() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    dsn = os.environ.get("SENTRY_DSN")
    if dsn:
        try:
            import sentry_sdk

            sentry_sdk.init(dsn=dsn, traces_sample_rate=0.0, send_default_pii=False)
            log.info("Sentry error reporting on")
        except ImportError:
            log.warning("SENTRY_DSN is set but sentry-sdk isn't installed")


_slots: asyncio.Semaphore | None = None


@contextlib.asynccontextmanager
async def job_slot():
    """Limit concurrent heavy jobs in hosted mode; no-op locally."""
    global _slots
    if not is_hosted():
        yield
        return
    if _slots is None:
        _slots = asyncio.Semaphore(int(os.environ.get("VIDEO_ANALYZER_MAX_CONCURRENT", "2")))
    async with _slots:
        yield


# ---------------------------------------------------------------------------
# mcprush gateway token check
# ---------------------------------------------------------------------------

def _token_ok(sent: str) -> bool:
    wants = [os.environ.get("MCPRUSH_TOKEN", ""), os.environ.get("MCPRUSH_TOKEN_NEXT", "")]
    return any(want and hmac.compare_digest(sent.encode(), want.encode()) for want in wants)


async def mcprush_only(request, call_next):
    from starlette.responses import JSONResponse

    if not request.scope["path"].startswith("/mcp"):
        return await call_next(request)
    if not _token_ok(request.headers.get("x-mcprush-token", "")):
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None,
             "error": {"code": -32001, "message": "This endpoint answers mcprush only."}},
            status_code=401,
        )
    return await call_next(request)


def build_app():
    """Build the ASGI app: streamable HTTP MCP behind the token check."""
    from starlette.middleware.base import BaseHTTPMiddleware

    os.environ[HOSTED_ENV] = "1"
    from video_analyzer_mcp.server import mcp

    # Public hostname behind the gateway: the SDK's localhost-only Host check
    # would answer 421, so turn it off; the token check is the gate.
    mcp.settings.transport_security = None
    mcp.settings.stateless_http = True
    # mcprush wants one answer per call, then the response closed: plain JSON
    # instead of an SSE stream.
    mcp.settings.json_response = True

    # Public routes (no token): health check, and job files behind random ids.
    mcp.custom_route("/healthz", methods=["GET"])(healthz)
    mcp.custom_route("/files/{bundle}/{path:path}", methods=["GET"])(serve_file)

    app = mcp.streamable_http_app()
    app.add_middleware(BaseHTTPMiddleware, dispatch=mcprush_only)
    return app


def http_main():
    missing = [v for v in ("MCPRUSH_TOKEN", "ANTHROPIC_API_KEY") if not os.environ.get(v)]
    if missing:
        sys.exit(f"Refusing to start: set {', '.join(missing)}.")

    import uvicorn

    _setup_observability()
    if not public_base_url():
        log.warning("PUBLIC_BASE_URL not set: results won't include file links")
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    http_main()
