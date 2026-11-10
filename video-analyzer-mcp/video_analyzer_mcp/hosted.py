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
import hmac
import ipaddress
import os
import socket
import sys
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

    app = mcp.streamable_http_app()
    app.add_middleware(BaseHTTPMiddleware, dispatch=mcprush_only)
    return app


def http_main():
    missing = [v for v in ("MCPRUSH_TOKEN", "ANTHROPIC_API_KEY") if not os.environ.get(v)]
    if missing:
        sys.exit(f"Refusing to start: set {', '.join(missing)}.")

    import uvicorn

    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    http_main()
