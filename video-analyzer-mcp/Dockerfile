# Hosted (HTTP) build of the Video Analyzer MCP server, for selling through mcprush.
# Required at runtime: MCPRUSH_TOKEN, ANTHROPIC_API_KEY. Recommended: PUBLIC_BASE_URL.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# yt-dlp needs a JavaScript runtime (Deno) to download from YouTube.
COPY --from=denoland/deno:bin-2.1.4 /deno /usr/local/bin/deno

WORKDIR /app
COPY pyproject.toml README.md ./
COPY video_analyzer_mcp ./video_analyzer_mcp
RUN pip install --no-cache-dir . sentry-sdk \
 && pip install --no-cache-dir --upgrade "yt-dlp[default]"

# Bake the Whisper models the hosted server allows, so the first call isn't a download.
ENV XDG_CACHE_HOME=/opt/cache HF_HOME=/opt/cache/huggingface
RUN python -c "from faster_whisper import WhisperModel as M; M('base', device='cpu', compute_type='int8'); M('small', device='cpu', compute_type='int8')" \
 && chmod -R a+rX /opt/cache
# The models are baked in: never ask the Hugging Face Hub at runtime.
ENV HF_HUB_OFFLINE=1

RUN useradd --create-home app
USER app
ENV PORT=8000 \
    VIDEO_ANALYZER_FILES_DIR=/tmp/video-analyzer-files
EXPOSE 8000
CMD ["video-analyzer-mcp-http"]
