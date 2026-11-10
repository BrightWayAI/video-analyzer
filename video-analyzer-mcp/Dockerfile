# Hosted (HTTP) build of the Video Analyzer MCP server, for selling through mcprush.
# Required at runtime: MCPRUSH_TOKEN, ANTHROPIC_API_KEY.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY video_analyzer_mcp ./video_analyzer_mcp
RUN pip install --no-cache-dir .

# Bake the Whisper models the hosted server allows, so the first call isn't a download.
ENV XDG_CACHE_HOME=/opt/cache
RUN python -c "import whisper; whisper.load_model('base'); whisper.load_model('small')"

RUN useradd --create-home app
USER app
ENV PORT=8000
EXPOSE 8000
CMD ["video-analyzer-mcp-http"]
