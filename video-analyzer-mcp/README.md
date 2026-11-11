# Video Analyzer MCP Server

<!-- mcp-name: io.github.BrightWayAI/video-analyzer -->

An MCP (Model Context Protocol) server that analyzes videos and generates storyboard breakdowns. Extract frames, transcribe audio, analyze visuals with Claude Vision, and compute stylistic fingerprints — all usable directly from Claude Desktop or Claude Code.

## Tools

| Tool | Description |
|------|-------------|
| `video_analyze` | Full pipeline: download, extract frames, transcribe, visual analysis, stylistic fingerprint, storyboard document |
| `video_extract_frames` | Extract representative frames using scene detection or fixed intervals |
| `video_transcribe` | Transcribe audio with OpenAI Whisper (timestamped, speaker-labeled) |
| `video_fingerprint` | Generate an 8-field Stylistic Fingerprint v3 classification |
| `video_check_deps` | Verify all required dependencies are installed |

## Prerequisites

- **Python 3.10+**
- **FFmpeg** — `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux)
- **ANTHROPIC_API_KEY** — set as an environment variable

## Installation

### From GitHub

```bash
pip install "git+https://github.com/BrightWayAI/video-analyzer#subdirectory=video-analyzer-mcp"
```

Install from GitHub for now: version 0.1.2 on PyPI installs mcp 2.x and crashes on startup.

### For Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "video-analyzer": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/BrightWayAI/video-analyzer#subdirectory=video-analyzer-mcp", "video-analyzer-mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

### For Claude Code

```bash
claude mcp add video-analyzer -s user -- uvx --from "git+https://github.com/BrightWayAI/video-analyzer#subdirectory=video-analyzer-mcp" video-analyzer-mcp
```

## Usage Examples

Once installed, you can ask Claude:

- **"Analyze this video and create a storyboard"** — runs `video_analyze`
- **"Extract frames from this YouTube video"** — runs `video_extract_frames`
- **"Transcribe the audio from this video"** — runs `video_transcribe`
- **"What's the stylistic fingerprint of this video?"** — runs `video_fingerprint`
- **"Check if video analyzer dependencies are installed"** — runs `video_check_deps`

## Hosted mode (remote HTTP)

`video-analyzer-mcp-http` serves the same tools over streamable HTTP at `/mcp`, for running behind a gateway such as mcprush. The hosted mode:

- answers only requests with a valid `x-mcprush-token` header and returns 401 to everything else
- accepts public `http(s)` video URLs only (no local paths, cookies files or output directories) and rejects private or local addresses
- returns the storyboard as Markdown inline and keeps nothing on disk
- caps video length and the number of jobs that run at once

```bash
docker build -t video-analyzer-mcp .
docker run -p 8000:8000 \
  -e MCPRUSH_TOKEN=mgw_... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e VIDEO_ANALYZER_MAX_SECONDS=300 \
  video-analyzer-mcp
```

| Variable | Default | Purpose |
|---|---|---|
| `MCPRUSH_TOKEN` | (required) | Gateway token; the server won't start without it |
| `MCPRUSH_TOKEN_NEXT` | | Second token accepted during a rotation |
| `ANTHROPIC_API_KEY` | (required) | Claude Vision calls, billed to the operator |
| `PORT` | `8000` | Listen port |
| `VIDEO_ANALYZER_MAX_SECONDS` | `600` | Longest video accepted |
| `VIDEO_ANALYZER_MAX_CONCURRENT` | `2` | Heavy jobs run at once; others wait |
| `PUBLIC_BASE_URL` | Railway domain | Base for the expiring file links in results (e.g. `https://videomcp.example.com`) |
| `VIDEO_ANALYZER_MODEL` | `claude-sonnet-5-5` | Claude model for frame and content analysis |
| `VIDEO_ANALYZER_MAX_FRAMES` | `60` | Most frames sent to Claude per video (an even sample) |
| `VIDEO_ANALYZER_FRAME_WORKERS` | `6` | Frames analyzed in parallel |
| `VIDEO_ANALYZER_FILE_TTL_HOURS` | `72` | How long storyboard, frame and zip links work |
| `VIDEO_ANALYZER_CACHE_HOURS` | `24` | A repeated identical call returns the cached result |
| `YTDLP_PROXY` | | Residential proxy URL (`http://user:pass@host:port`). YouTube and Instagram always go through it; other sites only when blocked. Proxied downloads are capped at 480p to save proxy bandwidth |
| `SENTRY_DSN` | | Send errors to Sentry |

Hosted results: `video_analyze` returns a JSON summary (with the fingerprint and file links), the storyboard as Markdown with working image links, and a contact-sheet image. `video_extract_frames` returns shots with `frame_url`s and a contact sheet. `/healthz` answers `ok` for health checks.

## Stylistic Fingerprint Fields

The fingerprint classifier produces 8 deterministic fields:

1. **Rendering Class** — Stylized 3D, Flat 2D, Minimalist Line Art, Textured 2D, Mixed Media, Photoreal
2. **World Type** — Stylized Real-World, Abstract Concept Space, Data/Presentation Space, Fictional Metaphor Universe
3. **Character Strategy** — None, Mascot-Led, Single Narrator, Single Protagonist Arc, Ensemble Cast
4. **Narrative Structure** — Direct Explanation, Step-by-Step, Problem-Solution, Analogy, Myth-Busting, etc.
5. **Visual Abstraction Index** — 1 (Photorealistic) to 5 (Maximum Abstraction)
6. **Visual Density** — Minimal, Sparse, Moderate, High
7. **Camera/Editing Language** — Cinematic, Social Vertical Punch, Presentation Deck, Static Slides, etc.
8. **Tonal Positioning** — Institutional, Corporate Professional, Gen Z Social, Child-Friendly, Dark Editorial

## License

MIT
