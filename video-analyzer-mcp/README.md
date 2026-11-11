# Video Analyzer MCP Server

<!-- mcp-name: io.github.BrightWayAI/video-analyzer -->

**Turn any video link into a shot-by-shot storyboard.** Every scene's frame, a Claude Vision description of it, the line spoken over it, and an eight-field stylistic fingerprint of the whole video, from one request to your AI assistant.

[![Video Analyzer MCP server](https://mcprush.com/badge/brightwayai-video-analyzer-mcp/card.svg)](https://mcprush.com/to/brightwayai-video-analyzer-mcp?ref=badge)

<p align="center">
  <img src="https://raw.githubusercontent.com/BrightWayAI/video-analyzer/main/docs/example-contact-sheet.jpg" width="560" alt="Contact sheet of all 27 shots found in the Big Buck Bunny trailer">
  <br>
  <sub>27 shots found in the 33-second <i>Big Buck Bunny</i> trailer. © Blender Foundation, CC BY 3.0.</sub>
</p>

More examples, who it's for and how it works: [the project README on GitHub](https://github.com/BrightWayAI/video-analyzer#readme).

## Tools

| Tool | What it does |
|------|-------------|
| `video_analyze` | The full breakdown: storyboard, contact sheet, fingerprint, transcript, Word file and frames |
| `video_extract_frames` | One frame per shot via scene detection (or fixed intervals), fast, no AI write-up |
| `video_transcribe` | Timestamped transcript (Whisper, runs locally, no OpenAI key) |
| `video_fingerprint` | Just the eight-field stylistic classification |
| `video_check_deps` | Checks FFmpeg, Whisper and friends are installed (self-hosted only) |

## Hosted: nothing to install

Subscribe on [mcprush](https://mcprush.com/to/brightwayai-video-analyzer-mcp?ref=badge) and add it to any MCP client. No API keys, FFmpeg or Python on your side; YouTube, Shorts, TikTok and direct links work.

## Run it yourself

Needs **Python 3.10+**, **FFmpeg** (`brew install ffmpeg` or `apt-get install ffmpeg`) and your own **`ANTHROPIC_API_KEY`**.

Install from GitHub for now: version 0.1.2 on PyPI installs mcp 2.x and crashes on startup.

```bash
pip install "git+https://github.com/BrightWayAI/video-analyzer#subdirectory=video-analyzer-mcp"
```

**Claude Code**

```bash
claude mcp add video-analyzer -s user -e ANTHROPIC_API_KEY=sk-ant-... -- \
  uvx --from "git+https://github.com/BrightWayAI/video-analyzer#subdirectory=video-analyzer-mcp" video-analyzer-mcp
```

**Claude Desktop**: add to `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "video-analyzer": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/BrightWayAI/video-analyzer#subdirectory=video-analyzer-mcp", "video-analyzer-mcp"],
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
    }
  }
}
```

Then ask your assistant:

- *"Break this video down into a storyboard: <link>"* → `video_analyze`
- *"Pull one frame per shot from this TikTok"* → `video_extract_frames`
- *"Transcribe this video"* → `video_transcribe`
- *"What's the visual style of this video?"* → `video_fingerprint`

## Hosted mode (remote HTTP)

`video-analyzer-mcp-http` serves the same tools over streamable HTTP at `/mcp`, for running behind a gateway such as mcprush. The hosted mode:

- answers only requests with a valid `x-mcprush-token` header and returns 401 to everything else
- accepts public `http(s)` video URLs only (no local paths, cookies files or output directories) and rejects private or local addresses
- returns results inline and serves its files (storyboard .docx, contact sheet, frames) behind unguessable links that expire
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
| `VIDEO_ANALYZER_CACHE_HOURS` | `24` | A repeated identical call returns the finished result |
| `VIDEO_ANALYZER_WAIT_SECONDS` | `35` | How long a call waits for its job before answering "still working" (mcprush allows 44s per call) |
| `VIDEO_ANALYZER_MAX_JOBS` | `12` | Jobs in progress at once; more are refused as busy |
| `YTDLP_PROXY` | | Residential proxy URL (`http://user:pass@host:port`). YouTube and Instagram always go through it; other sites only when blocked. Proxied downloads are capped at 480p to save proxy bandwidth |
| `SENTRY_DSN` | | Send errors to Sentry |

Long jobs: every hosted tool runs as a background job keyed by its arguments. A call waits up to `VIDEO_ANALYZER_WAIT_SECONDS`; if the job isn't finished it answers `isError` with "STILL WORKING … call again with exactly the same arguments". mcprush doesn't count `isError` answers, so the buyer pays for the one call that returns the result. Answers are plain JSON (no SSE stream) and kept under mcprush's 256 KiB cap: the inline contact sheet is shrunk, and dropped (then the storyboard shortened) if needed.

Hosted results: `video_analyze` returns a JSON summary (with the fingerprint and file links), the storyboard as Markdown with working image links, and a contact-sheet image. `video_extract_frames` returns shots with `frame_url`s and a contact sheet. `/healthz` answers `ok` for health checks.

## Stylistic Fingerprint Fields

A deterministic classifier over the shot descriptions and transcript, so the same video always gets the same labels.

1. **Rendering Class**: Live Action, Stylized 3D, Photoreal 3D, Flat 2D, Minimalist Line Art, Textured 2D, Mixed Media
2. **World Type**: Real-World Literal, Stylized Real-World, Abstract Concept Space, Data/Presentation Space, Fictional Metaphor Universe
3. **Character Strategy**: None, Mascot-Led, Single Narrator, Single Protagonist Arc, Ensemble Cast
4. **Narrative Structure**: Direct Explanation, Step-by-Step, Problem → Solution, Analogy, Myth-Busting, Listicle, Character Journey, …
5. **Visual Abstraction Index**: 1 (Photorealistic) to 5 (Maximum Abstraction)
6. **Visual Density**: Minimal, Sparse, Moderate, High
7. **Camera/Editing Language**: Cinematic, Social Vertical Punch, Presentation Deck, Dynamic, Static Slides, …
8. **Tonal Positioning**: Institutional, Corporate Professional, Gen Z Social, Child-Friendly, Dark Editorial

## Development

```bash
uv sync --extra dev
uv run pytest
```

## License

MIT
