# Video Analyzer

Analyze videos and generate storyboard breakdowns. Extracts frames via scene detection, transcribes audio with Whisper, analyzes visuals with Claude Vision, and computes 8-field stylistic fingerprints.

## Features

- **Frame Extraction** — Scene-detection-based keyframe selection (or fixed intervals)
- **Audio Transcription** — Timestamped transcription via OpenAI Whisper
- **Visual Analysis** — Per-frame descriptions using Claude Vision
- **Stylistic Fingerprint v3** — 8-field deterministic classification (rendering class, world type, character strategy, narrative structure, visual abstraction, visual density, camera language, tonal positioning)
- **Storyboard Output** — Combined shot-by-shot breakdown as `.docx` or `.md`

## MCP Server

This project includes an MCP (Model Context Protocol) server so you can use video analysis directly from Claude Desktop or Claude Code.

### Install via Claude Code

```bash
claude mcp add video-analyzer -s user -- uvx video-analyzer-mcp
```

### Install via Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "video-analyzer": {
      "command": "uvx",
      "args": ["video-analyzer-mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `video_analyze` | Full pipeline: download, extract frames, transcribe, analyze, fingerprint, storyboard |
| `video_extract_frames` | Extract representative frames using scene detection or fixed intervals |
| `video_transcribe` | Transcribe audio with OpenAI Whisper |
| `video_fingerprint` | Generate 8-field Stylistic Fingerprint v3 classification |
| `video_check_deps` | Verify all required dependencies are installed |

## Standalone Usage

```bash
pip install -r requirements.txt
python3 analyze_video.py "https://youtube.com/watch?v=..." --output-dir ./output
```

## Prerequisites

- **Python 3.10+**
- **FFmpeg** — `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux)
- **ANTHROPIC_API_KEY** — set as an environment variable

## Stylistic Fingerprint Fields

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
