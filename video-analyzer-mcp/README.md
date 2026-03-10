# Video Analyzer MCP Server

An MCP (Model Context Protocol) server that analyzes videos and generates storyboard breakdowns. Extract frames, transcribe audio, analyze visuals with Claude Vision, and compute stylistic fingerprints — all usable directly from Claude Desktop or Claude Code.

## Tools

| Tool | Description |
|------|-------------|
| `video_analyze` | Full pipeline: download → extract frames → transcribe → visual analysis → stylistic fingerprint → storyboard document |
| `video_extract_frames` | Extract representative frames using scene detection or fixed intervals |
| `video_transcribe` | Transcribe audio with OpenAI Whisper (timestamped, speaker-labeled) |
| `video_fingerprint` | Generate an 8-field Stylistic Fingerprint v3 classification |
| `video_check_deps` | Verify all required dependencies are installed |

## Prerequisites

- **Python 3.10+**
- **FFmpeg** — `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux)
- **ANTHROPIC_API_KEY** — set as an environment variable

## Installation

### For Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "video-analyzer": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/path/to/video-analyzer/video-analyzer-mcp",
        "video-analyzer-mcp"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

### For Claude Code

```bash
claude mcp add video-analyzer \
  -s user \
  -- uv run --directory /path/to/video-analyzer/video-analyzer-mcp video-analyzer-mcp
```

### Direct install with pip/uv

```bash
cd video-analyzer/video-analyzer-mcp
uv pip install -e .
```

## Usage Examples

Once installed, you can ask Claude:

- **"Analyze this video and create a storyboard"** → runs `video_analyze`
- **"Extract frames from this YouTube video"** → runs `video_extract_frames`
- **"Transcribe the audio from this video"** → runs `video_transcribe`
- **"What's the stylistic fingerprint of this video?"** → runs `video_fingerprint`
- **"Check if video analyzer dependencies are installed"** → runs `video_check_deps`

## Stylistic Fingerprint Fields

The fingerprint classifier produces 8 deterministic fields:

1. **Rendering Class** — Stylized 3D, Flat 2D, Minimalist Line Art, Textured 2D, Mixed Media, Photoreal
2. **World Type** — Stylized Real-World, Abstract Concept Space, Data/Presentation Space, Fictional Metaphor Universe
3. **Character Strategy** — None, Mascot-Led, Single Narrator, Single Protagonist Arc, Ensemble Cast
4. **Narrative Structure** — Direct Explanation, Step-by-Step, Problem→Solution, Analogy, Myth-Busting, etc.
5. **Visual Abstraction Index** — 1 (Photorealistic) to 5 (Maximum Abstraction)
6. **Visual Density** — Minimal, Sparse, Moderate, High
7. **Camera/Editing Language** — Cinematic, Social Vertical Punch, Presentation Deck, Static Slides, etc.
8. **Tonal Positioning** — Institutional, Corporate Professional, Gen Z Social, Child-Friendly, Dark Editorial

## Sharing

To share this MCP server with others:

1. Push the `video-analyzer` repo (including `video-analyzer-mcp/`) to GitHub
2. Others install with:
   ```bash
   git clone https://github.com/your-username/video-analyzer.git
   cd video-analyzer/video-analyzer-mcp
   uv pip install -e .
   ```
3. Then add to their Claude Desktop config or Claude Code as shown above
