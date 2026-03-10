#!/usr/bin/env python3
"""
MCP Server for Video Analyzer.

Provides tools to analyze videos: extract frames, transcribe audio,
analyze visuals with Claude Vision, generate stylistic fingerprints,
and produce storyboard breakdowns.

Requires: FFmpeg installed on the system, ANTHROPIC_API_KEY env var.
"""

import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field, ConfigDict, field_validator

from video_analyzer_mcp import analyze_video as av

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP("video_analyzer_mcp")

# ---------------------------------------------------------------------------
# Enums & Input Models
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    DOCX = "docx"
    MARKDOWN = "md"


class WhisperModel(str, Enum):
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class AnalyzeVideoInput(BaseModel):
    """Input for full video analysis pipeline."""
    model_config = ConfigDict(str_strip_whitespace=True)

    input_path: str = Field(
        ...,
        description="Path to a local video file or a URL (YouTube, Vimeo, etc.)",
        min_length=1,
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.DOCX,
        description="Output format: 'docx' for Word document or 'md' for Markdown",
    )
    interval: Optional[float] = Field(
        default=None,
        description="Extract frames every N seconds instead of using scene detection. Leave empty for automatic scene detection.",
        gt=0,
    )
    cookies_file: Optional[str] = Field(
        default=None,
        description="Path to cookies.txt for downloading from authenticated sites",
    )

    @field_validator("input_path")
    @classmethod
    def validate_input(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("input_path cannot be empty")
        if not av.is_url(v) and not os.path.isfile(v):
            raise ValueError(f"File not found: {v}")
        return v


class ExtractFramesInput(BaseModel):
    """Input for frame extraction only."""
    model_config = ConfigDict(str_strip_whitespace=True)

    input_path: str = Field(
        ...,
        description="Path to a local video file or URL",
        min_length=1,
    )
    interval: Optional[float] = Field(
        default=None,
        description="Extract frames every N seconds. If omitted, uses scene detection.",
        gt=0,
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save extracted frames. Defaults to ./output/<title>/frames",
    )
    cookies_file: Optional[str] = Field(
        default=None,
        description="Path to cookies.txt for authenticated downloads",
    )


class TranscribeInput(BaseModel):
    """Input for audio transcription only."""
    model_config = ConfigDict(str_strip_whitespace=True)

    input_path: str = Field(
        ...,
        description="Path to a local video/audio file or URL",
        min_length=1,
    )
    model_size: WhisperModel = Field(
        default=WhisperModel.BASE,
        description="Whisper model size: 'base' (fastest), 'small', 'medium', or 'large' (most accurate)",
    )
    cookies_file: Optional[str] = Field(
        default=None,
        description="Path to cookies.txt for authenticated downloads",
    )


class FingerprintInput(BaseModel):
    """Input for stylistic fingerprint analysis."""
    model_config = ConfigDict(str_strip_whitespace=True)

    input_path: str = Field(
        ...,
        description="Path to a local video file or URL",
        min_length=1,
    )
    interval: Optional[float] = Field(
        default=None,
        description="Frame extraction interval in seconds. If omitted, uses scene detection.",
        gt=0,
    )
    cookies_file: Optional[str] = Field(
        default=None,
        description="Path to cookies.txt for authenticated downloads",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_video(input_path: str, temp_dir: str, cookies_file: Optional[str] = None):
    """Download URL or validate local file. Returns (video_path, title)."""
    if av.is_url(input_path):
        video_path, title = av.download_video(input_path, temp_dir, cookies_file, verbose=False)
    else:
        video_path = input_path
        title = av.get_video_title(input_path)
    return video_path, title


def _extract_shots(video_path: str, frames_dir: str, interval: Optional[float], duration: float):
    """Run scene detection or interval extraction. Returns list of Shot objects."""
    if interval:
        frames = av.extract_frames_at_interval(video_path, frames_dir, interval, verbose=False)
        return av.create_shots_from_interval(frames, duration, interval)

    scenes = av.detect_scenes(video_path, verbose=False)
    if not scenes:
        frames = av.extract_frames_at_interval(video_path, frames_dir, 3.0, verbose=False)
        return av.create_shots_from_interval(frames, duration, 3.0)

    return av.create_shots_from_scenes(scenes, video_path, frames_dir, verbose=False)


def _shots_to_dicts(shots: list[av.Shot]) -> list[dict]:
    """Convert Shot objects to serializable dicts."""
    return [
        {
            "number": s.number,
            "timestamp": s.timestamp_str,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "frame_path": s.frame_path,
            "visual_description": s.visual_description,
            "dialogue": s.dialogue,
        }
        for s in shots
    ]


def _segments_to_dicts(segments: list[av.TranscriptSegment]) -> list[dict]:
    """Convert TranscriptSegment objects to serializable dicts."""
    return [
        {
            "start_time": round(seg.start_time, 2),
            "end_time": round(seg.end_time, 2),
            "text": seg.text,
            "speaker": av.infer_speaker(seg.text),
        }
        for seg in segments
    ]


def _check_deps():
    """Raise if critical dependencies are missing."""
    missing = av.check_dependencies()
    critical = [m for m in missing if "yt-dlp" not in m]
    if critical:
        raise RuntimeError(
            f"Missing required dependencies: {', '.join(critical)}. "
            "Install them and try again."
        )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="video_analyze",
    annotations={
        "title": "Analyze Video (Full Pipeline)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def video_analyze(params: AnalyzeVideoInput, ctx: Context) -> str:
    """Run the full video analysis pipeline: download, extract frames, transcribe
    audio, analyze visuals with Claude, generate stylistic fingerprint, and
    produce a storyboard breakdown document.

    Returns the path to the generated output file and a summary of the analysis
    including metadata, stylistic fingerprint, and shot count.

    Requires: FFmpeg, ANTHROPIC_API_KEY env var.
    Processing time depends on video length (typically 1-5 minutes).
    """
    _check_deps()

    await ctx.report_progress(0.05, "Starting video analysis pipeline...")

    try:
        output_path = av.process_video(
            input_path=params.input_path,
            interval=params.interval,
            output_format=params.output_format.value,
            cookies_file=params.cookies_file,
            verbose=False,
        )

        # Read back some metadata for the response
        output_dir = Path(output_path).parent.parent
        frames_dir = output_dir / "frames"
        frame_count = len(list(frames_dir.glob("*.jpg"))) + len(list(frames_dir.glob("*.png")))

        return json.dumps({
            "status": "success",
            "output_path": output_path,
            "output_format": params.output_format.value,
            "frames_extracted": frame_count,
            "output_directory": str(output_dir),
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "suggestion": _error_suggestion(e),
        }, indent=2)


@mcp.tool(
    name="video_extract_frames",
    annotations={
        "title": "Extract Video Frames",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def video_extract_frames(params: ExtractFramesInput, ctx: Context) -> str:
    """Extract representative frames from a video using scene detection or
    fixed intervals.

    Uses PySceneDetect with ContentDetector and AdaptiveDetector to find
    scene boundaries, then extracts the middle frame of each scene.
    Falls back to 3-second intervals if no scenes are detected.

    Returns a list of extracted shots with frame paths and timestamps.
    """
    _check_deps()

    await ctx.report_progress(0.1, "Preparing to extract frames...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path, title = _resolve_video(params.input_path, temp_dir, params.cookies_file)

            safe_title = av.sanitize_filename(title)
            script_dir = Path(__file__).resolve().parent.parent.parent
            frames_dir = params.output_dir or str(script_dir / "output" / safe_title / "frames")
            os.makedirs(frames_dir, exist_ok=True)

            duration = av.get_video_duration(video_path)
            width, height, orientation = av.get_video_dimensions(video_path)

            await ctx.report_progress(0.3, "Detecting scenes and extracting frames...")
            shots = _extract_shots(video_path, frames_dir, params.interval, duration)

        return json.dumps({
            "status": "success",
            "title": title,
            "duration_seconds": round(duration, 1),
            "dimensions": f"{width}x{height}",
            "orientation": orientation,
            "shots_extracted": len(shots),
            "frames_directory": frames_dir,
            "shots": _shots_to_dicts(shots),
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "suggestion": _error_suggestion(e),
        }, indent=2)


@mcp.tool(
    name="video_transcribe",
    annotations={
        "title": "Transcribe Video Audio",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def video_transcribe(params: TranscribeInput, ctx: Context) -> str:
    """Transcribe the audio track of a video using OpenAI Whisper.

    Returns timestamped transcript segments with inferred speaker labels
    (NARRATOR, TEACHER, CHILD, SPEAKER). Supports multiple Whisper model
    sizes trading off speed vs accuracy.

    The 'base' model is fastest; 'large' is most accurate but slower.
    """
    _check_deps()

    await ctx.report_progress(0.1, "Loading Whisper model...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path, title = _resolve_video(params.input_path, temp_dir, params.cookies_file)

            await ctx.report_progress(0.3, "Transcribing audio...")
            segments = av.transcribe_audio(video_path, model_size=params.model_size.value, verbose=False)

        full_text = " ".join(seg.text for seg in segments)
        word_count = len(full_text.split())

        return json.dumps({
            "status": "success",
            "title": title,
            "model_used": params.model_size.value,
            "segments": len(segments),
            "word_count": word_count,
            "transcript": _segments_to_dicts(segments),
            "full_text": full_text,
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "suggestion": _error_suggestion(e),
        }, indent=2)


@mcp.tool(
    name="video_fingerprint",
    annotations={
        "title": "Stylistic Fingerprint Analysis",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def video_fingerprint(params: FingerprintInput, ctx: Context) -> str:
    """Generate a Stylistic Fingerprint (v3) for a video.

    Performs frame extraction, transcription, and visual analysis, then runs
    the deterministic fingerprint classifier to produce 8 classification fields:

    1. Rendering Class (e.g., Stylized 3D, Flat 2D, Mixed Media)
    2. World Type (e.g., Stylized Real-World, Abstract Concept Space)
    3. Character Strategy (e.g., Mascot-Led, Single Narrator, Ensemble Cast)
    4. Narrative Structure (e.g., Direct Explanation, Problem-Solution)
    5. Visual Abstraction Index (1-5 scale, Photorealistic to Maximum Abstraction)
    6. Visual Density (Minimal to High)
    7. Camera/Editing Language (e.g., Cinematic, Social Vertical Punch)
    8. Tonal Positioning (e.g., Institutional, Gen Z Social, Child-Friendly)

    Requires: FFmpeg, ANTHROPIC_API_KEY.
    """
    _check_deps()

    await ctx.report_progress(0.05, "Starting fingerprint analysis...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path, title = _resolve_video(params.input_path, temp_dir, params.cookies_file)

            duration = av.get_video_duration(video_path)
            width, height, orientation = av.get_video_dimensions(video_path)

            # Extract frames
            await ctx.report_progress(0.15, "Extracting frames...")
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            shots = _extract_shots(video_path, frames_dir, params.interval, duration)

            # Transcribe
            await ctx.report_progress(0.35, "Transcribing audio...")
            transcript = av.transcribe_audio(video_path, verbose=False)

            # Analyze visuals
            await ctx.report_progress(0.55, "Analyzing frames with Claude Vision...")
            shots = av.analyze_all_frames(shots, orientation, verbose=False)

            # Align transcript
            shots = av.align_transcript_to_shots(shots, transcript)

            # Content analysis for reading level
            await ctx.report_progress(0.75, "Running content analysis...")
            key_details, reading_level, story_structure = av.analyze_video_content(
                transcript, shots, title, verbose=False
            )

            # Stylistic fingerprint
            await ctx.report_progress(0.9, "Computing stylistic fingerprint...")
            fingerprint = av.analyze_stylistic_fingerprint(
                shots, transcript, duration,
                orientation=orientation,
                reading_level=reading_level,
            )

        # Remove legacy v2 keys for clean output
        clean_fingerprint = {
            k: v for k, v in fingerprint.items()
            if not k.startswith("legacy_")
        }

        return json.dumps({
            "status": "success",
            "title": title,
            "duration_seconds": round(duration, 1),
            "orientation": orientation,
            "shots_analyzed": len(shots),
            "reading_level": reading_level,
            "story_structure": story_structure,
            "fingerprint": clean_fingerprint,
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "suggestion": _error_suggestion(e),
        }, indent=2)


@mcp.tool(
    name="video_check_deps",
    annotations={
        "title": "Check Dependencies",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def video_check_deps() -> str:
    """Check if all required dependencies (FFmpeg, Whisper, etc.) are installed.

    Returns a list of any missing dependencies with installation instructions.
    Use this before running other video analysis tools.
    """
    missing = av.check_dependencies()

    if not missing:
        return json.dumps({
            "status": "all_dependencies_installed",
            "message": "All required dependencies are available.",
        }, indent=2)

    install_hints = {
        "ffmpeg (system package)": "brew install ffmpeg (macOS) | apt-get install ffmpeg (Linux)",
        "openai-whisper": "pip install openai-whisper",
        "scenedetect[opencv]": "pip install scenedetect[opencv]",
        "anthropic": "pip install anthropic",
        "Pillow": "pip install Pillow",
        "yt-dlp (for URL support)": "pip install yt-dlp",
    }

    details = []
    for dep in missing:
        hint = install_hints.get(dep, f"pip install {dep}")
        details.append({"dependency": dep, "install": hint})

    return json.dumps({
        "status": "missing_dependencies",
        "missing": details,
        "message": f"{len(missing)} dependency(ies) missing. Install them to unlock full functionality.",
    }, indent=2)


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _error_suggestion(e: Exception) -> str:
    """Return a helpful suggestion based on the error type."""
    msg = str(e).lower()
    if "ffmpeg" in msg or "ffprobe" in msg:
        return "FFmpeg is not installed. Install it: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)"
    if "anthropic_api_key" in msg or "api_key" in msg:
        return "Set the ANTHROPIC_API_KEY environment variable before running the server."
    if "whisper" in msg:
        return "OpenAI Whisper is not installed. Run: pip install openai-whisper"
    if "no such file" in msg or "not found" in msg:
        return "The specified file was not found. Check the path and try again."
    if "yt-dlp" in msg or "download" in msg:
        return "yt-dlp may not be installed, or the URL is invalid. Run: pip install yt-dlp"
    return "Check the error message above and ensure all dependencies are installed."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run()


if __name__ == "__main__":
    main()
