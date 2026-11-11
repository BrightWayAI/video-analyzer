#!/usr/bin/env python3
"""
MCP Server for Video Analyzer.

Provides tools to analyze videos: extract frames, transcribe audio,
analyze visuals with Claude Vision, generate stylistic fingerprints,
and produce storyboard breakdowns.

Requires: FFmpeg installed on the system, ANTHROPIC_API_KEY env var.
"""

import functools
import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional

import anyio
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field, ConfigDict, field_validator

from video_analyzer_mcp import analyze_video as av
from video_analyzer_mcp import hosted

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
        if hosted.is_hosted():
            hosted.validate_remote_url(v)
        elif not av.is_url(v) and not os.path.isfile(v):
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

async def _run(fn, *args, **kwargs):
    """Run blocking work in a worker thread so the event loop stays responsive."""
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


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




class _Progress:
    """Progress reporting that works from worker threads, plus a heartbeat.

    Steps report through `callback` (safe to call from a worker thread). While a
    step runs, `heartbeat()` re-sends the latest progress every few seconds so
    long steps keep the connection active and the caller sees it's alive.
    """

    def __init__(self, ctx: Optional[Context], interval: float = 15.0, job=None):
        self.ctx = ctx      # None for a hosted background job: no request to report to
        self.job = job      # hosted.Job whose status 'still working' answers show
        self.interval = interval
        self.frac = 0.0
        self.msg = "Starting"

    async def report(self, frac: float, msg: str):
        self.frac, self.msg = frac, msg
        if self.job is not None:
            self.job.update(frac, msg)
        if self.ctx is None:
            return
        try:
            await self.ctx.report_progress(frac, 1.0, msg)
        except Exception:  # a client that went away mustn't kill the job
            pass

    def callback(self, frac: float, msg: str):
        try:
            anyio.from_thread.run(self.report, frac, msg)
        except Exception:
            pass

    async def _beat(self):
        while True:
            await anyio.sleep(self.interval)
            if self.ctx is None:
                continue
            try:
                await self.ctx.report_progress(self.frac, 1.0, f"{self.msg} (still working)")
            except Exception:
                pass

    async def run(self, fn, *args, **kwargs):
        """Run blocking fn in a worker thread with the heartbeat going."""
        # Catch inside the task group so a failure surfaces as itself rather
        # than wrapped in an ExceptionGroup ("unhandled errors in a TaskGroup").
        result, error = None, None
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._beat)
            try:
                result = await _run(fn, *args, **kwargs)
            except Exception as e:
                error = e
            finally:
                tg.cancel_scope.cancel()
        if error is not None:
            raise error
        return result


def _fail(e: Exception) -> None:
    """Re-raise a pipeline error as a tool error with a helpful hint."""
    raise RuntimeError(f"{e} — {_error_suggestion(e)}") from e


def _frames_url_rewrite(markdown: str, base: Optional[str]) -> str:
    """Point the storyboard's ../frames/ image links at the hosted copies."""
    return markdown.replace("](../frames/", f"]({base}/frames/") if base else markdown


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="video_analyze",
    structured_output=False,
    annotations={
        "title": "Analyze Video (Full Pipeline)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def video_analyze(params: AnalyzeVideoInput, ctx: Context):
    """Run the full video analysis pipeline: download, extract frames, transcribe
    audio, analyze visuals with Claude, generate the stylistic fingerprint, and
    produce a shot-by-shot storyboard.

    On the hosted server the result is a summary (with the fingerprint and links
    to the Word storyboard, frame images and a zip of all frames), the storyboard
    as Markdown, and a contact-sheet image of the shots. Locally it writes the
    storyboard to disk and returns its path.

    Processing time depends on video length (typically 1-3 minutes).
    """
    hosted.check_hosted_inputs(params.input_path, params.cookies_file)
    if hosted.is_hosted():
        return await hosted.run_job("video_analyze", params, lambda job: _analyze_hosted(params, _Progress(None, job=job)))

    _check_deps()
    progress = _Progress(ctx)
    await progress.report(0.05, "Starting video analysis pipeline")
    try:
        output_path = await progress.run(
            av.process_video,
            input_path=params.input_path,
            interval=params.interval,
            output_format=params.output_format.value,
            cookies_file=params.cookies_file,
            verbose=False,
        )
    except Exception as e:
        _fail(e)

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


async def _analyze_hosted(params: AnalyzeVideoInput, progress: "_Progress"):
    """Hosted pipeline: results inline, files behind expiring private links."""
    _check_deps()
    async with hosted.job_slot(), hosted.job_log("video_analyze", params.input_path) as log:
        await progress.report(0.02, "Downloading video")
        bundle = hosted.new_bundle()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path, title = await progress.run(_resolve_video, params.input_path, temp_dir)
                log.step("download")
                result = await progress.run(
                    av.analyze_video_file,
                    video_path, title, str(bundle.path / "frames"),
                    interval=params.interval,
                    max_duration=hosted.max_duration(),
                    max_frames=hosted.max_frames(),
                    progress=progress.callback,
                )
                log.step("analysis", shots=len(result.shots), frames_analyzed=result.frames_analyzed)
        except Exception as e:
            bundle.discard()
            _fail(e)

        await progress.report(0.95, "Writing storyboard")
        shots, meta, style = result.shots, result.metadata, result.visual_style
        markdown = await _run(av.generate_markdown_output, shots, meta, str(bundle.path / "frames"), style)
        await _run(av.generate_docx_output, shots, meta, str(bundle.path / "storyboard.docx"), style)
        sheet = await _run(av.make_contact_sheet, shots, str(bundle.path / "contact_sheet.jpg"))
        await _run(bundle.zip_dir, "frames", "frames.zip")
        log.step("outputs")

    base = bundle.url
    summary = {
        "status": "success",
        "title": meta.title,
        "duration_seconds": round(meta.duration, 1),
        "orientation": meta.orientation,
        "shots": len(shots),
        "frames_analyzed": result.frames_analyzed,
        "reading_level": meta.reading_level,
        "story_structure": meta.story_structure,
        "fingerprint": {k: v for k, v in style.items() if not k.startswith("legacy_")},
    }
    if base:
        summary["files"] = {
            "storyboard_docx": f"{base}/storyboard.docx",
            "contact_sheet": f"{base}/contact_sheet.jpg",
            "frames_zip": f"{base}/frames.zip",
            "links_expire_after_hours": hosted.file_ttl_hours(),
        }
    inline_sheet = await _run(hosted.shrink_jpeg, sheet) if sheet else None
    return hosted.fit_response(
        [json.dumps(summary, indent=2), _frames_url_rewrite(markdown, base)],
        inline_sheet,
        docs_hint="The full storyboard is in the storyboard_docx link above." if base else "",
    )


@mcp.tool(
    name="video_extract_frames",
    structured_output=False,
    annotations={
        "title": "Extract Video Frames",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def video_extract_frames(params: ExtractFramesInput, ctx: Context):
    """Extract representative frames from a video using scene detection or
    fixed intervals.

    Uses PySceneDetect to find scene boundaries, then extracts the middle frame
    of each scene. Falls back to 3-second intervals if no scenes are detected.

    Returns the shots with timestamps. On the hosted server each shot has a
    frame_url, and the result includes a contact-sheet image and a zip link.
    """

    hosted.check_hosted_inputs(params.input_path, params.cookies_file, params.output_dir)
    if hosted.is_hosted():
        return await hosted.run_job("video_extract_frames", params, lambda job: _extract_hosted(params, _Progress(None, job=job)))

    _check_deps()
    progress = _Progress(ctx)
    await progress.report(0.1, "Preparing to extract frames")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path, title = await progress.run(_resolve_video, params.input_path, temp_dir, params.cookies_file)
            safe_title = av.sanitize_filename(title)
            script_dir = Path(__file__).resolve().parent.parent.parent
            frames_dir = params.output_dir or str(script_dir / "output" / safe_title / "frames")
            os.makedirs(frames_dir, exist_ok=True)
            duration = await _run(av.get_video_duration, video_path)
            width, height, orientation = await _run(av.get_video_dimensions, video_path)
            await progress.report(0.3, "Detecting scenes and extracting frames")
            shots = await progress.run(_extract_shots, video_path, frames_dir, params.interval, duration)
    except Exception as e:
        _fail(e)

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


async def _extract_hosted(params: ExtractFramesInput, progress: "_Progress"):
    _check_deps()
    async with hosted.job_slot(), hosted.job_log("video_extract_frames", params.input_path) as log:
        await progress.report(0.05, "Downloading video")
        bundle = hosted.new_bundle()
        frames_dir = bundle.path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path, title = await progress.run(_resolve_video, params.input_path, temp_dir)
                log.step("download")
                duration = await _run(av.get_video_duration, video_path)
                hosted.check_duration(duration)
                width, height, orientation = await _run(av.get_video_dimensions, video_path)
                await progress.report(0.3, "Detecting scenes and extracting frames")
                shots = await progress.run(_extract_shots, video_path, str(frames_dir), params.interval, duration)
                log.step("frames", shots=len(shots))
        except Exception as e:
            bundle.discard()
            _fail(e)
        sheet = await _run(av.make_contact_sheet, shots, str(bundle.path / "contact_sheet.jpg"))
        await _run(bundle.zip_dir, "frames", "frames.zip")

    base = bundle.url
    shot_dicts = _shots_to_dicts(shots)
    for d in shot_dicts:
        name = Path(d.pop("frame_path") or "").name
        if base and name:
            d["frame_url"] = f"{base}/frames/{name}"
    summary = {
        "status": "success",
        "title": title,
        "duration_seconds": round(duration, 1),
        "dimensions": f"{width}x{height}",
        "orientation": orientation,
        "shots_extracted": len(shots),
        "shots": shot_dicts,
    }
    if base:
        summary["files"] = {
            "contact_sheet": f"{base}/contact_sheet.jpg",
            "frames_zip": f"{base}/frames.zip",
            "links_expire_after_hours": hosted.file_ttl_hours(),
        }
    inline_sheet = await _run(hosted.shrink_jpeg, sheet) if sheet else None
    return hosted.fit_response([json.dumps(summary, indent=2)], inline_sheet)


@mcp.tool(
    name="video_transcribe",
    structured_output=False,
    annotations={
        "title": "Transcribe Video Audio",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def video_transcribe(params: TranscribeInput, ctx: Context):
    """Transcribe the audio track of a video using OpenAI Whisper (runs locally,
    no OpenAI key needed).

    Returns timestamped transcript segments with inferred speaker labels
    (NARRATOR, TEACHER, CHILD, SPEAKER). 'base' is fastest; 'large' is most
    accurate (the hosted server offers 'base' and 'small').
    """
    hosted.check_hosted_inputs(params.input_path, params.cookies_file)
    if hosted.is_hosted() and params.model_size.value not in hosted.HOSTED_WHISPER_MODELS:
        raise ValueError("The hosted server supports Whisper 'base' and 'small' only.")
    if hosted.is_hosted():
        return await hosted.run_job("video_transcribe", params, lambda job: _transcribe(params, _Progress(None, job=job)))
    return await _transcribe(params, _Progress(ctx))


async def _transcribe(params: TranscribeInput, progress: "_Progress"):
    _check_deps()
    async with hosted.job_slot(), hosted.job_log("video_transcribe", params.input_path) as log:
        await progress.report(0.1, "Downloading video")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path, title = await progress.run(
                    _resolve_video, params.input_path, temp_dir,
                    None if hosted.is_hosted() else params.cookies_file,
                )
                log.step("download")
                hosted.check_duration(await _run(av.get_video_duration, video_path))
                await progress.report(0.3, "Transcribing audio")
                segments = await progress.run(
                    av.transcribe_audio, video_path, model_size=params.model_size.value, verbose=False
                )
                log.step("transcribe", segments=len(segments))
        except Exception as e:
            _fail(e)

    full_text = " ".join(seg.text for seg in segments)
    return json.dumps({
        "status": "success",
        "title": title,
        "model_used": params.model_size.value,
        "segments": len(segments),
        "word_count": len(full_text.split()),
        "transcript": _segments_to_dicts(segments),
        "full_text": full_text,
    }, indent=2)


@mcp.tool(
    name="video_fingerprint",
    structured_output=False,
    annotations={
        "title": "Stylistic Fingerprint Analysis",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def video_fingerprint(params: FingerprintInput, ctx: Context):
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
    """
    hosted.check_hosted_inputs(params.input_path, params.cookies_file)
    if hosted.is_hosted():
        return await hosted.run_job("video_fingerprint", params, lambda job: _fingerprint(params, _Progress(None, job=job)))
    return await _fingerprint(params, _Progress(ctx))


async def _fingerprint(params: FingerprintInput, progress: "_Progress"):
    _check_deps()
    async with hosted.job_slot(), hosted.job_log("video_fingerprint", params.input_path) as log:
        await progress.report(0.05, "Downloading video")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path, title = await progress.run(
                    _resolve_video, params.input_path, temp_dir,
                    None if hosted.is_hosted() else params.cookies_file,
                )
                log.step("download")
                result = await progress.run(
                    av.analyze_video_file,
                    video_path, title, os.path.join(temp_dir, "frames"),
                    interval=params.interval,
                    max_duration=hosted.max_duration() if hosted.is_hosted() else None,
                    max_frames=hosted.max_frames() if hosted.is_hosted() else None,
                    progress=progress.callback,
                )
                log.step("analysis", shots=len(result.shots), frames_analyzed=result.frames_analyzed)
        except Exception as e:
            _fail(e)

    meta = result.metadata
    return json.dumps({
        "status": "success",
        "title": title,
        "duration_seconds": round(meta.duration, 1),
        "orientation": meta.orientation,
        "shots_analyzed": result.frames_analyzed,
        "reading_level": meta.reading_level,
        "story_structure": meta.story_structure,
        "fingerprint": {k: v for k, v in result.visual_style.items() if not k.startswith("legacy_")},
    }, indent=2)


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
    details = [
        {"dependency": dep, "install": install_hints.get(dep, f"pip install {dep}")}
        for dep in missing
    ]
    return json.dumps({
        "status": "missing_dependencies",
        "missing": details,
        "message": f"{len(missing)} dependency(ies) missing. Install them to unlock full functionality.",
    }, indent=2)


# Local installs only: on the hosted server this would just use up a buyer's
# monthly calls, and the operator keeps the dependencies in order.
if not hosted.is_hosted():
    mcp.tool(
        name="video_check_deps",
        annotations={
            "title": "Check Dependencies",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )(video_check_deps)


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _error_suggestion(e: Exception) -> str:
    """Return a helpful suggestion based on the error type."""
    import subprocess

    msg = str(e).lower()
    if isinstance(e, subprocess.CalledProcessError) and ("ffmpeg" in msg or "ffprobe" in msg):
        return "FFmpeg couldn't read this video. It may be corrupt or in an unsupported format; try another copy or a different link."
    if isinstance(e, FileNotFoundError) and ("ffmpeg" in msg or "ffprobe" in msg):
        return "FFmpeg is not installed. Install it: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)"
    if "anthropic_api_key" in msg or "api_key" in msg:
        return "Set the ANTHROPIC_API_KEY environment variable before running the server."
    if "download failed" in msg and av._looks_blocked(msg):
        return ("The video site refused the download (bot check or login wall). Try again in a "
                "minute; if it keeps happening, send a direct link to the video file.")
    if "download failed" in msg:
        return "Check that the URL is a public video. Private, age-restricted and members-only videos can't be downloaded."
    if "limit is" in msg:
        return "Trim the video or send a shorter clip."
    if "whisper" in msg:
        return "OpenAI Whisper is not installed. Run: pip install openai-whisper"
    if "no such file" in msg or "not found" in msg:
        return "The specified file was not found. Check the path and try again."
    return "Check the error message above and try again."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run()


if __name__ == "__main__":
    main()
