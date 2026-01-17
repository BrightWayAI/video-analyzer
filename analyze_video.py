#!/usr/bin/env python3
"""
Video-to-Storyboard Analyzer for Riptoes

Analyzes educational videos and outputs a shot-by-shot breakdown table.
Helps reverse-engineer effective educational videos to inform storyboard creation.

Usage:
    python analyze_video.py input.mp4 -o breakdown.md
    python analyze_video.py "https://youtube.com/watch?v=xxx" -o breakdown.md
"""

import argparse
import base64
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Third-party imports - these will fail gracefully with helpful messages
try:
    import whisper
except ImportError:
    whisper = None

try:
    from scenedetect import detect, ContentDetector, AdaptiveDetector
except ImportError:
    detect = None
    ContentDetector = None
    AdaptiveDetector = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from PIL import Image
except ImportError:
    Image = None

# docx imports are done lazily in generate_docx_output to avoid import issues
Document = None


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Shot:
    """Represents a single shot/scene in the video."""
    number: int
    start_time: float  # seconds
    end_time: float    # seconds
    frame_path: Optional[str] = None
    visual_description: str = ""
    dialogue: str = ""

    @property
    def timestamp_str(self) -> str:
        """Format timestamp as M:SS-M:SS."""
        return f"{self._format_time(self.start_time)}-{self._format_time(self.end_time)}"

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Convert seconds to M:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"


@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed audio."""
    start_time: float
    end_time: float
    text: str
    speaker: str = ""  # Will be inferred


# =============================================================================
# Utility Functions
# =============================================================================

def check_dependencies() -> list[str]:
    """Check for required dependencies and return list of missing ones."""
    missing = []

    # Check FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("ffmpeg (system package)")

    # Check Python packages
    if whisper is None:
        missing.append("openai-whisper")
    if detect is None:
        missing.append("scenedetect[opencv]")
    if anthropic is None:
        missing.append("anthropic")
    if Image is None:
        missing.append("Pillow")

    # Check for yt-dlp (only needed for URLs)
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("yt-dlp (for URL support)")

    return missing


def is_url(path: str) -> bool:
    """Check if the input is a URL."""
    return path.startswith(("http://", "https://", "www."))


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using FFprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ],
        capture_output=True,
        text=True,
        check=True
    )
    return float(result.stdout.strip())


def get_video_title(video_path: str) -> str:
    """Extract video title from path or metadata."""
    path = Path(video_path)
    return path.stem


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename/folder name."""
    # Replace problematic characters with safe alternatives
    replacements = {
        '/': '-',
        '\\': '-',
        ':': '-',
        '*': '',
        '?': '',
        '"': '',
        '<': '',
        '>': '',
        '|': '-',
        '\n': ' ',
        '\r': '',
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    # Remove leading/trailing whitespace and dots
    name = name.strip().strip('.')
    # Limit length
    if len(name) > 100:
        name = name[:100]
    return name or "untitled"


# =============================================================================
# Video Download (for URLs)
# =============================================================================

def download_video(url: str, output_dir: str, verbose: bool = False) -> tuple[str, str]:
    """
    Download video from URL using yt-dlp.

    Returns:
        Tuple of (video_path, video_title)
    """
    if verbose:
        print(f"Downloading video from: {url}")

    # First get the title
    title_result = subprocess.run(
        ["yt-dlp", "--get-title", url],
        capture_output=True,
        text=True,
        check=True
    )
    title = title_result.stdout.strip()

    # Download the video with a safe filename to avoid FFmpeg issues with special chars
    safe_filename = "video_download.mp4"
    output_path = os.path.join(output_dir, safe_filename)
    subprocess.run(
        [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o", output_path,
            "--no-playlist",
            "--merge-output-format", "mp4",
            url
        ],
        check=True,
        capture_output=not verbose
    )

    if os.path.exists(output_path):
        return output_path, title

    # Fallback: find any video file
    for file in os.listdir(output_dir):
        if file.endswith((".mp4", ".mkv", ".webm", ".avi")):
            return os.path.join(output_dir, file), title

    raise RuntimeError("Failed to find downloaded video file")


# =============================================================================
# Frame Extraction
# =============================================================================

def extract_frames_at_interval(
    video_path: str,
    output_dir: str,
    interval: float = 3.0,
    verbose: bool = False
) -> list[tuple[float, str]]:
    """
    Extract frames at regular intervals using FFmpeg.

    Returns:
        List of (timestamp, frame_path) tuples
    """
    if verbose:
        print(f"Extracting frames every {interval} seconds...")

    os.makedirs(output_dir, exist_ok=True)

    # Extract frames using FFmpeg
    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps=1/{interval}",
            "-frame_pts", "1",
            os.path.join(output_dir, "frame_%04d.jpg"),
            "-y"
        ],
        check=True,
        capture_output=not verbose
    )

    # Collect frame paths with timestamps
    frames = []
    for i, frame_file in enumerate(sorted(os.listdir(output_dir))):
        if frame_file.startswith("frame_") and frame_file.endswith(".jpg"):
            timestamp = i * interval
            frame_path = os.path.join(output_dir, frame_file)
            frames.append((timestamp, frame_path))

    if verbose:
        print(f"Extracted {len(frames)} frames")

    return frames


def extract_frame_at_time(
    video_path: str,
    timestamp: float,
    output_path: str
) -> str:
    """Extract a single frame at a specific timestamp."""
    subprocess.run(
        [
            "ffmpeg", "-ss", str(timestamp),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            output_path,
            "-y"
        ],
        check=True,
        capture_output=True
    )
    return output_path


# =============================================================================
# Scene Detection
# =============================================================================

def detect_scenes(
    video_path: str,
    verbose: bool = False
) -> list[tuple[float, float]]:
    """
    Detect scene boundaries using PySceneDetect.

    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    if detect is None:
        raise ImportError("scenedetect is required for scene detection")

    if verbose:
        print("Detecting scene boundaries...")

    # Use ContentDetector for detecting cuts
    scene_list = detect(video_path, ContentDetector(threshold=27.0))

    if not scene_list:
        # Fallback: try AdaptiveDetector for gradual transitions
        scene_list = detect(video_path, AdaptiveDetector())

    # Convert to list of (start, end) tuples in seconds
    scenes = []
    for scene in scene_list:
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        scenes.append((start_time, end_time))

    if verbose:
        print(f"Detected {len(scenes)} scenes")

    return scenes


def create_shots_from_scenes(
    scenes: list[tuple[float, float]],
    video_path: str,
    frames_dir: str,
    verbose: bool = False
) -> list[Shot]:
    """Create Shot objects from detected scenes, extracting representative frames."""
    if verbose:
        print("Extracting representative frames for each scene...")

    os.makedirs(frames_dir, exist_ok=True)
    shots = []

    # Get video duration to avoid extracting frames past the end
    duration = get_video_duration(video_path)

    for i, (start, end) in enumerate(scenes):
        # Extract frame from middle of scene, but stay safely within video bounds
        mid_time = (start + end) / 2
        # Clamp to 0.5 seconds before video end to avoid FFmpeg edge issues
        mid_time = min(mid_time, duration - 0.5)
        mid_time = max(mid_time, 0.1)  # Also avoid very start

        frame_path = os.path.join(frames_dir, f"shot_{i+1:04d}.jpg")
        extract_frame_at_time(video_path, mid_time, frame_path)

        shot = Shot(
            number=i + 1,
            start_time=start,
            end_time=end,
            frame_path=frame_path
        )
        shots.append(shot)

    return shots


def create_shots_from_interval(
    frames: list[tuple[float, str]],
    video_duration: float,
    interval: float
) -> list[Shot]:
    """Create Shot objects from interval-extracted frames."""
    shots = []

    for i, (timestamp, frame_path) in enumerate(frames):
        end_time = min(timestamp + interval, video_duration)
        shot = Shot(
            number=i + 1,
            start_time=timestamp,
            end_time=end_time,
            frame_path=frame_path
        )
        shots.append(shot)

    return shots


# =============================================================================
# Audio Transcription
# =============================================================================

def transcribe_audio(
    video_path: str,
    model_size: str = "base",
    verbose: bool = False
) -> list[TranscriptSegment]:
    """
    Transcribe audio using OpenAI Whisper.

    Returns:
        List of TranscriptSegment objects with timestamps
    """
    if whisper is None:
        raise ImportError("openai-whisper is required for transcription")

    if verbose:
        print(f"Transcribing audio using Whisper ({model_size} model)...")

    # Load model
    model = whisper.load_model(model_size)

    # Transcribe with word-level timestamps
    result = model.transcribe(
        video_path,
        word_timestamps=True,
        verbose=verbose
    )

    # Convert to TranscriptSegment objects
    segments = []
    for segment in result["segments"]:
        ts = TranscriptSegment(
            start_time=segment["start"],
            end_time=segment["end"],
            text=segment["text"].strip()
        )
        segments.append(ts)

    if verbose:
        print(f"Transcribed {len(segments)} segments")

    return segments


def infer_speaker(text: str, context: list[TranscriptSegment] = None) -> str:
    """
    Infer speaker type from text content.

    Returns speaker label like "NARRATOR:", "CHILD:", "TEACHER:", etc.
    """
    text_lower = text.lower()

    # Check for common narrator patterns
    narrator_patterns = [
        "today we", "let's learn", "welcome to", "in this video",
        "once upon", "our story", "let me tell", "did you know"
    ]
    if any(pattern in text_lower for pattern in narrator_patterns):
        return "NARRATOR"

    # Check for question patterns (often teachers)
    if "?" in text and any(w in text_lower for w in ["class", "who can", "what do you", "can anyone"]):
        return "TEACHER"

    # Check for child-like speech patterns
    child_patterns = ["i saw", "i found", "look at", "wow", "cool", "mommy", "daddy"]
    if any(pattern in text_lower for pattern in child_patterns):
        return "CHILD"

    # Default to generic speaker
    return "SPEAKER"


# =============================================================================
# Visual Analysis with Claude Vision
# =============================================================================

def analyze_frame_with_claude(
    frame_path: str,
    client: "anthropic.Anthropic",
    verbose: bool = False
) -> str:
    """
    Analyze a video frame using Claude Vision API.

    Returns:
        Visual description of the frame
    """
    if verbose:
        print(f"  Analyzing frame: {os.path.basename(frame_path)}")

    # Read and encode image
    with open(frame_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Determine media type
    ext = Path(frame_path).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }.get(ext, "image/jpeg")

    # Create prompt for educational video analysis
    prompt = """Analyze this frame from an educational video and provide a concise visual description.

Guidelines:
- Describe the setting/location (classroom, outdoors, animated background, etc.)
- Describe characters visible: approximate age, appearance, what they're doing
- Don't assume names - use descriptions like "young girl with curly hair" or "adult man in blue shirt"
- Note camera framing if significant (wide shot, close-up, two-shot, etc.)
- Keep it concrete, factual, and under 30 words
- Focus on what would be relevant for a storyboard

Example output: "Wide shot of sunny classroom. Teacher at whiteboard pointing to math equation. Six students at desks, hands raised."

Provide ONLY the description, no preamble or explanation."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    return response.content[0].text.strip()


def analyze_all_frames(
    shots: list[Shot],
    verbose: bool = False
) -> list[Shot]:
    """Analyze all shot frames using Claude Vision API."""
    if anthropic is None:
        raise ImportError("anthropic package is required for visual analysis")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    client = anthropic.Anthropic(api_key=api_key)

    if verbose:
        print(f"Analyzing {len(shots)} frames with Claude Vision...")

    for shot in shots:
        if shot.frame_path and os.path.exists(shot.frame_path):
            shot.visual_description = analyze_frame_with_claude(
                shot.frame_path,
                client,
                verbose
            )

    return shots


# =============================================================================
# Alignment: Match Transcripts to Shots
# =============================================================================

def align_transcript_to_shots(
    shots: list[Shot],
    transcript: list[TranscriptSegment]
) -> list[Shot]:
    """
    Align transcribed audio segments to visual shots based on timestamps.

    Each shot gets the dialogue that overlaps with its time range.
    """
    for shot in shots:
        # Find all transcript segments that overlap with this shot
        overlapping_segments = []

        for segment in transcript:
            # Check for overlap
            if segment.end_time > shot.start_time and segment.start_time < shot.end_time:
                overlapping_segments.append(segment)

        # Combine overlapping segments into dialogue
        if overlapping_segments:
            dialogue_parts = []
            for seg in overlapping_segments:
                speaker = infer_speaker(seg.text)
                dialogue_parts.append(f'{speaker}: "{seg.text}"')
            shot.dialogue = " ".join(dialogue_parts)
        else:
            shot.dialogue = "(no dialogue)"

    return shots


# =============================================================================
# Output Generation
# =============================================================================

def generate_markdown_output(
    shots: list[Shot],
    title: str,
    duration: float
) -> str:
    """Generate markdown table output."""
    # Format duration
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    duration_str = f"{minutes}:{seconds:02d}"

    # Build markdown
    lines = [
        f"# Video Analysis: {title}",
        f"**Runtime:** {duration_str}",
        f"**Shots:** {len(shots)}",
        "",
        "| Shot | Timestamp | Visual | Voiceover/Dialogue |",
        "|------|-----------|--------|-------------------|"
    ]

    for shot in shots:
        # Escape pipe characters in content
        visual = shot.visual_description.replace("|", "\\|")
        dialogue = shot.dialogue.replace("|", "\\|")

        lines.append(
            f"| {shot.number} | {shot.timestamp_str} | {visual} | {dialogue} |"
        )

    return "\n".join(lines)


def generate_docx_output(
    shots: list[Shot],
    title: str,
    duration: float,
    output_path: str
) -> str:
    """Generate Word document (.docx) output with table."""
    # Lazy import to avoid import issues
    try:
        from docx import Document
        from docx.shared import Inches, Pt
    except ImportError:
        raise ImportError("python-docx is required for Word document output. Install with: pip install python-docx")

    # Format duration
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    duration_str = f"{minutes}:{seconds:02d}"

    # Create document
    doc = Document()

    # Add title
    doc.add_heading(f"Video Analysis: {title}", 0)

    # Add metadata
    doc.add_paragraph(f"Runtime: {duration_str}")
    doc.add_paragraph(f"Shots: {len(shots)}")
    doc.add_paragraph()  # Spacer

    # Create table
    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"

    # Header row
    header_cells = table.rows[0].cells
    header_cells[0].text = "Shot"
    header_cells[1].text = "Timestamp"
    header_cells[2].text = "Visual"
    header_cells[3].text = "Voiceover/Dialogue"

    # Make headers bold
    for cell in header_cells:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True

    # Add data rows
    for shot in shots:
        row_cells = table.add_row().cells
        row_cells[0].text = str(shot.number)
        row_cells[1].text = shot.timestamp_str
        row_cells[2].text = shot.visual_description
        row_cells[3].text = shot.dialogue

    # Set column widths
    for row in table.rows:
        row.cells[0].width = Inches(0.5)   # Shot
        row.cells[1].width = Inches(1.0)   # Timestamp
        row.cells[2].width = Inches(3.0)   # Visual
        row.cells[3].width = Inches(3.0)   # Dialogue

    # Save document
    doc.save(output_path)
    return output_path


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_video(
    input_path: str,
    interval: Optional[float] = None,
    output_format: str = "docx",
    verbose: bool = False
) -> str:
    """
    Main processing pipeline for video analysis.

    Args:
        input_path: Path to video file or URL
        interval: If set, use fixed interval instead of scene detection
        output_format: Output format - "docx" or "md"
        verbose: Print progress information

    Returns:
        Path to generated output file
    """
    # Get script directory for output folder
    script_dir = Path(__file__).parent.absolute()
    output_base = script_dir / "output"

    # Create temp directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Handle URL input
        if is_url(input_path):
            if verbose:
                print("=" * 50)
                print("Step 1: Downloading video")
                print("=" * 50)
            video_path, title = download_video(input_path, temp_dir, verbose)
        else:
            video_path = input_path
            title = get_video_title(input_path)

        # Create output folder structure: output/[video name]/frames and output/[video name]/storyboard
        safe_title = sanitize_filename(title)
        video_output_dir = output_base / safe_title
        frames_dir = video_output_dir / "frames"
        storyboard_dir = video_output_dir / "storyboard"

        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(storyboard_dir, exist_ok=True)

        if verbose:
            print(f"Output folder: {video_output_dir}")

        # Get video duration
        duration = get_video_duration(video_path)
        if verbose:
            print(f"Video duration: {duration:.1f} seconds")

        # Step 2: Extract frames / detect scenes
        if verbose:
            print("=" * 50)
            print("Step 2: Frame extraction / scene detection")
            print("=" * 50)

        if interval:
            # Use fixed interval extraction
            frames = extract_frames_at_interval(
                video_path, str(frames_dir), interval, verbose
            )
            shots = create_shots_from_interval(frames, duration, interval)
        else:
            # Use scene detection
            scenes = detect_scenes(video_path, verbose)

            # Fallback to interval if no scenes detected
            if not scenes:
                if verbose:
                    print("No scenes detected, falling back to 3-second intervals")
                frames = extract_frames_at_interval(
                    video_path, str(frames_dir), 3.0, verbose
                )
                shots = create_shots_from_interval(frames, duration, 3.0)
            else:
                shots = create_shots_from_scenes(
                    scenes, video_path, str(frames_dir), verbose
                )

        if verbose:
            print(f"Created {len(shots)} shots")

        # Step 3: Transcribe audio
        if verbose:
            print("=" * 50)
            print("Step 3: Audio transcription")
            print("=" * 50)

        transcript = transcribe_audio(video_path, verbose=verbose)

        # Step 4: Analyze visuals with Claude
        if verbose:
            print("=" * 50)
            print("Step 4: Visual analysis with Claude")
            print("=" * 50)

        shots = analyze_all_frames(shots, verbose)

        # Step 5: Align transcript to shots
        if verbose:
            print("=" * 50)
            print("Step 5: Aligning transcript to shots")
            print("=" * 50)

        shots = align_transcript_to_shots(shots, transcript)

        # Step 6: Generate output
        if verbose:
            print("=" * 50)
            print("Step 6: Generating output")
            print("=" * 50)

        # Determine output path based on format
        if output_format.lower() == "docx":
            output_path = storyboard_dir / "breakdown.docx"
            generate_docx_output(shots, title, duration, str(output_path))
        else:
            output_path = storyboard_dir / "breakdown.md"
            markdown = generate_markdown_output(shots, title, duration)
            with open(output_path, "w") as f:
                f.write(markdown)

        if verbose:
            print(f"\nOutput written to: {output_path}")
            print(f"Frames saved to: {frames_dir}")

        return str(output_path)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze educational videos and output shot-by-shot breakdown tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_video.py "https://youtube.com/watch?v=xxx" --verbose
  python analyze_video.py input.mp4 --verbose
  python analyze_video.py input.mp4 --format md --interval 3

Output structure:
  output/
  └── [video name]/
      ├── frames/       (extracted frames)
      └── storyboard/   (breakdown.docx or breakdown.md)

Environment:
  ANTHROPIC_API_KEY - Required for Claude Vision analysis
        """
    )

    parser.add_argument(
        "input",
        help="Input video file path or YouTube/Vimeo URL"
    )

    parser.add_argument(
        "--format", "-f",
        choices=["docx", "md"],
        default="docx",
        help="Output format: docx (Word) or md (Markdown). Default: docx"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Extract frames every N seconds instead of using scene detection"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show progress during processing"
    )

    args = parser.parse_args()

    # Check dependencies
    missing = check_dependencies()
    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print("  pip install openai-whisper scenedetect[opencv] anthropic Pillow")
        print("  brew install ffmpeg yt-dlp  # or apt-get install on Linux")
        sys.exit(1)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is required")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    # Validate input
    if not is_url(args.input) and not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    try:
        output_path = process_video(
            args.input,
            interval=args.interval,
            output_format=args.format,
            verbose=args.verbose
        )
        print(f"Analysis complete! Output saved to: {output_path}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
