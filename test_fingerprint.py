#!/usr/bin/env python3
"""
Tests for the stylistic fingerprint classifier (v1 compatibility checks).

These tests validate that the v3 fingerprint still correctly handles
the core classification scenarios from v1.

Run with:  python3 test_fingerprint.py
"""

import sys
from analyze_video import Shot, TranscriptSegment, analyze_stylistic_fingerprint


def _make_shots(descriptions: list[str], duration_each: float = 2.0) -> list[Shot]:
    """Helper: build Shot objects from a list of visual description strings."""
    shots = []
    for i, desc in enumerate(descriptions):
        shots.append(Shot(
            number=i + 1,
            start_time=i * duration_each,
            end_time=(i + 1) * duration_each,
            visual_description=desc,
        ))
    return shots


def _make_transcript(text: str) -> list[TranscriptSegment]:
    """Helper: wrap a single string into one TranscriptSegment."""
    return [TranscriptSegment(start_time=0, end_time=10, text=text)]


# ── Test 1: 3D videos must be classified as Stylized 3D ──────────────────
def test_3d_majority():
    """When >= 40% of shots say '3D animated style', rendering_class must be Stylized 3D."""
    descs = [
        "3D animated style. Cozy apartment interior. Young man with brown hair sits at desk with laptop, looking bored. Warm lighting.",
        "3D animated style. Close-up of young man's face staring at glowing laptop screen. Blue light reflection on face.",
        "3D rendered. Man typing rapidly on keyboard. Holographic code streams floating in air around him.",
        "3D animated style. Split screen showing man on left, glowing AI interface on right. Neon pink and blue.",
        "3D animated style. City park at sunset. Man walks with confident woman with red hair. Golden hour lighting.",
        "3D rendered. Two characters sitting on park bench laughing. Trees and skyline behind.",
        "3D animated style. Café interior. Man nervously holds coffee cup across from woman. Warm ambient lighting.",
        "3D animated style. Woman throws head back laughing. Bokeh lights in background.",
        "2D title card with stylized text 'Love Algorithm' on dark background.",
        "3D animated. Man and woman standing apart on opposite sides of split screen. Contrasting warm/cool palettes.",
    ]
    shots = _make_shots(descs)
    transcript = _make_transcript(
        "Johnny's life was a perfectly predictable loop. All that was missing was love. "
        "So he created an AI to find his perfect match."
    )
    result = analyze_stylistic_fingerprint(shots, transcript, duration=30.0)
    assert result["rendering_class"] == "Stylized 3D", (
        f"Expected 'Stylized 3D', got '{result['rendering_class']}'"
    )
    print("  PASS  test_3d_majority")


# ── Test 2: 2D flat explainer videos ──────────────────────────────────────
def test_2d_flat():
    """Flat vector / cartoon descriptions should yield 'Flat 2D'."""
    descs = [
        "Flat vector illustration. Cartoon piggy bank on solid blue background.",
        "2D animated character. Girl with pigtails in yellow dress stands in classroom.",
        "Flat design. Coins stacking up. Solid green background.",
        "Cartoon style. Boy and girl looking at chart on whiteboard.",
        "Flat animation. Family sitting around dinner table in kitchen interior.",
    ]
    shots = _make_shots(descs)
    transcript = _make_transcript("Today we will learn about saving money.")
    result = analyze_stylistic_fingerprint(shots, transcript, duration=120.0)
    assert result["rendering_class"] == "Flat 2D", (
        f"Expected 'Flat 2D', got '{result['rendering_class']}'"
    )
    print("  PASS  test_2d_flat")


# ── Test 3: Line art detection ───────────────────────────────────────────
def test_line_art():
    """Line art / stick figure descriptions should trigger Minimalist Line Art 2D."""
    descs = [
        "2D black and white cartoon style. Simple stick figure businessman carrying DEBT crate.",
        "2D black line art cartoon style. Stick figure on paper texture background.",
        "Simple black line drawing on crumpled paper texture. Stick figure holding brick.",
        "2D minimalist line art style. Simple stick figure businessman smiling.",
        "Black and white cartoon. Stick figure person standing near house outline.",
    ]
    shots = _make_shots(descs)
    transcript = _make_transcript("Let me explain how interest rates work.")
    result = analyze_stylistic_fingerprint(shots, transcript, duration=120.0)
    assert result["rendering_class"] == "Minimalist Line Art 2D", (
        f"Expected 'Minimalist Line Art 2D', got '{result['rendering_class']}'"
    )
    print("  PASS  test_line_art")


# ── Test 4: Textured 2D (watercolor/engraving) ───────────────────────────
def test_textured_2d():
    """Victorian/watercolor/engraving should yield Textured 2D."""
    descs = [
        "Victorian illustration style. Cross-hatching detail. Gentleman at wooden counter.",
        "Watercolor illustration style. Scattered coins on weathered surface. Fishing hook.",
        "Vintage illustration style with crosshatching. Workshop interior. Oil lamp lighting.",
        "2D watercolor illustration style. Courtroom with Victorian paneling.",
        "Engraving style. Detailed crosshatch shading. Old bank interior.",
    ]
    shots = _make_shots(descs)
    transcript = _make_transcript("The overdraft trap explained.")
    result = analyze_stylistic_fingerprint(shots, transcript, duration=120.0)
    assert result["rendering_class"] == "Textured 2D (engraving/watercolor)", (
        f"Expected 'Textured 2D (engraving/watercolor)', got '{result['rendering_class']}'"
    )
    print("  PASS  test_textured_2d")


# ── Test 5: All 8 fields must be populated ──────────────────────────────
def test_all_pillars():
    """Every field must return a non-empty string."""
    shots = _make_shots(["Simple scene." for _ in range(3)])
    transcript = _make_transcript("Hello world.")
    result = analyze_stylistic_fingerprint(shots, transcript, duration=10.0)
    required = [
        "rendering_class", "world_type", "character_strategy",
        "narrative_structure", "visual_abstraction_index",
        "visual_density", "camera_editing", "tonal_positioning",
    ]
    for key in required:
        assert key in result and result[key], f"Missing or empty: {key}"
    print("  PASS  test_all_pillars")


# ── Run all tests ────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_3d_majority,
        test_2d_flat,
        test_line_art,
        test_textured_2d,
        test_all_pillars,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1

    print(f"\n{'All' if not failed else failed} {'passed' if not failed else 'failed'} / {len(tests)} tests")
    sys.exit(1 if failed else 0)
