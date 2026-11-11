"""
PDF storyboard: a cover page (overview, fingerprint, pacing & hook) followed
by storyboard sheets, a grid of panels with each shot's frame, timing,
description, on-screen text and dialogue. Real text, so it stays searchable.
"""

from __future__ import annotations

import os
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from xml.sax.saxutils import escape

INK = colors.HexColor("#1d1d1f")
MUTED = colors.HexColor("#6e6e73")
RULE = colors.HexColor("#d2d2d7")
ACCENT = colors.HexColor("#3b5bdb")
PANEL_BG = colors.HexColor("#f5f5f7")

PAGE_W, PAGE_H = landscape(letter)
MARGIN = 0.45 * inch


def _safe(text: str) -> str:
    """Escape for Paragraph markup and drop characters the base fonts can't draw."""
    text = (text or "").encode("cp1252", "ignore").decode("cp1252")
    return escape(text)


def _style(size: float, color=INK, bold=False, leading=None, italic=False) -> ParagraphStyle:
    font = "Helvetica-Bold" if bold else ("Helvetica-Oblique" if italic else "Helvetica")
    return ParagraphStyle("s", fontName=font, fontSize=size, leading=leading or size * 1.28,
                          textColor=color, alignment=TA_LEFT)


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _footer(c: canvas.Canvas, title: str, page: int) -> None:
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, 0.28 * inch, _clip(title, 90))
    c.drawRightString(PAGE_W - MARGIN, 0.28 * inch, f"Video Analyzer storyboard  ·  {page}")


def _draw_para(c, text_markup: str, style: ParagraphStyle, x: float, y_top: float, w: float, max_h: float) -> float:
    """Draw a paragraph with its top at y_top; returns its height (clipped to max_h)."""
    p = Paragraph(text_markup, style)
    _, h = p.wrap(w, max_h)
    if h > max_h:  # too long: shorten until it fits
        plain = text_markup
        while h > max_h and len(plain) > 20:
            plain = plain[: int(len(plain) * 0.85)].rsplit(" ", 1)[0] + "…"
            p = Paragraph(plain, style)
            _, h = p.wrap(w, max_h)
    p.drawOn(c, x, y_top - h)
    return h


def _cover(c, shots, metadata, visual_style, pacing, contact_sheet: Optional[str]) -> None:
    top = PAGE_H - MARGIN
    c.setFillColor(ACCENT)
    c.rect(MARGIN, top - 4, 0.6 * inch, 4, stroke=0, fill=1)
    h = _draw_para(c, _safe(metadata.title), _style(20, bold=True), MARGIN, top - 14, PAGE_W * 0.55, 60)
    y = top - 14 - h - 6
    facts = f"{metadata.duration_str}  ·  {metadata.aspect_ratio}  ·  {len(shots)} shots"
    if pacing:
        facts += f"  ·  {pacing['cuts_per_minute']} cuts/min"
    _draw_para(c, _safe(facts), _style(10, MUTED), MARGIN, y, PAGE_W * 0.55, 20)
    y -= 26

    col_w = PAGE_W * 0.55 - MARGIN
    rows = []
    if metadata.story_structure:
        rows.append(("Story structure", _clip(metadata.story_structure, 260)))
    if metadata.reading_level:
        rows.append(("Reading level", _clip(metadata.reading_level, 160)))
    if visual_style:
        for label, key in (("Rendering", "rendering_class"), ("World", "world_type"),
                           ("Characters", "character_strategy"), ("Narrative", "narrative_structure"),
                           ("Abstraction", "visual_abstraction_index"), ("Density", "visual_density"),
                           ("Camera / editing", "camera_editing"), ("Tone", "tonal_positioning")):
            if visual_style.get(key):
                rows.append((label, visual_style[key]))
    if pacing:
        rows.append(("Pace", f"{pacing['pace']}; average shot {pacing['average_shot_seconds']}s"))
        hook = pacing.get("hook") or {}
        hook_bits = [f"{hook.get('shots', 0)} shot(s), {hook.get('words_spoken', 0)} word(s) in the first {hook.get('seconds', 3)}s"]
        if hook.get("first_spoken_line"):
            hook_bits.append(f"“{_clip(hook['first_spoken_line'], 90)}”")
        if hook.get("on_screen_text"):
            hook_bits.append(f"on screen: {_clip(hook['on_screen_text'], 80)}")
        rows.append(("Hook", "; ".join(hook_bits)))

    data = [[Paragraph(_safe(k), _style(8.5, MUTED, bold=True)), Paragraph(_safe(v), _style(8.5))] for k, v in rows]
    if data:
        t = Table(data, colWidths=[1.25 * inch, col_w - 1.25 * inch])
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))
        _, th = t.wrap(col_w, y - MARGIN)
        t.drawOn(c, MARGIN, max(MARGIN, y - th))

    if contact_sheet and os.path.exists(contact_sheet):
        img = ImageReader(contact_sheet)
        iw, ih = img.getSize()
        box_x, box_w = PAGE_W * 0.58, PAGE_W * 0.42 - MARGIN
        box_h = PAGE_H - 2 * MARGIN - 0.2 * inch
        scale = min(box_w / iw, box_h / ih)
        w, hgt = iw * scale, ih * scale
        c.drawImage(img, box_x + (box_w - w) / 2, PAGE_H - MARGIN - hgt, w, hgt)
    _footer(c, metadata.title, 1)


def generate_pdf_storyboard(shots, metadata, output_path: str, visual_style: Optional[dict] = None,
                            pacing: Optional[dict] = None, contact_sheet: Optional[str] = None) -> str:
    """Write the PDF storyboard and return its path."""
    vertical = metadata.orientation == "vertical"
    cols, rows = (4, 1) if vertical else (3, 2)

    c = canvas.Canvas(output_path, pagesize=landscape(letter))
    c.setTitle(f"Storyboard: {metadata.title}")
    c.setAuthor("Video Analyzer")
    _cover(c, shots, metadata, visual_style, pacing, contact_sheet)
    c.showPage()

    gap = 0.22 * inch
    grid_w = PAGE_W - 2 * MARGIN
    grid_h = PAGE_H - 2 * MARGIN - 0.25 * inch
    panel_w = (grid_w - gap * (cols - 1)) / cols
    panel_h = (grid_h - gap * (rows - 1)) / rows
    pad = 7
    frame_h = min(panel_h * (0.62 if vertical else 0.55), (panel_w - 2 * pad) * (16 / 9 if vertical else 9 / 16))

    per_page = cols * rows
    page = 2
    for start in range(0, len(shots), per_page):
        for i, shot in enumerate(shots[start:start + per_page]):
            col, row = i % cols, i // cols
            x = MARGIN + col * (panel_w + gap)
            y_top = PAGE_H - MARGIN - row * (panel_h + gap)
            c.setFillColor(PANEL_BG)
            c.roundRect(x, y_top - panel_h, panel_w, panel_h, 6, stroke=0, fill=1)

            # Frame, fitted into its box
            box_w = panel_w - 2 * pad
            if shot.frame_path and os.path.exists(shot.frame_path):
                img = ImageReader(shot.frame_path)
                iw, ih = img.getSize()
                s = min(box_w / iw, frame_h / ih)
                w, h = iw * s, ih * s
                c.drawImage(img, x + pad + (box_w - w) / 2, y_top - pad - h, w, h)
            else:
                c.setStrokeColor(RULE)
                c.rect(x + pad, y_top - pad - frame_h, box_w, frame_h, stroke=1, fill=0)

            ty = y_top - pad - frame_h - 6
            limit = y_top - panel_h + pad
            hdr = f"<b>Shot {shot.number}</b>  <font color='#6e6e73'>{_safe(shot.timestamp_str)}</font>"
            ty -= _draw_para(c, hdr, _style(9), x + pad, ty, box_w, 14) + 3
            if shot.visual_description:
                ty -= _draw_para(c, _safe(shot.visual_description), _style(7.2, leading=8.9),
                                 x + pad, ty, box_w, max(10, (ty - limit) * 0.62)) + 3
            if shot.on_screen_text and ty - limit > 12:
                ty -= _draw_para(c, "<b>On screen:</b> " + _safe(shot.on_screen_text), _style(7, ACCENT),
                                 x + pad, ty, box_w, max(9, (ty - limit) * 0.45)) + 3
            if shot.dialogue and not shot.dialogue.startswith("(no") and ty - limit > 10:
                _draw_para(c, "“" + _safe(shot.dialogue) + "”", _style(7, MUTED, italic=True),
                           x + pad, ty, box_w, ty - limit)
        _footer(c, metadata.title, page)
        c.showPage()
        page += 1

    c.save()
    return output_path
