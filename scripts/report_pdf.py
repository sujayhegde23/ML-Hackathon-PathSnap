"""Render the report markdown subset to a plain black-and-white PDF with reportlab.

The HTML route through LibreOffice kept clipping table headers and squeezing columns, so the
PDF is built directly instead. Supported markdown: headings, paragraphs, pipe tables and
images. Text is black on white, tables have thin black rules, and no other styling is used.
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BODY = ParagraphStyle(
    "body", fontName="Times-Roman", fontSize=10.5, leading=14, alignment=TA_LEFT,
    spaceBefore=2, spaceAfter=6,
)
H1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=16, leading=20, spaceAfter=10)
H2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=12.5, leading=16,
                    spaceBefore=14, spaceAfter=6)
H3 = ParagraphStyle("h3", fontName="Times-Bold", fontSize=11, leading=14,
                    spaceBefore=10, spaceAfter=4)
H4 = ParagraphStyle("h4", fontName="Times-Bold", fontSize=10.5, leading=13,
                    spaceBefore=8, spaceAfter=3)
CELL = ParagraphStyle("cell", fontName="Times-Roman", fontSize=7.8, leading=9.6)
CELL_H = ParagraphStyle("cellh", fontName="Times-Bold", fontSize=7.8, leading=9.6)


def inline(text: str) -> str:
    """Markdown inline to reportlab inline, with XML escaped first."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="9">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def build_table(rows: list[list[str]], width: float) -> Table:
    """Column widths from content length, every cell a wrapping Paragraph."""
    ncols = len(rows[0])
    lens = [max(len(r[c]) if c < len(r) else 0 for r in rows) for c in range(ncols)]
    raw = [max(1.4 * cm, min(6.5 * cm, width * l / max(sum(lens), 1) * 1.15)) for l in lens]
    scale = width / sum(raw)
    widths = [w * scale for w in raw]

    body = [[Paragraph(inline(c), CELL_H if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    table = Table(body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return table


def render(md_path: Path, pdf_path: Path, base_dir: Path) -> dict:
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title="PathSnap technical report", author="PathSnap team",
    )
    width = doc.width
    lines = md_path.read_text().split("\n")
    story, i, stats = [], 0, {"tables": 0, "images": 0, "paragraphs": 0}

    while i < len(lines):
        ln = lines[i]

        if ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block]
            rows = [r for r in rows if not set("".join(r)) <= set("- ")]
            if rows:
                story.append(build_table(rows, width))
                story.append(Spacer(1, 8))
                stats["tables"] += 1
            continue

        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", ln)
        if m:
            img_path = base_dir / m.group(2)
            if img_path.is_file():
                from PIL import Image as PILImage

                # Downscale before embedding, otherwise four 1080p screenshots make an 11 MB PDF.
                cache = base_dir.parent / ".tmp" / "report_img"
                cache.mkdir(parents=True, exist_ok=True)
                small = cache / (img_path.stem + ".jpg")
                if not small.is_file() or small.stat().st_mtime < img_path.stat().st_mtime:
                    with PILImage.open(img_path) as im:
                        im = im.convert("RGB")
                        if im.width > 1400:
                            im = im.resize((1400, round(im.height * 1400 / im.width)), PILImage.LANCZOS)
                        im.save(small, quality=82, optimize=True)
                img_path = small

                with PILImage.open(img_path) as im:
                    w_px, h_px = im.size
                w = min(width, 15 * cm)
                h = w * h_px / w_px
                if h > 10 * cm:
                    h = 10 * cm
                    w = h * w_px / h_px
                story.append(KeepTogether([Image(str(img_path), width=w, height=h)]))
                story.append(Spacer(1, 4))
                stats["images"] += 1
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            lvl, text = len(m.group(1)), m.group(2)
            style = {1: H1, 2: H2, 3: H3, 4: H4}[lvl]
            story.append(Paragraph(inline(text), style))
            i += 1
            continue

        if ln.strip() == "---":
            i += 1
            continue

        if ln.strip():
            para = [ln.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("|", "#", "!")):
                para.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline(" ".join(para)), BODY))
            stats["paragraphs"] += 1
            continue

        i += 1

    doc.build(story)
    return stats
