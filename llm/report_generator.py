"""
TrueSight — PDF forensic report + Streamlit bytes helper (v3.0).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    import ollama
except ImportError:
    ollama = None


def _build_qwen_prompt(analysis: dict[str, Any]) -> str:
    subs = analysis.get("sub_scores") or {}
    return f"""You are TrueSight forensic AI. Write a professional forensic report.

FILE: {analysis.get("filename", "unknown")}
OVERALL RISK: {analysis.get("score", 0)}% — {analysis.get("verdict", "N/A")}
LIVENESS: {"Confirmed" if analysis.get("liveness_detected") else "Not detected"}
SUB-SCORES: Image={subs.get("image", 0):.0f}%, Audio={subs.get("audio", 0):.0f}%, Video={subs.get("video", 0):.0f}%
TOP INDICATORS:
{chr(10).join("- " + r for r in (analysis.get("reasons") or [])[:12])}

Write EXACTLY 3 plain-text paragraphs. No headers. No bullets. No markdown.
Para 1 (2-3 sentences): Overall risk assessment.
Para 2 (3-4 sentences): Most significant forensic indicators and technical meaning.
Para 3 (2-3 sentences): Recommended action.
Plain text only."""


def _ollama_narrative(analysis: dict[str, Any], model: str) -> str:
    if not ollama:
        return (
            "Automated heuristic fusion produced the stated risk score. "
            "Ollama is not installed or unavailable — narrative omitted."
        )
    try:
        r = ollama.generate(
            model=model,
            prompt=_build_qwen_prompt(analysis),
            options={"temperature": 0.2, "num_predict": 400},
        )
        return (r.get("response") or "").strip() or "Narrative empty."
    except Exception:
        return "Ollama narrative unavailable (start `ollama serve` and pull the model)."


def generate_report(
    analysis: dict[str, Any],
    output_path: str,
    *,
    use_ollama: bool = True,
    ollama_model: str = "qwen2:0.5b",
) -> str:
    """Build a compact multi-section PDF from a `compute_final_score`-style dict."""
    styles = getSampleStyleSheet()
    story: list = []

    navy = colors.HexColor("#0B1F3B")
    accent = colors.HexColor("#1E5AA8")

    title = ParagraphStyle(
        "TS_Title",
        parent=styles["Heading1"],
        textColor=navy,
        fontSize=18,
        spaceAfter=12,
    )
    h2 = ParagraphStyle(
        "TS_H2",
        parent=styles["Heading2"],
        textColor=navy,
        fontSize=12,
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle("TS_Body", parent=styles["Normal"], fontSize=9, leading=12)

    story.append(Paragraph("TRUESIGHT — Forensic Analysis Report", title))
    story.append(
        Paragraph(
            f"<font color='grey'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — "
            f"ID {uuid.uuid4().hex[:8].upper()}</font>",
            body,
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    score = analysis.get("score", 0)
    verdict = analysis.get("verdict", "N/A")
    tbl = Table(
        [
            ["Overall risk", f"{score}%"],
            ["Verdict", verdict],
            [
                "Liveness",
                "Detected" if analysis.get("liveness_detected") else "Not detected",
            ],
        ],
        colWidths=[2 * inch, 4 * inch],
    )
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (1, 0), (1, 0), accent),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Forensic indicators (deduped)", h2))
    by_tag: dict[str, list[str]] = {}
    for r in analysis.get("reasons") or []:
        tag = (
            r.split("]")[0].strip("[")
            if isinstance(r, str) and r.startswith("[")
            else "General"
        )
        by_tag.setdefault(tag, []).append(r)
    for tag, msgs in by_tag.items():
        line = msgs[0] if len(msgs) == 1 else f"{msgs[0]} (+{len(msgs) - 1} similar)"
        story.append(Paragraph(line.replace("&", "&amp;"), body))

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("AI reasoning (Qwen2 via Ollama)", h2))
    narrative = (
        _ollama_narrative(analysis, ollama_model)
        if use_ollama
        else "Ollama disabled for this report."
    )
    for para in narrative.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip().replace("&", "&amp;"), body))

    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            "<i>Method: heuristic detectors, weighted fusion, optional local LLM. "
            "Fully offline when models are local. Not a legal finding.</i>",
            body,
        )
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.6 * inch,
    )

    def _header_footer(canvas, doc_):
        canvas.saveState()
        canvas.setFillColor(navy)
        canvas.rect(0, letter[1] - 0.45 * inch, letter[0], 0.45 * inch, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(0.6 * inch, letter[1] - 0.28 * inch, "TRUESIGHT")
        canvas.setFont("Helvetica", 9)
        canvas.drawCentredString(letter[0] / 2, letter[1] - 0.28 * inch, "Forensic Analysis Report")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(
            letter[0] - 0.6 * inch,
            letter[1] - 0.28 * inch,
            datetime.now().strftime("%H:%M:%S"),
        )
        canvas.setStrokeColor(accent)
        canvas.setLineWidth(1.5)
        canvas.line(0.5 * inch, letter[1] - 0.46 * inch, letter[0] - 0.5 * inch, letter[1] - 0.46 * inch)
        canvas.setFillColor(colors.grey)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(
            0.6 * inch,
            0.35 * inch,
            "Confidential — TrueSight forensic aid only.",
        )
        canvas.drawRightString(
            letter[0] - 0.6 * inch,
            0.35 * inch,
            f"Page {doc_.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output_path


def get_report_bytes(
    analysis: dict[str, Any],
    *,
    use_ollama: bool = True,
    ollama_model: str = "qwen2:0.5b",
) -> bytes:
    fd, tmp = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        generate_report(
            analysis,
            tmp,
            use_ollama=use_ollama,
            ollama_model=ollama_model,
        )
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
