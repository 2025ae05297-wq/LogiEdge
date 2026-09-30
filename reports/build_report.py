"""Render final_report.md and its figures into a readable PDF."""

from pathlib import Path
import re

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]


def clean(text: str) -> str:
    return text.replace("₹", "Rs ").replace("—", "-").replace("–", "-")


def build() -> None:
    source = ROOT / "reports" / "final_report.md"
    output = ROOT / "reports" / "final_report.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19, leading=24, alignment=TA_CENTER, textColor="#075985", spaceAfter=14))
    styles.add(ParagraphStyle(name="H1Custom", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor="#075985", spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor="#0f766e", spaceBefore=9, spaceAfter=5))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.4, leading=13, spaceAfter=6, alignment=4))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=8, leading=10, alignment=TA_CENTER, spaceAfter=8))
    story = []
    lines = source.read_text().splitlines()
    paragraph = []
    def flush() -> None:
        if paragraph:
            text = clean(" ".join(paragraph))
            text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
            text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
            text = text.replace("![", "[")
            story.append(Paragraph(text, styles["BodyCustom"]))
            paragraph.clear()
    for line in lines:
        if not line.strip():
            flush(); continue
        if line.startswith("# "):
            flush(); story.append(Paragraph(clean(line[2:]), styles["ReportTitle"]))
        elif line.startswith("## "):
            flush(); story.append(Paragraph(clean(line[3:]), styles["H1Custom"]))
        elif line.startswith("### "):
            flush(); story.append(Paragraph(clean(line[4:]), styles["H2Custom"]))
        elif line.startswith("!["):
            flush()
            match = re.search(r"\(([^)]+)\)", line)
            if match:
                image_path = (source.parent / match.group(1)).resolve()
                if image_path.exists() and image_path.stat().st_size:
                    story.append(Image(str(image_path), width=6.4 * inch, height=3.3 * inch, kind="proportional"))
        elif line.startswith("**Figure"):
            flush(); story.append(Paragraph(clean(line.replace("**", "")), styles["Caption"]))
        elif line.startswith("-"):
            flush(); story.append(Paragraph("• " + clean(line[1:].strip()), styles["BodyCustom"]))
        elif line.startswith("```"):
            flush()
        else:
            paragraph.append(line)
    flush()
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=0.65 * inch, leftMargin=0.65 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch, title="LogiEdge Final Report")
    doc.build(story)
    print(f"wrote {output}")


if __name__ == "__main__":
    build()