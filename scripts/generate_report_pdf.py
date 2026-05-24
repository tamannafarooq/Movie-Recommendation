from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import textwrap


def md_to_pdf(md_path: Path, pdf_path: Path):
    text = md_path.read_text(encoding="utf-8")
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    left = 20 * mm
    right = 20 * mm
    top = 20 * mm
    bottom = 20 * mm
    usable_width = width - left - right
    max_chars = int(usable_width // 6)
    lines = []
    for paragraph in text.split("\n\n"):
        for line in paragraph.splitlines():
            wrapped = textwrap.wrap(line, width=max_chars) or [""]
            lines.extend(wrapped)
        lines.append("")

    y = height - top
    text_obj = c.beginText(left, y)
    text_obj.setFont("Helvetica", 10)
    line_height = 12

    for line in lines:
        if text_obj.getY() < bottom + line_height:
            c.drawText(text_obj)
            c.showPage()
            text_obj = c.beginText(left, height - top)
            text_obj.setFont("Helvetica", 10)
        text_obj.textLine(line)

    c.drawText(text_obj)
    c.save()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    md = root / "Tamana_Farooq_Project_Report.md"
    pdf = root / "Tamana_Farooq_Project_Report.pdf"
    md_to_pdf(md, pdf)
    print(f"Updated PDF generated at {pdf}")
