import os
import re
from pathlib import Path
import markdown
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def clean_markdown_formatting(text: str) -> str:
    """Safely convert basic markdown bold/italic/code to HTML tags for ReportLab."""
    # Convert & first
    text = text.replace('&', '&amp;')
    # Replace **bold** with <b>bold</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Replace *italic* with <i>italic</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Replace `code` with <font name="Courier">code</font>
    text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)
    return text

def create_technical_doc_pdf():
    base_dir = Path(__file__).resolve().parent.parent
    md_path = base_dir / "CineSum_AI_Technical_Documentation.md"
    pdf_path = base_dir / "CineSum_AI_Technical_Documentation.pdf"

    # Register Turkish-compatible Windows fonts (Arial)
    font_name = "Helvetica"
    font_bold = "Helvetica-Bold"
    
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font_bold_path = Path("C:/Windows/Fonts/arialbd.ttf")

    if font_path.exists() and font_bold_path.exists():
        try:
            pdfmetrics.registerFont(TTFont('ArialTR', str(font_path)))
            pdfmetrics.registerFont(TTFont('ArialTR-Bold', str(font_bold_path)))
            font_name = 'ArialTR'
            font_bold = 'ArialTR-Bold'
            print("ArialTR font successfully registered.")
        except Exception as e:
            print("Font fallback:", e)

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName=font_bold,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1a2a3a'),
        spaceAfter=15,
        alignment=1 # Center
    )

    h1_style = ParagraphStyle(
        'Heading1Custom',
        parent=styles['Heading1'],
        fontName=font_bold,
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#0f4c81'),
        spaceBefore=14,
        spaceAfter=8
    )

    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName=font_bold,
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2c3e50'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=body_style,
        leftIndent=12,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'CodeCustom',
        parent=styles['Code'],
        fontName=font_name,
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#2b2b2b'),
        backColor=colors.HexColor('#f4f6f8'),
        borderColor=colors.HexColor('#e1e8ed'),
        borderWidth=1,
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=6
    )

    story = []
    lines = md_text.split('\n')
    
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    for line in lines:
        stripped = line.strip()

        # Code block handling
        if stripped.startswith('```'):
            if in_code_block:
                in_code_block = False
                code_content = "<br/>".join([
                    c.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace(' ', '&nbsp;')
                    for c in code_lines
                ])
                story.append(Paragraph(code_content, code_style))
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Markdown Table handling
        if '|' in line and stripped.startswith('|') and stripped.endswith('|'):
            if '---' in line:
                continue # Header separator line
            cols = [c.strip() for c in stripped.split('|')[1:-1]]
            if cols:
                table_rows.append(cols)
            in_table = True
            continue
        else:
            if in_table and table_rows:
                # Render accumulated table
                formatted_table_data = []
                for row_idx, r in enumerate(table_rows):
                    formatted_row = []
                    for cell in r:
                        clean_cell = clean_markdown_formatting(cell)
                        style_to_use = ParagraphStyle(
                            'TableCell', parent=body_style, fontSize=8, leading=10.5,
                            fontName=font_bold if row_idx == 0 else font_name
                        )
                        formatted_row.append(Paragraph(clean_cell, style_to_use))
                    formatted_row_data = formatted_row
                    formatted_table_data.append(formatted_row_data)

                # Responsive column widths
                num_cols = len(table_rows[0])
                available_width = 532 # Letter page width minus margins
                col_width = available_width / num_cols if num_cols > 0 else 100

                t = Table(formatted_table_data, colWidths=[col_width] * num_cols)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf2f7')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1a202c')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('LEFTPADDING', (0, 0), (-1, -1), 4),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(t)
                story.append(Spacer(1, 8))
                table_rows = []
                in_table = False

        if not stripped:
            story.append(Spacer(1, 4))
            continue

        # Horizontal rule
        if stripped == '---':
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceBefore=6, spaceAfter=6))
            continue

        # Headings
        if stripped.startswith('# '):
            text = clean_markdown_formatting(stripped[2:].strip())
            story.append(Paragraph(text, title_style))
            story.append(Spacer(1, 6))
            continue
        elif stripped.startswith('## '):
            text = clean_markdown_formatting(stripped[3:].strip())
            story.append(Paragraph(text, h1_style))
            continue
        elif stripped.startswith('### '):
            text = clean_markdown_formatting(stripped[4:].strip())
            story.append(Paragraph(text, h2_style))
            continue
        elif stripped.startswith('#### '):
            text = clean_markdown_formatting(stripped[5:].strip())
            story.append(Paragraph(text, body_style))
            continue

        # Bullet points
        if stripped.startswith('- ') or stripped.startswith('* '):
            item_text = clean_markdown_formatting(stripped[2:].strip())
            story.append(Paragraph(f"• {item_text}", bullet_style))
            continue

        # Paragraph text
        text_html = clean_markdown_formatting(stripped)
        story.append(Paragraph(text_html, body_style))

    doc.build(story)
    print(f"PDF successfully generated: {pdf_path}")

if __name__ == "__main__":
    create_technical_doc_pdf()
