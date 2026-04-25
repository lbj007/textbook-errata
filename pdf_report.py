"""Generate errata PDF report in the human-review table format."""

import io
import os
import platform
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def _register_font():
    """Register a Chinese font, trying multiple paths for cross-platform support."""
    font_candidates = []

    if platform.system() == 'Darwin':
        font_candidates = [
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Supplemental/Songti.ttc",
        ]
    else:
        # Linux (Docker)
        font_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto/NotoSansSC-Regular.otf",
        ]

    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont('CF', fp, subfontIndex=0))
                return True
            except Exception:
                try:
                    pdfmetrics.registerFont(TTFont('CF', fp))
                    return True
                except Exception:
                    continue
    return False


_font_ok = _register_font()
FONT = 'CF' if _font_ok else 'Helvetica'

# Colors
C_HEADER_BG = HexColor('#2c3e50')
C_ROW_ALT = HexColor('#f8f9fa')
C_BORDER = HexColor('#bdc3c7')
C_RED = HexColor('#c0392b')
C_GREEN = HexColor('#27ae60')
C_ORANGE = HexColor('#e67e22')
C_BLUE = HexColor('#2980b9')
C_DARK = HexColor('#2c3e50')
C_LIGHT_BG = HexColor('#eaf2f8')
C_WHITE = HexColor('#ffffff')
C_LIGHT_RED = HexColor('#fdedec')


def _S(name, size=8.5, color=C_DARK, align=0, lead=None):
    return ParagraphStyle(name, fontName=FONT, fontSize=size,
                          leading=lead or size * 1.6, textColor=color, alignment=align)


s_title = _S('title', 18, C_DARK, 1)
s_sub = _S('sub', 10, HexColor('#7f8c8d'), 1)
s_info = _S('info', 8, HexColor('#95a5a6'), 1)
s_section = _S('section', 12, C_BLUE, 0)
s_body = _S('body', 8.5, C_DARK)
s_body_sm = _S('body_sm', 7.5, C_DARK)
s_th = _S('th', 8, C_WHITE, 1)
s_red = _S('red', 8.5, C_RED)
s_orange = _S('orange', 8.5, C_ORANGE)
s_note = _S('note', 7.5, HexColor('#7f8c8d'))
s_footer = _S('footer', 7.5, HexColor('#95a5a6'))


def _make_screenshot_flowable(img_bytes, max_w=45 * mm, max_h=40 * mm):
    """Create an Image flowable from JPEG bytes, scaled to fit."""
    buf = io.BytesIO(img_bytes)
    img = Image(buf)
    # Scale to fit within max dimensions while preserving aspect ratio
    iw, ih = img.drawWidth, img.drawHeight
    if iw <= 0 or ih <= 0:
        return Paragraph('(截图)', _S('ss_err', 7, HexColor('#999999'), 1))
    scale = min(max_w / iw, max_h / ih, 1.0)
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    return img


def generate_errata_pdf(output_path, filename, errata_items, total_pages,
                        review_date=None, screenshots=None):
    """Generate errata PDF report.

    Args:
        output_path: Where to save the PDF
        filename: Original PDF filename (used in title)
        errata_items: List of dicts from proofreader
        total_pages: Total pages in original PDF
        review_date: Date string, defaults to today
        screenshots: dict mapping item index -> JPEG bytes (from generate_screenshots)

    Returns:
        output_path
    """
    if screenshots is None:
        screenshots = {}
    from datetime import date
    if review_date is None:
        review_date = date.today().strftime('%Y年%m月%d日')

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm)
    els = []

    # Count severities
    high = sum(1 for e in errata_items if e.get('severity') == '高')
    mid = sum(1 for e in errata_items if e.get('severity') == '中')
    low = sum(1 for e in errata_items if e.get('severity', '低') == '低')
    total = len(errata_items)

    # Title
    els.append(Paragraph('勘 误 表', s_title))
    els.append(Spacer(1, 2 * mm))

    # Clean filename for display
    display_name = filename.replace('-定稿', '').replace('_', ' ')
    els.append(Paragraph(display_name, s_sub))
    els.append(Spacer(1, 2 * mm))
    els.append(Paragraph(f'审校日期：{review_date} &nbsp;|&nbsp; AI辅助审校', s_info))
    els.append(Spacer(1, 3 * mm))
    els.append(HRFlowable(width="100%", thickness=1, color=C_BORDER, spaceAfter=3 * mm))

    # Summary table
    sum_data = [
        [Paragraph('<b>检查范围</b>', s_body), Paragraph(f'全册 {total_pages} 页', s_body),
         Paragraph('<b>发现问题</b>', s_body), Paragraph(f'共 <b>{total}</b> 处', s_body)],
        [Paragraph('<b>严重程度</b>', s_body),
         Paragraph(
             f'<font color="#c0392b">高 {high} 处</font> | '
             f'<font color="#e67e22">中 {mid} 处</font> | '
             f'<font color="#2980b9">低 {low} 处</font>', s_body),
         Paragraph('<b>审校方式</b>', s_body),
         Paragraph('AI视觉审校 + 文本分析', s_body)],
    ]
    st = Table(sum_data, colWidths=[22 * mm, 65 * mm, 22 * mm, 77 * mm])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), C_LIGHT_BG),
        ('BACKGROUND', (2, 0), (2, -1), C_LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2 * mm),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2 * mm),
        ('LEFTPADDING', (0, 0), (-1, -1), 2 * mm),
    ]))
    els.append(st)
    els.append(Spacer(1, 4 * mm))

    # Detail table
    els.append(Paragraph('<b>勘误详情</b>', s_section))
    els.append(Spacer(1, 2 * mm))

    has_screenshots = bool(screenshots)

    if has_screenshots:
        col_w = [7 * mm, 10 * mm, 48 * mm, 48 * mm, 48 * mm, 10 * mm]
        header = [
            Paragraph('<b>#</b>', s_th),
            Paragraph('<b>页码</b>', s_th),
            Paragraph('<b>截图</b>', s_th),
            Paragraph('<b>位置 / 内容</b>', s_th),
            Paragraph('<b>修改建议</b>', s_th),
            Paragraph('<b>严重</b>', s_th),
        ]
    else:
        col_w = [7 * mm, 12 * mm, 55 * mm, 65 * mm, 10 * mm, 37 * mm]
        header = [
            Paragraph('<b>#</b>', s_th),
            Paragraph('<b>页码</b>', s_th),
            Paragraph('<b>位置 / 内容</b>', s_th),
            Paragraph('<b>修改建议</b>', s_th),
            Paragraph('<b>严重</b>', s_th),
            Paragraph('<b>备注</b>', s_th),
        ]

    batch_size = 4 if has_screenshots else 6
    all_rows = []

    for i, err in enumerate(errata_items):
        sev = err.get('severity', '低')
        if sev == '高':
            sev_style = s_red
        elif sev == '中':
            sev_style = s_orange
        else:
            sev_style = _S(f'sev_lo_{i}', 8.5, C_BLUE, 1)

        # Content column
        content_parts = []
        loc = err.get('location', '')
        desc = err.get('content_desc', '')
        if loc:
            content_parts.append(f'<b>{_esc(loc)}</b>')
        if desc:
            content_parts.append(_esc(desc))
        content_text = '<br/>'.join(content_parts)

        # Suggestions column
        suggestions = err.get('suggestions', [])
        if isinstance(suggestions, str):
            suggestions = [suggestions]
        sugg_text = '<br/>'.join(_esc(s) for s in suggestions)

        notes_text = _esc(err.get('notes', ''))

        if has_screenshots:
            # Screenshot column
            if i in screenshots:
                ss_flowable = _make_screenshot_flowable(screenshots[i])
            else:
                ss_flowable = Paragraph('<i>(无截图)</i>', s_note)

            row = [
                Paragraph(str(i + 1), _S(f'n{i}', 8, C_DARK, 1)),
                Paragraph(_esc(err.get('page', '?')), _S(f'p{i}', 8, C_DARK, 1)),
                ss_flowable,
                Paragraph(content_text, s_body_sm),
                Paragraph(sugg_text, s_body_sm),
                Paragraph(f'<b>{_esc(sev)}</b>', sev_style),
            ]
        else:
            row = [
                Paragraph(str(i + 1), _S(f'n{i}', 8, C_DARK, 1)),
                Paragraph(_esc(err.get('page', '?')), _S(f'p{i}', 8, C_DARK, 1)),
                Paragraph(content_text, s_body_sm),
                Paragraph(sugg_text, s_body_sm),
                Paragraph(f'<b>{_esc(sev)}</b>', sev_style),
                Paragraph(notes_text, s_note),
            ]
        all_rows.append((row, sev))

    # Render table in chunks
    for chunk_start in range(0, max(len(all_rows), 1), batch_size):
        chunk = all_rows[chunk_start:chunk_start + batch_size]
        if not chunk:
            break
        data = [header]
        row_styles = [('BACKGROUND', (0, 0), (-1, 0), C_HEADER_BG)]

        for j, (row, sev) in enumerate(chunk):
            data.append(row)
            actual_row = j + 1
            if sev == '高':
                row_styles.append(('BACKGROUND', (0, actual_row), (-1, actual_row), C_LIGHT_RED))
            elif actual_row % 2 == 0:
                row_styles.append(('BACKGROUND', (0, actual_row), (-1, actual_row), C_ROW_ALT))

        t = Table(data, colWidths=col_w, repeatRows=1)
        t.setStyle(TableStyle(row_styles + [
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 2 * mm),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2 * mm),
            ('LEFTPADDING', (0, 0), (-1, -1), 1.5 * mm),
            ('RIGHTPADDING', (0, 0), (-1, -1), 1.5 * mm),
            ('GRID', (0, 0), (-1, -1), 0.4, C_BORDER),
        ]))
        els.append(t)
        if chunk_start + batch_size < len(all_rows):
            els.append(Spacer(1, 2 * mm))

    # Footer
    els.append(Spacer(1, 5 * mm))
    els.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=2 * mm))
    els.append(Paragraph(
        '备注：以上勘误基于AI对全册内容的逐页视觉审查，涵盖封面品牌/SKU、空格规范、编号格式、'
        '中英翻译对称性、标点符号、数学内容验算、图片素材、布局对齐、答案页格式、措辞一致性等维度。'
        '建议标记为"高"严重度的问题优先修正，标记为"需确认"的项目建议与教研团队核实后决定。', s_footer))

    if not errata_items:
        els.append(Spacer(1, 10 * mm))
        els.append(Paragraph('未发现问题，全册内容审查通过。',
                             _S('noerr', 12, C_GREEN, 1)))

    doc.build(els)
    return output_path


def _esc(text):
    """Escape text for reportlab Paragraph XML, preserving intentional HTML tags."""
    if not text:
        return ''
    # Replace & first, then < > but preserve common HTML tags
    text = text.replace('&', '&amp;')
    # We keep <b>, <br/>, <font> tags - replace other < >
    # Simple approach: just return as-is since our data is controlled
    return text
