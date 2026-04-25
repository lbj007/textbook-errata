"""Core proofreading logic: PDF compression, text/image extraction, AI review."""

import os
import io
import json
import re
import base64
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import anthropic


# Compression threshold: 10MB
COMPRESS_THRESHOLD = 10 * 1024 * 1024
# Page image DPI for vision
PAGE_DPI = 150
# DPI for screenshot annotations in the report
SCREENSHOT_DPI = 200
# Pages per batch for API calls
BATCH_SIZE = 4

PROOFREAD_SYSTEM_PROMPT = """你是一位拥有20年经验的资深教育出版物审校专家。你的任务是对教材/教辅PDF页面进行出版级别的全面审查。

## 审查维度（必须逐一检查每个维度）

1. **空格规范**：数字和汉字之间是否有空格（如"第 1 组"而非"第1组"，"由 1 个"而非"由1个"，"3 个苹果"而非"3个苹果"）
2. **编号规范**：小题编号格式是否统一（abc vs 123），括号全角/半角是否一致，标题是否有编号（例题1、练习1）
3. **中英文对称性**：双语教材中英翻译是否完整对应，是否有一方遗漏信息（如英文有"as a fraction"但中文漏译"分数"）
4. **标点符号**：中文是否使用全角标点（？而非?，。而非.），是否有重复标点（。。），句号/逗号是否缺少或多余
5. **数学内容**：计算答案是否正确，公式是否有误，题目条件是否自洽
6. **翻译准确性**：中英文语义是否准确对应（如size=大小≠长度，each用于多个vs the用于单个）
7. **图文一致性**：文字描述与图片内容是否匹配，题目引用的物品是否存在于图中
8. **措辞一致性**：同类题目（如例题和练习）措辞是否统一，"转化成"vs"转化为"不应混用
9. **答案页格式**：页码标注是否统一（"Page 6"vs"Page6"），答案与题目是否对应
10. **量词与用语**：是否有量词重复（"块砖块"→"块砖"），翻译是否恰当（"一堆"围成圈→"一圈"）
11. **品牌标识**：封面Logo、SKU码、版本号是否正确
12. **装饰性英文**：CN版教材中是否有需要翻译或删除的英文标语

## 输出格式要求

请以严格的JSON数组格式输出发现的所有问题，每个问题包含：
```json
[
  {
    "page": "P2",
    "location": "Example 1 题干",
    "content_desc": "问题的具体描述，引用原文",
    "error_text": "页面上能定位该错误的原文关键词或短语（用于文本搜索定位，越精确越好）",
    "suggestions": ["修改建议1", "修改建议2"],
    "severity": "高/中/低",
    "notes": "备注说明"
  }
]
```

严重程度标准：
- **高**：影响学生解题或理解的错误（答案错误、题目无法解答、关键翻译遗漏）
- **中**：不影响解题但需要修正的问题（翻译不准确、措辞不一致、中英不对称）
- **低**：排版/格式问题（空格、标点、编号格式）

重要：
- 不要遗漏任何问题，哪怕是很小的空格问题
- 系统性问题（如全册空格不规范）请合并为一条，标注影响范围
- 只输出JSON数组，不要包含其他文字
- error_text字段非常重要，请填入页面上与该错误最相关的原文文字片段（5-30字），用于在PDF中精确定位错误位置
- 如果没有发现问题，输出空数组 []"""


def compress_pdf(input_path: str, output_path: str = None) -> str:
    """Compress PDF using Ghostscript if available, otherwise return original."""
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.pdf', prefix='compressed_')

    try:
        subprocess.run([
            'gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
            '-dPDFSETTINGS=/ebook', '-dNOPAUSE', '-dQUIET', '-dBATCH',
            f'-sOutputFile={output_path}', input_path
        ], check=True, capture_output=True, timeout=120)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        return input_path


def extract_pages(pdf_path: str) -> list:
    """Extract page images and text from PDF.

    Returns list of dicts: {page_num, image_b64, text}
    """
    doc = fitz.open(pdf_path)
    pages = []

    for i in range(len(doc)):
        page = doc[i]

        # Render page to image
        mat = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
        pix = page.get_pixmap(matrix=mat)

        # Convert to JPEG bytes
        img_bytes = pix.tobytes("jpeg", jpg_quality=80)
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        # Extract text
        text = page.get_text("text")

        pages.append({
            'page_num': i + 1,
            'image_b64': img_b64,
            'text': text.strip(),
        })

    doc.close()
    return pages


def _call_claude_vision(client, pages_batch, filename, model="claude-sonnet-4-5-20250514"):
    """Send a batch of page images to Claude for proofreading."""
    content = []

    # Add context about which file and pages
    page_range = f"P{pages_batch[0]['page_num']}-P{pages_batch[-1]['page_num']}"
    content.append({
        "type": "text",
        "text": f"以下是教材《{filename}》的{page_range}页。请仔细审查每一页的所有问题。"
    })

    for p in pages_batch:
        # Add page image
        content.append({
            "type": "text",
            "text": f"\n--- 第 {p['page_num']} 页 ---"
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": p['image_b64'],
            }
        })
        # Add extracted text as supplement
        if p['text']:
            content.append({
                "type": "text",
                "text": f"[第{p['page_num']}页OCR文本辅助参考]\n{p['text'][:2000]}"
            })

    response = client.messages.create(
        model=model,
        max_tokens=8000,
        system=PROOFREAD_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    # Parse JSON from response
    response_text = response.content[0].text.strip()

    # Try to extract JSON from possible markdown code blocks
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(1).strip()

    try:
        items = json.loads(response_text)
        if isinstance(items, list):
            return items
    except json.JSONDecodeError:
        # Try to find JSON array in the text
        arr_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if arr_match:
            try:
                return json.loads(arr_match.group(0))
            except json.JSONDecodeError:
                pass

    return []


def generate_screenshots(pdf_path: str, errata_items: list) -> dict:
    """Generate annotated page screenshots with red boxes around error locations.

    Args:
        pdf_path: Path to the original PDF
        errata_items: List of errata dicts (must have 'page' and 'error_text')

    Returns:
        dict mapping item index -> JPEG bytes of the annotated page crop
    """
    # Group errors by page number
    page_errors = {}
    for idx, item in enumerate(errata_items):
        page_str = item.get('page', 'P0')
        nums = re.findall(r'\d+', page_str)
        if not nums:
            continue
        page_num = int(nums[0])
        page_errors.setdefault(page_num, []).append((idx, item))

    doc = fitz.open(pdf_path)
    screenshots = {}

    for page_num, errors in page_errors.items():
        page_idx = page_num - 1
        if page_idx < 0 or page_idx >= len(doc):
            continue

        page = doc[page_idx]

        for idx, item in errors:
            error_text = item.get('error_text', '')
            if not error_text:
                # Fall back: use first 15 chars of content_desc
                desc = item.get('content_desc', '')
                # Try to extract quoted text
                quoted = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]', desc)
                error_text = quoted[0] if quoted else desc[:15]

            if not error_text:
                # Generate full page screenshot without annotation
                mat = fitz.Matrix(SCREENSHOT_DPI / 72, SCREENSHOT_DPI / 72)
                pix = page.get_pixmap(matrix=mat)
                screenshots[idx] = pix.tobytes("jpeg", jpg_quality=85)
                continue

            # Search for the error text on the page
            rects = page.search_for(error_text)

            # If not found, try shorter substrings
            if not rects and len(error_text) > 6:
                for trim in range(2, len(error_text) // 2):
                    rects = page.search_for(error_text[:len(error_text) - trim])
                    if rects:
                        break

            if not rects:
                # Still not found — render full page
                mat = fitz.Matrix(SCREENSHOT_DPI / 72, SCREENSHOT_DPI / 72)
                pix = page.get_pixmap(matrix=mat)
                screenshots[idx] = pix.tobytes("jpeg", jpg_quality=85)
                continue

            # Merge all found rects into a bounding box with padding
            union = rects[0]
            for r in rects[1:]:
                union = union | r  # union of rects

            # Add padding (30pt each side)
            pad = 30
            clip = fitz.Rect(
                max(0, union.x0 - pad),
                max(0, union.y0 - pad),
                min(page.rect.width, union.x1 + pad),
                min(page.rect.height, union.y1 + pad),
            )

            # Ensure minimum height/width for readability
            min_dim = 80
            if clip.width < min_dim:
                cx = (clip.x0 + clip.x1) / 2
                clip.x0 = max(0, cx - min_dim / 2)
                clip.x1 = min(page.rect.width, cx + min_dim / 2)
            if clip.height < min_dim:
                cy = (clip.y0 + clip.y1) / 2
                clip.y0 = max(0, cy - min_dim / 2)
                clip.y1 = min(page.rect.height, cy + min_dim / 2)

            # Draw red rectangle annotations on a temp copy
            # We use a shape to draw on the page pixmap
            mat = fitz.Matrix(SCREENSHOT_DPI / 72, SCREENSHOT_DPI / 72)

            # Draw red rectangles on the page
            shape = page.new_shape()
            for r in rects:
                shape.draw_rect(r)
            shape.finish(color=(1, 0, 0), width=1.5, fill=None)
            shape.commit()

            # Render the clipped area
            pix = page.get_pixmap(matrix=mat, clip=clip)
            screenshots[idx] = pix.tobytes("jpeg", jpg_quality=85)

            # Remove annotations we just added (cleanup for next iteration)
            # Undo the drawing by re-cleaning the page
            page.clean_contents()

    doc.close()
    return screenshots


def proofread_pdf(pdf_path: str, api_key: str, progress_callback=None, model="claude-sonnet-4-5-20250514"):
    """Main proofreading function.

    Args:
        pdf_path: Path to PDF file
        api_key: Anthropic API key
        progress_callback: callable(stage, progress_pct, message)
        model: Claude model to use

    Returns:
        dict with keys: filename, total_pages, errata_items, compressed
    """
    filename = Path(pdf_path).stem
    file_size = os.path.getsize(pdf_path)
    compressed = False
    working_path = pdf_path

    # Step 1: Compress if needed
    if progress_callback:
        progress_callback('compress', 0, '检查文件大小...')

    if file_size > COMPRESS_THRESHOLD:
        if progress_callback:
            progress_callback('compress', 5, f'文件较大({file_size / 1024 / 1024:.1f}MB)，正在压缩...')
        working_path = compress_pdf(pdf_path)
        compressed = (working_path != pdf_path)
        if compressed and progress_callback:
            new_size = os.path.getsize(working_path)
            progress_callback('compress', 10, f'压缩完成: {file_size / 1024 / 1024:.1f}MB → {new_size / 1024 / 1024:.1f}MB')

    # Step 2: Extract pages
    if progress_callback:
        progress_callback('extract', 15, '正在提取页面图像和文本...')

    pages = extract_pages(working_path)
    total_pages = len(pages)

    if progress_callback:
        progress_callback('extract', 25, f'已提取 {total_pages} 页')

    # Step 3: Call AI in batches
    client = anthropic.Anthropic(api_key=api_key)
    all_items = []
    batches = [pages[i:i + BATCH_SIZE] for i in range(0, len(pages), BATCH_SIZE)]

    for bi, batch in enumerate(batches):
        pct = 25 + int(65 * (bi / len(batches)))
        page_range = f"P{batch[0]['page_num']}-P{batch[-1]['page_num']}"
        if progress_callback:
            progress_callback('proofread', pct, f'正在审校 {page_range}（第{bi + 1}/{len(batches)}批）...')

        try:
            items = _call_claude_vision(client, batch, filename, model=model)
            all_items.extend(items)
        except Exception as e:
            if progress_callback:
                progress_callback('error', pct, f'审校 {page_range} 出错: {str(e)[:100]}')

    # Step 4: Deduplicate and sort
    if progress_callback:
        progress_callback('finalize', 92, '正在整理勘误结果...')

    # Sort by page number
    def page_sort_key(item):
        page = item.get('page', 'P0')
        nums = re.findall(r'\d+', page)
        return int(nums[0]) if nums else 999

    all_items.sort(key=page_sort_key)

    # Step 5: Generate annotated screenshots
    screenshots = {}
    if all_items:
        if progress_callback:
            progress_callback('screenshot', 93, '正在生成错误截图标注...')
        try:
            screenshots = generate_screenshots(working_path, all_items)
            if progress_callback:
                progress_callback('screenshot', 97, f'已生成 {len(screenshots)} 张标注截图')
        except Exception as e:
            if progress_callback:
                progress_callback('screenshot', 97, f'截图生成部分失败: {str(e)[:100]}')

    # Clean up temp file
    if compressed and os.path.exists(working_path):
        os.unlink(working_path)

    if progress_callback:
        progress_callback('done', 100, f'审校完成，发现 {len(all_items)} 处问题')

    return {
        'filename': filename,
        'total_pages': total_pages,
        'errata_items': all_items,
        'screenshots': screenshots,
        'compressed': compressed,
    }
