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

## 核心原则（CN版教材）

- **CN版以中文为主体**：所有英文栏目标签必须改为中文，装饰性英文必须删除，不要建议"补充英文"
- **明确给出建议**：不要标注"需确认CN版规范"，直接给出明确的修改建议
- **答案必须验算**：不仅检查公式对错，还要逐一验算具体数字结果

## 审查维度（必须逐一检查每个维度）

### 一、封面与品牌 [必检·高优先级]
1. **封面品牌检查**：Logo是否为最新版本，产品名称是否正确（如"火花秘籍""专项练习册""Spark Math 培优课"），SKU码是否正确，是否有旧版元素需要删除
2. **版本标识**：版本号、印次信息是否正确

### 二、内容正确性
3. **数学内容验算**：计算答案是否正确（必须逐一验算具体数字，如"24个冰激凌""18颗樱桃"等），公式是否有误，题目条件是否自洽
4. **答案页核对**：答案与题目是否一一对应，答案数值是否正确，页码标注是否统一（统一使用"P6"格式而非"Page 6"或"Page6"）
5. **图文一致性**：文字描述与图片内容是否匹配，题目引用的物品是否存在于图中（如文字说"荔枝"但图片实为"树莓"）

### 三、分数与数学格式
6. **分数书写格式**：分数是否应写成上下分子分母竖式格式（而非行内"1/6"格式），检查全册分数表示方式是否统一规范

### 四、语言规范
7. **空格规范**：数字和汉字之间是否有空格（如"第 1 组"而非"第1组"，"由 1 个"而非"由1个"，"3 个苹果"而非"3个苹果"）
8. **标点符号**：中文是否使用全角标点（？而非?，。而非.），是否有重复标点（。。），句号/逗号是否缺少或多余
9. **编号规范**：小题编号格式是否统一（CN版统一使用(1)(2)(3)(4)而非(a)(b)(c)(d)），括号全角/半角是否一致

### 五、翻译与中英文
10. **翻译准确性**：中英文语义是否准确对应（如size=大小≠长度，each用于多个vs the用于单个）
11. **中英文对称性**：双语教材中英翻译是否完整对应，是否有一方遗漏关键信息
12. **栏目标签翻译**：所有英文栏目标签必须改为中文。标准对照：Preview→课前预习，Explore and Learn→新知探究，Example→例题，Practice→练习，Homework→课后巩固，Tackle Exam Questions→拓展练习，Notes→笔记，Basic→基础，Advanced→进阶，Challenge→挑战，Test→检验，Answers→参考答案

### 六、量词与措辞
13. **量词搭配**：量词是否正确（"块砖"而非"块砖块"，石柱用"根/个"而非"块"，冰激凌用"个"等），是否有量词重复
14. **措辞一致性**：同类题目（如例题和练习）措辞是否统一，"转化成"vs"转化为"不应混用
15. **翻译用语恰当性**：翻译是否自然通顺（"一堆砖块"围成圈→"一些砖块"，"一堆柱子"→"一些石柱"）

### 七、装饰性英文与多余元素 [必须逐页标注]
16. **装饰性英文删除**：CN版中每一处需要删除的英文必须逐页具体标注，包括：英文motto/标语、"Practice makes perfect"等英文装饰文字
17. **QR码/二维码**：所有QR码及"Scan for solution"文字必须标注删除，包括Homework页的二维码

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
- **高**：影响学生解题或理解的错误（答案错误、题目无法解答、关键翻译遗漏），封面品牌错误，分数格式系统性错误
- **中**：不影响解题但需要修正的问题（翻译不准确、措辞不一致、中英不对称、栏目标签未翻译、量词搭配错误）
- **低**：排版/格式问题（空格、标点、编号格式、装饰性英文）

重要：
- 不要遗漏任何问题，哪怕是很小的空格问题
- 系统性问题（如全册空格不规范）请合并为一条，标注影响范围
- 装饰性英文和QR码必须逐页具体标注，不要笼统概括
- 栏目标签翻译直接给出中文建议，不要标注"需确认"
- CN版策略是精简/删除英文，绝对不要建议"补充英文翻译"
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
        dict mapping item index -> JPEG bytes of the annotated full-page image
    """
    from PIL import Image, ImageDraw

    def _parse_page_num(page_str):
        """Extract first page number from various formats like 'P3', 'P3-P14(全册)', '全书系统性'.
        Returns None for system-wide items that need multi-page search."""
        nums = re.findall(r'\d+', page_str)
        return int(nums[0]) if nums else None

    def _get_search_candidates(item):
        """Build a prioritized list of search strings from the errata item."""
        candidates = []
        seen = set()

        def _add(text):
            t = text.strip() if text else ''
            if t and len(t) >= 2 and t not in seen:
                seen.add(t)
                candidates.append(t)

        # 1) error_text as-is
        et = item.get('error_text', '').strip()
        _add(et)

        # 2) Individual words from error_text (for multi-word phrases OCR can't match)
        if et and ' ' in et:
            for word in et.split():
                word = word.strip('.,;:!?()[]{}"\'"')
                if len(word) >= 3:
                    _add(word)

        # 3) Quoted strings from content_desc and suggestions
        for field in ['content_desc', 'suggestions', 'notes']:
            val = item.get(field, '')
            if isinstance(val, list):
                val = ' '.join(val)
            quoted = re.findall(r'["\u201c\u201d\u300c]([^"\u201c\u201d\u300d]+)["\u201c\u201d\u300d]', val)
            for q in quoted:
                _add(q)

        # 4) Location text
        _add(item.get('location', ''))

        # 5) Extract CJK/English fragments from content_desc as last resort
        desc = item.get('content_desc', '')
        # English words 4+ chars
        for w in re.findall(r'[A-Za-z]{4,}', desc):
            _add(w)
        # Chinese fragments 2+ chars
        for w in re.findall(r'[\u4e00-\u9fff]{2,}', desc):
            _add(w)

        return candidates

    # Group errors by page number; None = needs multi-page search
    page_errors = {}
    multipage_errors = []  # items that need searching across pages
    for idx, item in enumerate(errata_items):
        page_str = item.get('page', '')
        page_num = _parse_page_num(page_str)
        if page_num is None:
            multipage_errors.append((idx, item))
        else:
            page_errors.setdefault(page_num, []).append((idx, item))

    doc = fitz.open(pdf_path)
    scale = SCREENSHOT_DPI / 72
    mat = fitz.Matrix(scale, scale)
    screenshots = {}

    has_text = any(doc[i].get_text("text").strip() for i in range(min(3, len(doc))))
    ocr_cache = {}

    def _search_text(page, page_idx, text):
        """Search for exact text, then progressively shorter prefixes."""
        if not text or len(text) < 2:
            return []

        # Native text search
        if has_text:
            rects = page.search_for(text)
            if rects:
                return rects

        # OCR search
        if page_idx not in ocr_cache:
            try:
                ocr_cache[page_idx] = page.get_textpage_ocr(
                    language="eng+chi_sim", dpi=150
                )
            except Exception:
                ocr_cache[page_idx] = None

        tp = ocr_cache[page_idx]
        if not tp:
            return []

        rects = page.search_for(text, textpage=tp)
        if rects:
            return rects

        # Progressively shorter substrings
        if len(text) > 2:
            min_len = max(2, len(text) // 3)
            for length in range(len(text) - 1, min_len - 1, -1):
                sub = text[:length]
                rects = page.search_for(sub, textpage=tp)
                if rects:
                    return rects

        # Retry with higher DPI OCR as last resort
        hi_key = (page_idx, 'hi')
        if hi_key not in ocr_cache:
            try:
                ocr_cache[hi_key] = page.get_textpage_ocr(
                    language="eng+chi_sim", dpi=300
                )
            except Exception:
                ocr_cache[hi_key] = None
        tp_hi = ocr_cache[hi_key]
        if tp_hi:
            rects = page.search_for(text, textpage=tp_hi)
            if rects:
                return rects
            if len(text) > 2:
                for length in range(len(text) - 1, min_len - 1, -1):
                    rects = page.search_for(text[:length], textpage=tp_hi)
                    if rects:
                        return rects

        return []

    def _find_rects(page, page_idx, item):
        """Try multiple search strategies to find annotatable rects."""
        candidates = _get_search_candidates(item)
        for text in candidates:
            rects = _search_text(page, page_idx, text)
            if rects:
                return rects
        return []

    for page_num, errors in page_errors.items():
        page_idx = page_num - 1
        if page_idx < 0 or page_idx >= len(doc):
            continue

        page = doc[page_idx]

        # Pre-render the page once (all errors on same page share same base image)
        pix = page.get_pixmap(matrix=mat)
        page_png = pix.tobytes("png")

        for idx, item in errors:
            rects = _find_rects(page, page_idx, item)

            pil_img = Image.open(io.BytesIO(page_png)).convert("RGB")

            if rects:
                # Merge all rects into ONE bounding box
                union = rects[0]
                for r in rects[1:]:
                    union = union | r

                draw = ImageDraw.Draw(pil_img)
                x0 = int(union.x0 * scale) - 4
                y0 = int(union.y0 * scale) - 4
                x1 = int(union.x1 * scale) + 4
                y1 = int(union.y1 * scale) + 4
                for w in range(4):
                    draw.rectangle([x0 - w, y0 - w, x1 + w, y1 + w],
                                   outline=(255, 0, 0))

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            screenshots[idx] = buf.getvalue()

    # Handle system-wide items: search across pages to find the best match
    for idx, item in multipage_errors:
        candidates = _get_search_candidates(item)
        best_rects = None
        best_page_idx = None

        # Search pages 1..N (skip cover which is usually graphical)
        search_order = list(range(1, len(doc))) + [0]
        for pi in search_order:
            page = doc[pi]
            # Don't use cache here — get fresh OCR per page to avoid weak ref issues
            try:
                tp = page.get_textpage_ocr(language="eng+chi_sim", dpi=150)
            except Exception:
                tp = None
            for text in candidates:
                if not text or len(text) < 2:
                    continue
                rects = page.search_for(text, textpage=tp) if tp else page.search_for(text)
                if rects:
                    best_rects = rects
                    best_page_idx = pi
                    break
            if best_rects:
                break

        if best_page_idx is None:
            best_page_idx = 0  # fallback to cover

        page = doc[best_page_idx]
        pix = page.get_pixmap(matrix=mat)
        pil_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        if best_rects:
            union = best_rects[0]
            for r in best_rects[1:]:
                union = union | r
            draw = ImageDraw.Draw(pil_img)
            x0 = int(union.x0 * scale) - 4
            y0 = int(union.y0 * scale) - 4
            x1 = int(union.x1 * scale) + 4
            y1 = int(union.y1 * scale) + 4
            for w in range(4):
                draw.rectangle([x0 - w, y0 - w, x1 + w, y1 + w],
                               outline=(255, 0, 0))

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        screenshots[idx] = buf.getvalue()

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
