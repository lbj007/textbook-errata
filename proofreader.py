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

    # Clean up temp file
    if compressed and os.path.exists(working_path):
        os.unlink(working_path)

    if progress_callback:
        progress_callback('done', 100, f'审校完成，发现 {len(all_items)} 处问题')

    return {
        'filename': filename,
        'total_pages': total_pages,
        'errata_items': all_items,
        'compressed': compressed,
    }
