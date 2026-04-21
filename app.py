"""Flask web app for textbook errata checking."""

import os
import uuid
import json
import threading
import time
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_file, render_template, Response

from proofreader import proofread_pdf, COMPRESS_THRESHOLD
from pdf_report import generate_errata_pdf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

UPLOAD_DIR = Path(__file__).parent / 'uploads'
OUTPUT_DIR = Path(__file__).parent / 'outputs'
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory task store
tasks = {}


class Task:
    def __init__(self, task_id, files):
        self.id = task_id
        self.files = files  # list of {original_name, saved_path}
        self.status = 'pending'  # pending, processing, done, error
        self.progress = []  # list of {stage, pct, message, timestamp}
        self.results = []  # list of {filename, errata_count, report_path}
        self.error = None
        self.created_at = datetime.now()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    api_key = request.form.get('api_key', '').strip()
    if not api_key:
        api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        return jsonify({'error': '请提供 Anthropic API Key'}), 400

    model = request.form.get('model', 'claude-sonnet-4-5-20250514').strip()

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': '请上传至少一个PDF文件'}), 400

    task_id = str(uuid.uuid4())[:8]
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(exist_ok=True)

    saved_files = []
    for f in files:
        if f.filename and f.filename.lower().endswith('.pdf'):
            safe_name = f.filename
            save_path = task_dir / safe_name
            f.save(str(save_path))
            saved_files.append({
                'original_name': f.filename,
                'saved_path': str(save_path),
            })

    if not saved_files:
        return jsonify({'error': '未找到有效的PDF文件'}), 400

    task = Task(task_id, saved_files)
    tasks[task_id] = task

    # Start processing in background
    thread = threading.Thread(
        target=_process_task, args=(task, api_key, model), daemon=True
    )
    thread.start()

    return jsonify({
        'task_id': task_id,
        'file_count': len(saved_files),
        'files': [f['original_name'] for f in saved_files],
    })


@app.route('/status/<task_id>')
def status_stream(task_id):
    """SSE endpoint for task progress."""
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404

    def generate():
        last_idx = 0
        while True:
            # Send new progress items
            while last_idx < len(task.progress):
                item = task.progress[last_idx]
                data = json.dumps(item, ensure_ascii=False)
                yield f"data: {data}\n\n"
                last_idx += 1

            if task.status in ('done', 'error'):
                # Send final status
                final = {
                    'stage': 'final',
                    'status': task.status,
                    'results': task.results,
                    'error': task.error,
                }
                yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                break

            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/download/<task_id>/<filename>')
def download(task_id, filename):
    """Download generated errata PDF."""
    file_path = OUTPUT_DIR / task_id / filename
    if not file_path.exists():
        return jsonify({'error': '文件不存在'}), 404
    return send_file(str(file_path), as_attachment=True, download_name=filename)


def _process_task(task, api_key, model):
    """Background processing of uploaded PDFs."""
    task.status = 'processing'
    out_dir = OUTPUT_DIR / task.id
    out_dir.mkdir(exist_ok=True)

    total_files = len(task.files)

    for fi, file_info in enumerate(task.files):
        original_name = file_info['original_name']
        pdf_path = file_info['saved_path']
        file_prefix = f'[{fi + 1}/{total_files}] {original_name}'

        def progress_cb(stage, pct, message):
            # Scale progress per file
            overall_pct = int((fi * 100 + pct) / total_files)
            task.progress.append({
                'stage': stage,
                'pct': overall_pct,
                'message': f'{file_prefix}: {message}',
                'file_index': fi,
                'timestamp': time.time(),
            })

        try:
            result = proofread_pdf(pdf_path, api_key, progress_callback=progress_cb, model=model)

            # Generate report PDF
            stem = Path(original_name).stem
            report_name = f'勘误-{stem}.pdf'
            report_path = out_dir / report_name

            generate_errata_pdf(
                str(report_path),
                filename=stem,
                errata_items=result['errata_items'],
                total_pages=result['total_pages'],
            )

            task.results.append({
                'filename': report_name,
                'original_name': original_name,
                'errata_count': len(result['errata_items']),
                'total_pages': result['total_pages'],
                'compressed': result['compressed'],
                'download_url': f'/download/{task.id}/{report_name}',
            })

        except Exception as e:
            task.progress.append({
                'stage': 'error',
                'pct': int((fi + 1) * 100 / total_files),
                'message': f'{file_prefix}: 处理失败 - {str(e)[:200]}',
                'file_index': fi,
                'timestamp': time.time(),
            })
            task.results.append({
                'filename': None,
                'original_name': original_name,
                'errata_count': -1,
                'error': str(e)[:200],
            })

    # Cleanup uploaded files
    try:
        import shutil
        upload_dir = UPLOAD_DIR / task.id
        if upload_dir.exists():
            shutil.rmtree(str(upload_dir))
    except Exception:
        pass

    task.status = 'done'


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8899))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', '0') == '1')
