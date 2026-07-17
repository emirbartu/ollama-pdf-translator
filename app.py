import base64
import json
import os
import queue
import shutil
import tempfile
import threading
from pathlib import Path

import fitz
import uvicorn
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from main import translate_pdf

MAX_BYTES = 3 * 1024 * 1024
MAX_PAGES = 10
MAX_OUTPUT_BYTES = 3_200_000
MODEL = "gpt-oss:120b"

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Ollama PDF Translator")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if not os.environ.get("VERCEL"):
    app.mount("/static", StaticFiles(directory="public/static"), name="static")

templates = Jinja2Templates(directory="templates")
job_lock = threading.Lock()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/translate")
@limiter.limit("5/hour")
async def translate(
    request: Request,
    file: UploadFile,
    source_lang: str = Form("English"),
    target_lang: str = Form("Turkish"),
    preserve_layout: bool = Form(True),
):
    data = await file.read()
    error = _validate(data)
    if error:
        return HTMLResponse(_ndjson_error(error), media_type="application/x-ndjson")
    if not job_lock.acquire(blocking=False):
        return HTMLResponse(
            _ndjson_error("Another translation is in progress — please try again shortly."),
            media_type="application/x-ndjson",
        )
    stem = Path(file.filename or "document.pdf").stem
    return StreamingResponse(
        _stream_translation(data, stem, source_lang, target_lang, preserve_layout),
        media_type="application/x-ndjson",
    )


def _stream_translation(data: bytes, stem: str, source_lang: str, target_lang: str, preserve_layout: bool):
    q = queue.Queue()
    tmpdir = Path(tempfile.mkdtemp(prefix="pdfjob_"))

    def work():
        try:
            input_path = tmpdir / "input.pdf"
            output_path = tmpdir / "output.pdf"
            input_path.write_bytes(data)
            result = translate_pdf(
                input_path,
                output_path,
                source_lang=source_lang,
                target_lang=target_lang,
                model=MODEL,
                preserve_layout=preserve_layout,
                progress_callback=lambda k, n: q.put(("progress", k, n)),
            )
            q.put(("done", output_path) if result else ("error", None))
        except Exception:
            q.put(("error", None))

    threading.Thread(target=work, daemon=True).start()

    try:
        while True:
            kind, *rest = q.get()
            if kind == "progress":
                page, total = rest
                percent = min(round(page / total * 100), 100) if total else 0
                yield json.dumps({"type": "progress", "page": page, "total": total, "percent": percent}) + "\n"
            elif kind == "done":
                output_bytes = rest[0].read_bytes()
                if len(output_bytes) > MAX_OUTPUT_BYTES:
                    yield _ndjson_error("Translated PDF is too large to return.")
                else:
                    yield json.dumps({
                        "type": "done",
                        "filename": f"{stem}_translated.pdf",
                        "pdf_b64": base64.b64encode(output_bytes).decode(),
                    }) + "\n"
                return
            else:
                yield _ndjson_error("Translation failed. Please try again.")
                return
    finally:
        job_lock.release()
        shutil.rmtree(tmpdir, ignore_errors=True)


def _ndjson_error(message: str) -> str:
    return json.dumps({"type": "error", "message": message}) + "\n"


def _validate(data: bytes):
    if not data:
        return "No file received."
    if not data.startswith(b"%PDF-"):
        return "Only PDF files are supported."
    if len(data) > MAX_BYTES:
        return "File too large — 3 MB max."
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            if len(doc) > MAX_PAGES:
                return f"PDF too long — {MAX_PAGES} pages max for this demo."
    except Exception:
        return "Could not read this PDF."
    return None


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)
